#!/usr/bin/env python3
"""
Standalone evaluator for position probe (pos_probe).

Goal: evaluate pos_probe as a SPAN detector on answer tokens.
- Teacher-forced decode on saved full_ids (prompt+answer).
- Compute pos_score for each answer token (decode step).
- Convert scores -> predicted spans (thresholded runs).
- SINGLE-SPAN prediction: pick one predicted span by max(sum(scores)).
Reports span-level metrics: IoU, boundary errors, hit@end, early/miss/delay.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from baukit import TraceDict
from tqdm import tqdm

# Import LLMsKnow utilities
sys.path.append("/mnt/pcllzy_2/LLMsKnow/src")
from probing_utils import (  # noqa: E402
    MODEL_FRIENDLY_NAMES,
    load_model_and_validate_gpu,
    tokenize,
    find_prompt_end_in_full_ids,
    get_indices_of_exact_answer,
    get_probing_layer_names,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate pos_probe span localization (single-span) on teacher-forced decode.")
    p.add_argument("--dataset", type=str, default="triviaqa_test")
    p.add_argument("--model", type=str, default="/mnt/pcllzy/llama3-instruction-8b")
    p.add_argument("--pos_probe_path", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--balanced_eval", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)

    # Span extraction from score sequence
    p.add_argument("--thr", type=float, default=0.90, help="Threshold to binarize pos_score.")
    p.add_argument("--min_len", type=int, default=2, help="Drop predicted spans shorter than this length.")
    p.add_argument(
        "--pick",
        type=str,
        default="max_sum",
        choices=["max_sum", "longest", "first"],
        help="How to pick a SINGLE predicted span when multiple spans exist.",
    )
    # Optional hysteresis to build spans (more stable blocks)
    p.add_argument("--use_hysteresis", action="store_true", help="If set, build predicted span tokens via enter/exit hysteresis instead of single threshold.")
    p.add_argument("--enter_thr", type=float, default=0.90, help="Hysteresis: enter threshold.")
    p.add_argument("--exit_thr", type=float, default=0.80, help="Hysteresis: stay/exit threshold (leave span when score < exit_thr).")
    p.add_argument("--enter_k", type=int, default=2, help="Hysteresis: require k consecutive >= enter_thr to enter span.")
    p.add_argument("--dump_scores", type=str, default="", help="If set, dump per-sample token scores to this JSON file.")
    return p.parse_args()


@dataclass
class PosProbeBundle:
    clf: object
    pca: object | None
    scaler: object | None
    probe_at: str
    layer: int | None  # for mlp


def _load_pos_probe(path: str, *, num_layers: int) -> PosProbeBundle:
    d = joblib.load(path)
    probe_at = str(d.get("probe_at", "resid") or "resid")
    layer_raw = d.get("layer", None)
    layer = None
    if probe_at == "mlp":
        if layer_raw is None or str(layer_raw) == "last":
            layer = int(num_layers) - 1
        else:
            layer = int(layer_raw)
    return PosProbeBundle(
        clf=d["clf"],
        pca=d.get("pca", None),
        scaler=d.get("scaler", None),
        probe_at=probe_at,
        layer=layer,
    )


def _extract_runs(pred_1d: np.ndarray, *, min_len: int) -> list[tuple[int, int]]:
    """
    pred_1d: 0/1 array aligned to decode steps [1..T] (pred_1d[0] is step=1).
    returns list of (start_step, end_step), inclusive, 1-indexed.
    """
    runs: list[tuple[int, int]] = []
    i = 0
    while i < pred_1d.size:
        if int(pred_1d[i]) == 1:
            j = i
            while j + 1 < pred_1d.size and int(pred_1d[j + 1]) == 1:
                j += 1
            a = int(i + 1)
            b = int(j + 1)
            if (b - a + 1) >= int(min_len):
                runs.append((a, b))
            i = j + 1
        else:
            i += 1
    return runs


def _pick_single_span(
    spans: list[tuple[int, int]],
    scores_by_step: np.ndarray,
    *,
    method: str,
) -> tuple[int, int] | None:
    if not spans:
        return None
    if method == "first":
        # earliest span by start, then longest
        return min(spans, key=lambda ab: (ab[0], -(ab[1] - ab[0] + 1)))
    if method == "longest":
        return max(spans, key=lambda ab: (ab[1] - ab[0] + 1, ab[0]))
    # max_sum
    best = None
    best_sum = -1.0
    for (a, b) in spans:
        ssum = float(scores_by_step[a - 1 : b].sum())
        if ssum > best_sum:
            best_sum = ssum
            best = (a, b)
    return best


def _iou(ps: int, pe: int, gs: int, ge: int) -> float:
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    union = (pe - ps + 1) + (ge - gs + 1) - inter
    return float(inter / union) if union > 0 else 0.0


def _forward_step_pos_feat(
    model,
    *,
    step_input: torch.Tensor,  # [1,1]
    past_key_values,
    probe: PosProbeBundle,
    mlp_layer_name: str | None,
):
    with torch.no_grad():
        if probe.probe_at == "mlp":
            assert mlp_layer_name is not None
            with TraceDict(model, [mlp_layer_name], retain_input=False) as td:
                out = model(
                    input_ids=step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )
            feat = td[mlp_layer_name].output[0, -1, :].detach().cpu().float().numpy().reshape(1, -1)
            return out.past_key_values, feat
        else:
            out = model(
                input_ids=step_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            feat = out.hidden_states[-1][0, -1, :].detach().cpu().float().numpy().reshape(1, -1)
            return out.past_key_values, feat


def main():
    args = parse_args()
    model, tokenizer = load_model_and_validate_gpu(args.model)
    friendly = MODEL_FRIENDLY_NAMES[args.model]

    # Data
    llms_know_root = "/mnt/pcllzy_2/LLMsKnow"
    df = pd.read_csv(f"{llms_know_root}/output/{friendly}-answers-{args.dataset}.csv").copy()
    df["_orig_idx"] = df.index.astype(int)
    if args.max_samples > 0:
        if args.balanced_eval:
            df_pos = df[df["automatic_correctness"] == 1]
            df_neg = df[df["automatic_correctness"] == 0]
            half = max(1, int(args.max_samples // 2))
            n = min(half, len(df_pos), len(df_neg))
            df = (
                pd.concat([df_pos.sample(n=n, random_state=42), df_neg.sample(n=n, random_state=42)], axis=0)
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )
            print(f"📊 Using balanced subset: correct={n}, incorrect={n}, total={2*n}")
        else:
            df = df.sample(n=min(args.max_samples, len(df)), random_state=42).reset_index(drop=True)

    ids_all = torch.load(f"{llms_know_root}/output/{friendly}-input_output_ids-{args.dataset}.pt", map_location="cpu")

    # Probe
    probe = _load_pos_probe(args.pos_probe_path, num_layers=int(model.config.num_hidden_layers))
    mlp_layer_name = None
    if probe.probe_at == "mlp":
        names = get_probing_layer_names("mlp", model.config._name_or_path)
        assert probe.layer is not None and 0 <= int(probe.layer) < len(names)
        mlp_layer_name = names[int(probe.layer)]
        print(f"Using pos probe at MLP layer={int(probe.layer)} name={mlp_layer_name}")
    else:
        print("Using pos probe at resid (last hidden state)")

    # Metrics accumulators
    n_total = 0
    n_span_found = 0
    n_predicted = 0
    n_overlap = 0  # predicted spans that have IoU > 0 with GT
    ious = []
    start_errs = []
    end_errs = []
    hit_end_0 = 0
    hit_end_1 = 0
    early = 0
    miss = 0
    # first trigger hit: first token with score >= thr falls in GT span
    n_first_trigger = 0
    n_first_trigger_hit = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        n_total += 1
        orig_idx = int(row["_orig_idx"])
        full_ids_1d: torch.Tensor = ids_all[orig_idx]
        prompt = str(row["question"])
        exact_answer = str(row["exact_answer"]) if ("exact_answer" in row and not pd.isna(row["exact_answer"])) else ""
        valid_exact = int(row["valid_exact_answer"]) if ("valid_exact_answer" in row and not pd.isna(row["valid_exact_answer"])) else 0
        if valid_exact != 1 or not exact_answer or exact_answer == "NO ANSWER":
            continue

        prompt_ids_1d = tokenize(prompt, tokenizer, args.model)[0]
        q_len = find_prompt_end_in_full_ids(full_ids_1d, prompt_ids_1d, allow_bos_mismatch=True)
        output_ids_1d = full_ids_1d[int(q_len) :]
        span = get_indices_of_exact_answer(tokenizer, full_ids_1d, exact_answer, args.model, output_ids=output_ids_1d)
        if not span:
            continue
        # Convert to decode steps (1-indexed)
        span_steps = [int(t - int(q_len) + 1) for t in span if int(t) >= int(q_len)]
        if not span_steps:
            continue
        gs, ge = int(min(span_steps)), int(max(span_steps))
        n_span_found += 1

        # Teacher-forced decode to get per-step pos_score
        current_input = full_ids_1d[: int(q_len)].to(model.device).unsqueeze(0)
        with torch.no_grad():
            pre = model(input_ids=current_input, use_cache=True, output_hidden_states=False)
        past = pre.past_key_values

        gold_out = full_ids_1d[int(q_len) :].to(model.device)
        max_steps = min(int(args.max_new_tokens), int(gold_out.shape[0]))
        scores_by_step = []
        for step in range(max_steps):
            step_input = gold_out[step : step + 1].unsqueeze(0)  # [1,1]
            past, feat = _forward_step_pos_feat(
                model,
                step_input=step_input,
                past_key_values=past,
                probe=probe,
                mlp_layer_name=mlp_layer_name,
            )
            if probe.scaler is not None:
                feat = probe.scaler.transform(feat)
            if probe.pca is not None:
                feat = probe.pca.transform(feat)
            score = float(probe.clf.predict_proba(feat)[0, 1])
            scores_by_step.append(score)
        if not scores_by_step:
            continue

        scores_arr = np.asarray(scores_by_step, dtype=float)
        
        # First trigger hit: find first token with score >= thr, check if in GT span
        first_above = np.where(scores_arr >= float(args.thr))[0]
        if first_above.size > 0:
            first_step = int(first_above[0]) + 1  # 1-indexed decode step
            n_first_trigger += 1
            if gs <= first_step <= ge:
                n_first_trigger_hit += 1
        
        if not args.use_hysteresis:
            pred = (scores_arr >= float(args.thr)).astype(np.int32)
        else:
            # Build pred via hysteresis: enter after k consecutive >= enter_thr; stay while >= exit_thr.
            enter_thr = float(args.enter_thr)
            exit_thr = float(args.exit_thr)
            k = int(max(1, args.enter_k))
            pred = np.zeros_like(scores_arr, dtype=np.int32)
            in_span = False
            consec = 0
            for i, s in enumerate(scores_arr.tolist(), start=1):  # i is 1-indexed step
                if not in_span:
                    if float(s) >= enter_thr:
                        consec += 1
                    else:
                        consec = 0
                    if consec >= k:
                        # mark the last k steps as inside span
                        start_i = max(1, i - k + 1)
                        pred[start_i - 1 : i] = 1
                        in_span = True
                        consec = 0
                else:
                    if float(s) >= exit_thr:
                        pred[i - 1] = 1
                    else:
                        in_span = False
                        consec = 0
        spans_pred = _extract_runs(pred, min_len=int(args.min_len))
        best = _pick_single_span(spans_pred, scores_arr, method=str(args.pick))
        if best is None:
            miss += 1
            continue
        n_predicted += 1
        ps, pe = best

        iou_val = _iou(ps, pe, gs, ge)
        ious.append(iou_val)
        if iou_val > 0:
            n_overlap += 1
        start_errs.append(float(ps - gs))
        end_errs.append(float(pe - ge))
        if pe == ge:
            hit_end_0 += 1
        if abs(pe - ge) <= 1:
            hit_end_1 += 1
        if ps < gs:
            early += 1

    # Summary
    print("\n=== Pos probe SINGLE-SPAN span-level evaluation ===")
    print(f"dataset={args.dataset} n_total={n_total} (span_found={n_span_found})")
    if args.use_hysteresis:
        print(f"hysteresis enter_thr={float(args.enter_thr):.2f} exit_thr={float(args.exit_thr):.2f} k={int(args.enter_k)} min_len={int(args.min_len)} pick={args.pick}")
    else:
        print(f"thr={float(args.thr):.2f} min_len={int(args.min_len)} pick={args.pick}")
    if n_span_found == 0:
        print("No GT spans found; nothing to report.")
        return
    print(f"predicted={n_predicted}/{n_span_found} miss_rate={miss/max(1,n_span_found):.3f}")
    print(f"overlap_hit_rate (IoU>0 among predicted): {n_overlap}/{n_predicted} = {n_overlap/max(1,n_predicted):.3f}")
    print(f"first_trigger_hit_rate (1st token>=thr in GT): {n_first_trigger_hit}/{n_first_trigger} = {n_first_trigger_hit/max(1,n_first_trigger):.3f}")
    if n_predicted == 0:
        print("No predicted spans; nothing to report.")
        return

    iou_arr = np.asarray(ious, dtype=float)
    se_arr = np.asarray(start_errs, dtype=float)
    ee_arr = np.asarray(end_errs, dtype=float)

    def q(arr):
        return {k: float(np.quantile(arr, k)) for k in [0.1, 0.5, 0.9]}

    # Overall-on-hit == same here because we only store metrics for predicted samples
    print(f"IoU_on_hit: mean={float(iou_arr.mean()):.3f} median={float(np.median(iou_arr)):.3f} q10/50/90={q(iou_arr)}")
    print(f"start_err_on_hit: mean={float(se_arr.mean()):.3f} median={float(np.median(se_arr)):.3f} q10/50/90={q(se_arr)}")
    print(f"end_err_on_hit:   mean={float(ee_arr.mean()):.3f} median={float(np.median(ee_arr)):.3f} q10/50/90={q(ee_arr)}")
    print(f"hit@end: exact={hit_end_0/n_predicted:.3f}  ±1={hit_end_1/n_predicted:.3f} (n={n_predicted})")
    print(f"early_rate (pred_start < gt_start): {early/n_predicted:.3f}")


if __name__ == "__main__":
    main()


