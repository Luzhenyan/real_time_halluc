#!/usr/bin/env python3
import argparse
import os
import torch
import numpy as np
import pandas as pd
import joblib
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict

# 导入 LLMsKnow 中的工具
import sys
sys.path.append('/mnt/pcllzy_2/LLMsKnow/src')
from probing_utils import (
    MODEL_FRIENDLY_NAMES,
    load_model_and_validate_gpu,
    get_indices_of_exact_answer,
    get_probing_layer_names,
    tokenize,
    find_prompt_end_in_full_ids,
)

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Real-time Hallucination Detection")
    parser.add_argument('--dataset', type=str, default='triviaqa_test')
    parser.add_argument('--model', type=str, default='/mnt/pcllzy/llama3-instruction-8b')
    parser.add_argument('--pos_probe_path', type=str, required=True, help="Path to Position Probe (.joblib)")
    parser.add_argument(
        '--pos_probe_at',
        type=str,
        default='resid',
        choices=['resid', 'mlp'],
        help="Feature source for pos probe: resid uses last-layer hidden state; mlp uses traced MLP output (see --pos_probe_layer).",
    )
    parser.add_argument(
        '--pos_probe_layer',
        type=int,
        default=None,
        help="When --pos_probe_at mlp: which layer index's MLP output to feed into pos probe (default: last layer).",
    )
    parser.add_argument('--hallu_probe_base', type=str, required=True, help="Base path for Hallucination Probes (checkpoints/...)")
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--stop_threshold', type=float, default=0.6)
    parser.add_argument('--pos_threshold', type=float, default=0.5)
    parser.add_argument(
        '--pos_trigger_mode',
        type=str,
        default='threshold',
        choices=['threshold', 'exit_span'],
        help="How pos_probe triggers decode-stage hallucination probing. "
             "'threshold': trigger on any step with pos_score>=pos_threshold (legacy). "
             "'exit_span': treat high pos_score as being inside answer span; trigger when leaving span, "
             "using cached features from the last high-scoring step (aims to approximate exact_answer_last_token).",
    )
    parser.add_argument('--pos_enter_threshold', type=float, default=0.90, help="For pos_trigger_mode=exit_span: threshold to enter span.")
    parser.add_argument('--pos_exit_threshold', type=float, default=0.80, help="For pos_trigger_mode=exit_span: threshold to stay in span; falling below triggers exit.")
    parser.add_argument('--pos_enter_k', type=int, default=2, help="For pos_trigger_mode=exit_span: require k consecutive >= enter_threshold to enter span.")
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--layers', type=int, nargs='+', default=None, help="Specific layers to use for detection (e.g., 10 15 20)")
    parser.add_argument(
        '--prefill_layers',
        type=int,
        nargs='+',
        default=None,
        help="Optional override: which layers to use for prefill probes (default: --layers or 10..last).",
    )
    parser.add_argument(
        '--decode_layers',
        type=int,
        nargs='+',
        default=None,
        help="Optional override: which layers to use for decode probes (default: --layers or 10..last). "
             "Example: --decode_layers 31",
    )
    parser.add_argument(
        '--prefill_only',
        action='store_true',
        help="If set, only evaluate prefill probes (last_q_token). Skip decode stage and position probe gating.",
    )
    parser.add_argument(
        '--decode_only',
        action='store_true',
        help="If set, only evaluate decode probes (exact_answer_last_token). Skip prefill early-exit and do not use prefill probes for decision.",
    )
    parser.add_argument(
        '--diagnose_decode',
        action='store_true',
        help="If set, compute exact_answer_last_token position from saved full_ids and report position-probe alignment + decode-probe-at-target AUROC. "
             "Also reports streaming key-token detection metrics for pos probe (token-level precision/recall, delay, early/miss rates).",
    )
    parser.add_argument(
        '--pos_score_dist',
        action='store_true',
        help="If set (recommended with --teacher_force_decode), collect token-level pos_probe score distributions on answer tokens: "
             "positives = exact_answer span tokens, negatives = other answer tokens. Prints quantiles and a small threshold sweep.",
    )
    parser.add_argument(
        '--pos_span_eval',
        action='store_true',
        help="If set, compute SINGLE-SPAN (contiguous segment) localization metrics for pos probe on the answer tokens. "
             "We threshold pos_score to get positive runs, then pick one span by max(sum of scores). "
             "Reports IoU, boundary errors, and hit@k for span end.",
    )
    parser.add_argument('--pos_span_threshold', type=float, default=0.90, help="For --pos_span_eval: threshold to binarize pos_score into predicted span tokens.")
    parser.add_argument('--pos_span_min_len', type=int, default=1, help="For --pos_span_eval: drop predicted spans shorter than this length.")
    parser.add_argument(
        '--teacher_force_decode',
        action='store_true',
        help="If set, in decode stage feed the saved answer tokens from full_ids (teacher forcing) instead of free-running generation.",
    )
    parser.add_argument(
        '--balanced_eval',
        action='store_true',
        help="If set, sample a balanced subset by `automatic_correctness` (equal #correct and #incorrect).",
    )
    parser.add_argument(
        '--score_layer_min',
        type=int,
        default=None,
        help="If set, only use layers >= this value when aggregating hallu_score for AUROC.",
    )
    parser.add_argument(
        '--score_agg',
        type=str,
        default='max',
        choices=['max', 'mean', 'topk_mean'],
        help="Aggregation used to compute hallu_score from per-layer halluc_prob values.",
    )
    parser.add_argument(
        '--score_top_k',
        type=int,
        default=3,
        help="For score_agg=topk_mean: average of top-k halluc_prob values across selected layers.",
    )
    parser.add_argument(
        '--auc_sweep',
        action='store_true',
        help="If set, print AUROC for several aggregations without rerunning the model.",
    )
    return parser.parse_args()

def load_hallu_probes_dict(base_path, layers, token_type):
    """加载多个层的探针"""
    probes = {}
    for l in layers:
        # 尝试几种可能的命名模式
        paths_to_try = [
            f"{base_path}_prefill_seed-0_layer-{l}_token-{token_type}.pkl",
            f"{base_path}_seed-0_layer-{l}_token-{token_type}.pkl",
            # legacy (prefer new prefill probes when present)
            f"{base_path}_layer-{l}_token-{token_type}.pkl",
        ]
        found = False
        for path in paths_to_try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    probes[l] = pickle.load(f)
                    print(f"✅ Loaded {token_type} probe for layer {l} from {os.path.basename(path)}")
                found = True
                break
        if not found:
            print(f"⚠️ Warning: No {token_type} probe found for layer {l}")
    return probes

def forward_with_mlp_traces(model, input_ids, *, past_key_values=None, use_cache=True, output_hidden_states=False, layer_indices=None):
    """
    Forward pass while tracing per-layer MLP outputs for specified layers.
    Returns: (outputs, mlp_outputs_dict[layer_idx] -> Tensor[B, T, H] on CPU)
    """
    mlp_layer_names = get_probing_layer_names('mlp', model.config._name_or_path)
    if layer_indices is not None and len(layer_indices) == 0:
        # Fast path: no MLP layers requested; just run forward.
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
            )
        return outputs, {}
    if layer_indices is None:
        layer_indices = list(range(len(mlp_layer_names)))
    trace_names = [mlp_layer_names[int(i)] for i in layer_indices]

    with torch.no_grad():
        with TraceDict(model, trace_names, retain_input=False) as ret:
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
            )

    mlp_out = {}
    for li, name in zip(layer_indices, trace_names):
        # ret[name].output: [B, T, H]
        mlp_out[int(li)] = ret[name].output.detach().cpu()
    return outputs, mlp_out


def run_e2e_realtime_eval(args):
    if args.prefill_only and args.decode_only:
        raise ValueError("Cannot set both --prefill_only and --decode_only.")

    model, tokenizer = load_model_and_validate_gpu(args.model)
    friendly = MODEL_FRIENDLY_NAMES[args.model]
    
    # 确定探测层（允许 prefill / decode 分别指定）
    num_layers = model.config.num_hidden_layers
    default_layers = list(range(10, num_layers))
    base_layers = sorted(args.layers) if args.layers else default_layers
    prefill_layers = sorted(args.prefill_layers) if args.prefill_layers else base_layers
    decode_layers = sorted(args.decode_layers) if args.decode_layers else base_layers
    if args.decode_only:
        prefill_layers = []
    if args.prefill_only:
        decode_layers = []
    detect_layers = sorted(set(prefill_layers) | set(decode_layers))

    # 加载位置探测器（prefill_only 时不需要；decode_only 需要）
    pos_clf, pos_pca = None, None
    pos_probe_at = str(getattr(args, "pos_probe_at", "resid"))
    pos_probe_layer = None
    if not args.prefill_only:
        pos_data = joblib.load(args.pos_probe_path)
        pos_clf, pos_pca = pos_data['clf'], pos_data.get('pca', None)
        pos_scaler = pos_data.get('scaler', None) if isinstance(pos_data, dict) else None
        # Probe file can declare expected feature source
        if isinstance(pos_data, dict) and ("probe_at" in pos_data) and (pos_data["probe_at"] is not None):
            pos_probe_at = str(pos_data["probe_at"])
        if pos_probe_at == "mlp":
            if getattr(args, "pos_probe_layer", None) is None:
                # Prefer layer recorded in probe file (if present), otherwise default to last layer.
                recorded = pos_data.get("layer", None) if isinstance(pos_data, dict) else None
                if recorded is not None:
                    try:
                        if str(recorded) == "last":
                            args.pos_probe_layer = int(model.config.num_hidden_layers) - 1
                        else:
                            args.pos_probe_layer = int(recorded)
                    except Exception:
                        args.pos_probe_layer = int(model.config.num_hidden_layers) - 1
                else:
                    args.pos_probe_layer = int(model.config.num_hidden_layers) - 1
            pos_probe_layer = int(args.pos_probe_layer)
    
    # 加载指定的探测器
    prefill_probes = load_hallu_probes_dict(args.hallu_probe_base, prefill_layers, "last_q_token")
    decode_probes = load_hallu_probes_dict(args.hallu_probe_base, decode_layers, "exact_answer_last_token")

    # 准备数据
    llms_know_root = '/mnt/pcllzy_2/LLMsKnow'
    df = pd.read_csv(f'{llms_know_root}/output/{friendly}-answers-{args.dataset}.csv')
    # IMPORTANT: keep original row index for alignment with `ids_all` (which is saved in original order)
    df = df.copy()
    df["_orig_idx"] = df.index.astype(int)
    if args.max_samples > 0:
        if args.balanced_eval:
            # automatic_correctness: 1=correct, 0=incorrect(hallu)
            df_pos = df[df["automatic_correctness"] == 1]
            df_neg = df[df["automatic_correctness"] == 0]
            half = max(1, int(args.max_samples // 2))
            n = min(half, len(df_pos), len(df_neg))
            df = pd.concat(
                [
                    df_pos.sample(n=n, random_state=42),
                    df_neg.sample(n=n, random_state=42),
                ],
                axis=0,
            ).sample(frac=1.0, random_state=42)
            print(f"📊 Using balanced eval subset: correct={n}, incorrect={n}, total={2*n}")
        else:
            df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
    
    ids_file = f'{llms_know_root}/output/{friendly}-input_output_ids-{args.dataset}.pt'
    ids_all = torch.load(ids_file, map_location='cpu')
    
    results = []
    # token-level pos score distribution (across all samples; only where span is found)
    pos_scores_pos = []
    pos_scores_neg = []
    # span-level metrics for pos probe (single predicted span per sample)
    span_iou_list = []
    span_start_err_list = []
    span_end_err_list = []
    span_hit_end_0 = 0
    span_hit_end_1 = 0
    span_found = 0
    span_predicted = 0
    
    mode = "prefill-only" if args.prefill_only else ("decode-only" if args.decode_only else "e2e")
    layers_msg = decode_layers if args.decode_only else (prefill_layers if args.prefill_only else detect_layers)
    print(f"\n🚀 Starting {mode} evaluation on {len(df)} samples using layers: {layers_msg}")
    
    def aggregate_score(layer_to_prob: dict, *, agg: str, top_k: int, layer_min: int | None):
        items = [(int(l), float(p)) for l, p in layer_to_prob.items()]
        if layer_min is not None:
            items = [(l, p) for (l, p) in items if l >= int(layer_min)]
        vals = [p for _, p in items]
        if not vals:
            return 0.0
        if agg == 'max':
            return float(max(vals))
        if agg == 'mean':
            return float(np.mean(vals))
        if agg == 'topk_mean':
            k = max(1, int(top_k))
            top = sorted(vals, reverse=True)[:k]
            return float(np.mean(top))
        return float(max(vals))

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            orig_idx = int(row["_orig_idx"]) if ("_orig_idx" in row and not pd.isna(row["_orig_idx"])) else None
            prompt = str(row['question'])
            true_label = row['automatic_correctness']
            exact_answer = str(row["exact_answer"]) if ("exact_answer" in row and not pd.isna(row["exact_answer"])) else None
            valid_exact = int(row["valid_exact_answer"]) if ("valid_exact_answer" in row and not pd.isna(row["valid_exact_answer"])) else 0
            
            if orig_idx is None:
                continue
            full_ids_1d = ids_all[int(orig_idx)]
            prompt_ids_1d = tokenize(prompt, tokenizer, args.model)[0]
            q_len = find_prompt_end_in_full_ids(full_ids_1d, prompt_ids_1d, allow_bos_mismatch=True)
            input_ids = full_ids_1d.to(model.device).unsqueeze(0)
            
            current_input = input_ids[:, :q_len]
            generated = current_input
            past_key_values = None
            
            status = "proceed"
            trigger_info = {}
            # Continuous score for ROC-AUC: max halluc_prob seen during prefill/decode
            max_hallu_prob = 0.0
            prefill_layer_probs = {}
            decode_layer_probs = {}
            max_decode_hallu_prob = 0.0
            saw_pos_trigger = False
            target_span = []
            target_full_idx = None
            expected_decode_step = None  # note: decode loop uses step=0 on last prompt token, so output token t => step=t+1
            span_decode_steps = None
            span_start_step = None
            span_end_step = None
            max_pos_score = 0.0
            max_pos_step = None
            pos_score_at_expected = None
            # streaming pos-probe diagnostics (token-level)
            first_pos_trigger_step = None
            pos_tp = 0
            pos_fp = 0
            pos_fn = 0
            # pos trigger state machine (for exit_span)
            in_span = False
            consec_enter = 0
            last_high_step = None
            last_high_pos_score = None
            last_high_feats = None  # dict[int(layer) -> np.ndarray(1,H)] in decode feature space (after scaler/pca if needed)
            triggered_at_step = None
            decode_layer_probs_at_target = {}
            hallu_score_decode_at_target = None
            # Span-based decode probe evaluation (token-level within exact_answer span, aggregated to sample-level)
            decode_span_scores = []
            # store pos_score by decode step (1-indexed) to support span-level localization evaluation
            pos_scores_by_step = []
            
            # --- Prefill Stage ---
            outputs, mlp_out = forward_with_mlp_traces(
                model,
                current_input,
                past_key_values=None,
                use_cache=True,
                output_hidden_states=True,  # still needed for decode-stage residual/pos probe if we proceed
                layer_indices=detect_layers,
            )
            past_key_values = outputs.past_key_values
                
            # Diagnose: compute exact_answer_last_token position from saved full_ids (no re-tokenization of prompt)
            if args.diagnose_decode and valid_exact == 1 and exact_answer is not None and isinstance(exact_answer, str) and len(exact_answer.strip()) > 0:
                try:
                    # output_ids boundary is q_len; this avoids prompt re-tokenization inside get_indices_of_exact_answer
                    output_ids_1d = full_ids_1d[int(q_len):]
                    target_span = get_indices_of_exact_answer(tokenizer, full_ids_1d, exact_answer, args.model, output_ids=output_ids_1d)
                    if target_span:
                        target_full_idx = int(target_span[-1])
                        if target_full_idx >= int(q_len):
                            expected_decode_step = int(target_full_idx - int(q_len) + 1)
                        # span steps in decode coordinate (1-indexed decode step)
                        span_decode_steps = []
                        for ti in target_span:
                            t = int(ti)
                            if t >= int(q_len):
                                span_decode_steps.append(int(t - int(q_len) + 1))
                        span_decode_steps = sorted(set(span_decode_steps)) if span_decode_steps else None
                        if span_decode_steps:
                            span_start_step = int(min(span_decode_steps))
                            span_end_step = int(max(span_decode_steps))
                except Exception:
                    target_span = []
                    target_full_idx = None
                    expected_decode_step = None
                    span_decode_steps = None
                    span_start_step = None
                    span_end_step = None

            # prefill probes (only used when not decode-only)
            if not args.decode_only:
                for l in prefill_layers:
                    if l in prefill_probes and l in mlp_out:
                        hs = mlp_out[l][0, -1, :].float().numpy()
                        probe_obj = prefill_probes[l]
                        if isinstance(probe_obj, dict) and 'scaler' in probe_obj:
                            hs_scaled = probe_obj['scaler'].transform(hs.reshape(1, -1))
                            prob_right = probe_obj['clf'].predict_proba(hs_scaled)[0, 1]
                        else:
                            clf = probe_obj['clf'] if isinstance(probe_obj, dict) else probe_obj
                            prob_right = clf.predict_proba(hs.reshape(1, -1))[0, 1]
                            
                        halluc_prob = float(1.0 - prob_right)
                        if halluc_prob > max_hallu_prob:
                            max_hallu_prob = halluc_prob
                        prefill_layer_probs[int(l)] = halluc_prob

                        if halluc_prob >= args.stop_threshold:
                            status = "early_exit_prefill"
                            trigger_info = {"step": 0, "prob": halluc_prob, "layer": l}
                            break

            # --- Decode Stage ---
            if (not args.prefill_only) and status == "proceed":
                gold_output = None
                if args.teacher_force_decode:
                    # Saved output tokens (prompt is input_ids[:q_len])
                    gold_output = input_ids[:, int(q_len):]
                max_steps = args.max_new_tokens
                if gold_output is not None:
                    max_steps = min(int(max_steps), int(gold_output.shape[1]))

                for step in range(max_steps):
                    if gold_output is not None:
                        step_input = gold_output[:, step: step + 1]
                    else:
                        step_input = generated[:, -1:]
                    # Ensure we trace MLP outputs for decode probes AND (optionally) pos probe layer.
                    trace_layers = list(decode_layers)
                    if (pos_probe_at == "mlp") and (pos_probe_layer is not None) and (int(pos_probe_layer) not in trace_layers):
                        trace_layers.append(int(pos_probe_layer))
                        out, mlp_out_step = forward_with_mlp_traces(
                            model,
                            step_input,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_hidden_states=True,
                            layer_indices=trace_layers,
                        )
                        past_key_values = out.past_key_values
                        
                        hs_resid = out.hidden_states[-1][0, -1, :].cpu().float().numpy()
                        
                        # 1. 位置探测
                    if pos_probe_at == "mlp":
                        if pos_probe_layer is None or int(pos_probe_layer) not in mlp_out_step:
                            # Fallback to resid if MLP not available (shouldn't happen unless misconfigured)
                            pos_feat = hs_resid.reshape(1, -1)
                        else:
                            pos_feat = mlp_out_step[int(pos_probe_layer)][0, -1, :].cpu().float().numpy().reshape(1, -1)
                    else:
                        pos_feat = hs_resid.reshape(1, -1)
                    if pos_scaler is not None:
                        pos_feat = pos_scaler.transform(pos_feat)
                        if pos_pca: pos_feat = pos_pca.transform(pos_feat)
                        pos_score = pos_clf.predict_proba(pos_feat)[0, 1]
                    # record for span-level evaluation (only meaningful on answer tokens / decode steps)
                    if args.pos_span_eval:
                        pos_scores_by_step.append(float(pos_score))
                    if float(pos_score) > float(max_pos_score):
                        max_pos_score = float(pos_score)
                        max_pos_step = int(step + 1)
                    if expected_decode_step is not None and int(step + 1) == int(expected_decode_step):
                        pos_score_at_expected = float(pos_score)

                    # --- Pos trigger logic ---
                    should_trigger_now = False
                    trigger_use_feats = None
                    trigger_step_for_info = int(step + 1)
                    trigger_pos_score_for_info = float(pos_score)

                    if args.pos_trigger_mode == "threshold":
                        # Legacy: trigger immediately on threshold
                        if float(pos_score) >= float(args.pos_threshold):
                            should_trigger_now = True
                            trigger_use_feats = None  # use current step's mlp_out_step
                    else:
                        # exit_span: interpret high score as "inside span"
                        enter_thr = float(args.pos_enter_threshold)
                        exit_thr = float(args.pos_exit_threshold)
                        k = int(max(1, args.pos_enter_k))

                        if not in_span:
                            if float(pos_score) >= enter_thr:
                                consec_enter += 1
                            else:
                                consec_enter = 0
                            if consec_enter >= k:
                                in_span = True
                                consec_enter = 0

                        if in_span:
                            if float(pos_score) >= exit_thr:
                                # update last-high snapshot (we'll trigger on exit using this token)
                                last_high_step = int(step + 1)
                                last_high_pos_score = float(pos_score)
                                # cache decode-layer features from this step
                                snap = {}
                                for l in decode_layers:
                                    if l in mlp_out_step:
                                        f = mlp_out_step[int(l)][0, -1, :].cpu().float().numpy().reshape(1, -1)
                                        # apply per-probe scaler if present (decode probes store their own scalers; for caching we keep raw)
                                        snap[int(l)] = f
                                last_high_feats = snap if snap else None
                            else:
                                # exit detected: trigger once using last-high snapshot
                                if triggered_at_step is None and last_high_step is not None and last_high_feats is not None:
                                    should_trigger_now = True
                                    triggered_at_step = int(last_high_step)
                                    trigger_step_for_info = int(last_high_step)
                                    trigger_pos_score_for_info = float(last_high_pos_score) if last_high_pos_score is not None else float(pos_score)
                                    trigger_use_feats = dict(last_high_feats)
                                # reset state after exit
                                in_span = False
                                consec_enter = 0
                                last_high_step = None
                                last_high_pos_score = None
                                last_high_feats = None

                    # Streaming key-token detection metrics (mode-aware)
                    # NOTE: we now evaluate against the exact-answer span (not only the last token).
                    if args.diagnose_decode and span_decode_steps is not None:
                        if args.pos_trigger_mode == "threshold":
                            # token-level: positives are ANY token in the exact-answer span.
                            is_target_step = (int(step + 1) in set(span_decode_steps))
                            is_pred_key = (float(pos_score) >= float(args.pos_threshold))
                            if is_pred_key and first_pos_trigger_step is None:
                                first_pos_trigger_step = int(step + 1)
                            if is_pred_key and is_target_step:
                                pos_tp += 1
                            elif is_pred_key and (not is_target_step):
                                pos_fp += 1
                            elif (not is_pred_key) and is_target_step:
                                pos_fn += 1
                        else:
                            # sample-level: first_pos_trigger_step is the triggered step (if any)
                            if should_trigger_now and first_pos_trigger_step is None:
                                first_pos_trigger_step = int(trigger_step_for_info)

                    # Optional: collect token-level score distributions (span tokens vs other answer tokens)
                    if args.pos_score_dist and span_decode_steps is not None:
                        is_pos_tok = int(step + 1) in set(span_decode_steps)
                        if is_pos_tok:
                            pos_scores_pos.append(float(pos_score))
                        else:
                            pos_scores_neg.append(float(pos_score))
                        
                    # Diagnose decode probes at the *true* target token regardless of pos gating.
                    if args.diagnose_decode and expected_decode_step is not None and int(step + 1) == int(expected_decode_step):
                        for l in decode_layers:
                            if l in decode_probes and l in mlp_out_step:
                                hs_mlp = mlp_out_step[l][0, -1, :].float().numpy()
                                probe_obj = decode_probes[l]
                                if isinstance(probe_obj, dict) and 'scaler' in probe_obj:
                                    hs_scaled = probe_obj['scaler'].transform(hs_mlp.reshape(1, -1))
                                    prob_right = probe_obj['clf'].predict_proba(hs_scaled)[0, 1]
                                else:
                                    clf = probe_obj['clf'] if isinstance(probe_obj, dict) else probe_obj
                                    prob_right = clf.predict_proba(hs_mlp.reshape(1, -1))[0, 1]
                                decode_layer_probs_at_target[int(l)] = float(1.0 - prob_right)

                    # Span-based decode probe scoring: evaluate hallu probes on EVERY token in the exact_answer span.
                    if args.diagnose_decode and span_decode_steps is not None and int(step + 1) in set(span_decode_steps):
                        step_layer_probs = {}
                        for l in decode_layers:
                            if l in decode_probes and l in mlp_out_step:
                                hs_mlp = mlp_out_step[l][0, -1, :].float().numpy()
                                probe_obj = decode_probes[l]
                                if isinstance(probe_obj, dict) and 'scaler' in probe_obj:
                                    hs_scaled = probe_obj['scaler'].transform(hs_mlp.reshape(1, -1))
                                    prob_right = probe_obj['clf'].predict_proba(hs_scaled)[0, 1]
                                else:
                                    clf = probe_obj['clf'] if isinstance(probe_obj, dict) else probe_obj
                                    prob_right = clf.predict_proba(hs_mlp.reshape(1, -1))[0, 1]
                                step_layer_probs[int(l)] = float(1.0 - prob_right)
                        if step_layer_probs:
                            decode_span_scores.append(
                                float(
                                    aggregate_score(
                                        step_layer_probs,
                                        agg=str(args.score_agg),
                                        top_k=int(args.score_top_k),
                                        layer_min=(int(args.score_layer_min) if args.score_layer_min is not None else None),
                                    )
                                )
                            )

                    # 2. 动态触发幻觉检测（若存在 decode probes）
                    if should_trigger_now:
                        saw_pos_trigger = True
                        for l in decode_layers:
                                if l in decode_probes:
                                # feature source: current step (threshold) or cached last-high (exit_span)
                                    if trigger_use_feats is not None:
                                        if int(l) not in trigger_use_feats:
                                            continue
                                        hs_mlp = trigger_use_feats[int(l)].reshape(-1)
                                    else:
                                        if l not in mlp_out_step:
                                            continue
                                        hs_mlp = mlp_out_step[l][0, -1, :].float().numpy()
                                        probe_obj = decode_probes[l]
                                        if isinstance(probe_obj, dict) and 'scaler' in probe_obj:
                                            hs_scaled = probe_obj['scaler'].transform(hs_mlp.reshape(1, -1))
                                            prob_right = probe_obj['clf'].predict_proba(hs_scaled)[0, 1]
                                        else:
                                            clf = probe_obj['clf'] if isinstance(probe_obj, dict) else probe_obj
                                            prob_right = clf.predict_proba(hs_mlp.reshape(1, -1))[0, 1]
                                        
                                    halluc_prob = 1.0 - prob_right
                                if float(halluc_prob) > max_hallu_prob:
                                    max_hallu_prob = float(halluc_prob)
                                if float(halluc_prob) > max_decode_hallu_prob:
                                    max_decode_hallu_prob = float(halluc_prob)
                                # keep per-layer max across decode steps
                                prev = decode_layer_probs.get(int(l), 0.0)
                                if float(halluc_prob) > float(prev):
                                    decode_layer_probs[int(l)] = float(halluc_prob)
                                    if halluc_prob >= args.stop_threshold:
                                        status = "early_exit_decode"
                                    trigger_info = {
                                        "step": int(trigger_step_for_info),
                                        "prob": float(halluc_prob),
                                        "layer": int(l),
                                        "pos_score": float(trigger_pos_score_for_info),
                                        "pos_mode": str(args.pos_trigger_mode),
                                    }
                                    break
                        if status != "proceed":
                                        break
                        
                    if gold_output is not None:
                        generated = torch.cat([generated, step_input], dim=-1)
                        if step_input.item() == tokenizer.eos_token_id:
                            break
                    else:
                        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=-1)
                        if next_token.item() == tokenizer.eos_token_id:
                            break
            
            # hallu_score for AUROC: aggregate across layers (prefill+decode)
            combined_layer_probs = dict(prefill_layer_probs)
            for l, p in decode_layer_probs.items():
                combined_layer_probs[int(l)] = max(float(combined_layer_probs.get(int(l), 0.0)), float(p))

            # IMPORTANT: make scores comparable: only consider layers we actually want to score on.
            if args.score_layer_min is not None:
                combined_layer_probs = {l: p for l, p in combined_layer_probs.items() if int(l) >= int(args.score_layer_min)}

            # Fallback: if filtering removed everything, fall back to unfiltered (avoid degenerate 0.0 scores)
            if not combined_layer_probs:
                combined_layer_probs = dict(prefill_layer_probs)
                for l, p in decode_layer_probs.items():
                    combined_layer_probs[int(l)] = max(float(combined_layer_probs.get(int(l), 0.0)), float(p))

            hallu_score = aggregate_score(
                combined_layer_probs,
                agg=args.score_agg,
                top_k=args.score_top_k,
                layer_min=args.score_layer_min,
            )

            if args.diagnose_decode and decode_layer_probs_at_target:
                hallu_score_decode_at_target = aggregate_score(
                    decode_layer_probs_at_target,
                    agg=args.score_agg,
                    top_k=args.score_top_k,
                    layer_min=args.score_layer_min,
                )
            
            results.append({
                "idx": orig_idx,
                "true_correctness": true_label,
                "status": status,
                "trigger_info": trigger_info,
                # keep both scores for debugging
                "hallu_score": float(hallu_score),
                "hallu_score_max": float(max_hallu_prob),
                "generated_text": "" if args.prefill_only else tokenizer.decode(generated[0][q_len:], skip_special_tokens=True),
                "prefill_layer_probs": prefill_layer_probs,
                "decode_layer_probs": decode_layer_probs,
                "saw_pos_trigger": bool(saw_pos_trigger),
                "max_decode_hallu_prob": float(max_decode_hallu_prob),
                # diagnostics (may be None)
                "expected_decode_step": expected_decode_step,
                "max_pos_score": float(max_pos_score),
                "max_pos_step": max_pos_step,
                "pos_score_at_expected": pos_score_at_expected,
                "first_pos_trigger_step": first_pos_trigger_step,
                "pos_tp": int(pos_tp),
                "pos_fp": int(pos_fp),
                "pos_fn": int(pos_fn),
                "span_start_step": int(span_start_step) if span_start_step is not None else np.nan,
                "span_end_step": int(span_end_step) if span_end_step is not None else np.nan,
                "hallu_score_decode_at_target": float(hallu_score_decode_at_target) if hallu_score_decode_at_target is not None else np.nan,
                "hallu_score_decode_on_span": float(np.max(decode_span_scores)) if decode_span_scores else np.nan,
            })

            # --- Pos probe single-span localization evaluation (per-sample) ---
            if args.pos_span_eval and span_start_step is not None and span_end_step is not None:
                span_found += 1
                scores = np.asarray(pos_scores_by_step, dtype=float)  # length = #decoded tokens (<=max_steps)
                if scores.size == 0:
                    continue
                thr = float(args.pos_span_threshold)
                pred = (scores >= thr).astype(np.int32)

                # extract contiguous positive spans (1-indexed step coordinates, inclusive)
                spans = []
                i = 0
                while i < len(pred):
                    if pred[i] == 1:
                        j = i
                        while j + 1 < len(pred) and pred[j + 1] == 1:
                            j += 1
                        # convert to 1-indexed steps
                        a = int(i + 1)
                        b = int(j + 1)
                        if (b - a + 1) >= int(args.pos_span_min_len):
                            spans.append((a, b))
                        i = j + 1
                    else:
                        i += 1

                if not spans:
                    continue
                span_predicted += 1

                # pick a single span by max(sum of scores) within the span
                best = None
                best_sum = -1.0
                for (a, b) in spans:
                    ssum = float(scores[a - 1:b].sum())
                    if ssum > best_sum:
                        best_sum = ssum
                        best = (a, b)
                if best is None:
                    continue
                ps, pe = best
                gs, ge = int(span_start_step), int(span_end_step)

                inter = max(0, min(pe, ge) - max(ps, gs) + 1)
                union = (pe - ps + 1) + (ge - gs + 1) - inter
                iou = float(inter / union) if union > 0 else 0.0
                span_iou_list.append(iou)
                span_start_err_list.append(float(ps - gs))
                span_end_err_list.append(float(pe - ge))
                if pe == ge:
                    span_hit_end_0 += 1
                if abs(pe - ge) <= 1:
                    span_hit_end_1 += 1
        except Exception as e:
            continue

    # 统计性能
    res_df = pd.DataFrame(results)
    print("\n=== End-to-End Real-time Performance ===")
    print(res_df['status'].value_counts())
    
    if not res_df[res_df['status'] != 'proceed'].empty:
        print("\nTrigger Layer Distribution:")
        exit_data = res_df[res_df['status'] != 'proceed'].copy()
        exit_data['trigger_layer'] = exit_data['trigger_info'].apply(lambda x: x.get('layer'))
        print(exit_data.groupby(['status', 'trigger_layer']).size())

    res_df['pred_hallu'] = (res_df['status'] != 'proceed').astype(int)
    res_df['true_hallu'] = (res_df['true_correctness'] == 0).astype(int)
    
    from sklearn.metrics import classification_report, roc_auc_score
    print("\nDetection Metrics (Predicting Hallucination):")
    print(classification_report(res_df['true_hallu'], res_df['pred_hallu'], zero_division=0))

    def _print_auc(name: str, scores: np.ndarray):
        try:
            auc = roc_auc_score(res_df["true_hallu"], scores)
            print(f"ROC-AUC ({name}): {auc:.4f}")
        except Exception:
            print(f"ROC-AUC ({name}): N/A")

    print("")
    _print_auc("hallu_score", res_df["hallu_score"].values.astype(float))
    _print_auc("hallu_score_max", res_df["hallu_score_max"].values.astype(float))

    # Prefill-only diagnostics: per-layer probe metrics at stop_threshold
    if args.prefill_only and len(res_df) > 0:
        print("\n=== Prefill-only per-layer probe metrics (threshold-based) ===")
        layer_metrics = []
        for l in prefill_layers:
            try:
                scores = res_df["prefill_layer_probs"].apply(lambda d: float(d.get(int(l), np.nan)))
                mask = scores.notna()
                if mask.sum() < 10:
                    continue
                y_true = res_df.loc[mask, "true_hallu"].astype(int).values
                y_score = scores[mask].values.astype(float)
                y_pred = (y_score >= float(args.stop_threshold)).astype(int)
                rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                auc = None
                try:
                    auc = roc_auc_score(y_true, y_score)
                except Exception:
                    auc = None
                layer_metrics.append({
                    "layer": int(l),
                    "n": int(mask.sum()),
                    "acc": float(rep.get("accuracy", np.nan)),
                    "p1": float(rep.get("1", {}).get("precision", np.nan)),
                    "r1": float(rep.get("1", {}).get("recall", np.nan)),
                    "f1": float(rep.get("1", {}).get("f1-score", np.nan)),
                    "auc": float(auc) if auc is not None else np.nan,
                })
            except Exception:
                continue
        if layer_metrics:
            lm = pd.DataFrame(layer_metrics).sort_values(["auc", "acc"], ascending=False)
            print("Top-8 layers by AUROC:")
            print(lm.head(8).to_string(index=False))
        else:
            print("No per-layer metrics available (missing prefill_layer_probs).")

    # Decode-only diagnostics: per-layer probe metrics & pos-trigger stats
    if args.decode_only and len(res_df) > 0:
        try:
            print("\n=== Decode-only pos-trigger stats ===")
            print(res_df["saw_pos_trigger"].value_counts(dropna=False))
        except Exception:
            pass

        print("\n=== Decode-only per-layer probe metrics (threshold-based, gated by pos trigger) ===")
        layer_metrics = []
        # only consider samples where we ever triggered pos, otherwise decode probes never ran
        gated = res_df[res_df["saw_pos_trigger"] == True].copy()
        if len(gated) == 0:
            print("No samples had pos_trigger; cannot evaluate decode probes.")
        else:
            for l in decode_layers:
                try:
                    scores = gated["decode_layer_probs"].apply(lambda d: float(d.get(int(l), np.nan)))
                    mask = scores.notna()
                    if mask.sum() < 10:
                        continue
                    y_true = gated.loc[mask, "true_hallu"].astype(int).values
                    y_score = scores[mask].values.astype(float)
                    y_pred = (y_score >= float(args.stop_threshold)).astype(int)
                    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    auc = None
                    try:
                        auc = roc_auc_score(y_true, y_score)
                    except Exception:
                        auc = None
                    layer_metrics.append({
                        "layer": int(l),
                        "n": int(mask.sum()),
                        "acc": float(rep.get("accuracy", np.nan)),
                        "p1": float(rep.get("1", {}).get("precision", np.nan)),
                        "r1": float(rep.get("1", {}).get("recall", np.nan)),
                        "f1": float(rep.get("1", {}).get("f1-score", np.nan)),
                        "auc": float(auc) if auc is not None else np.nan,
                    })
                except Exception:
                    continue
            if layer_metrics:
                lm = pd.DataFrame(layer_metrics).sort_values(["auc", "acc"], ascending=False)
                print("Top-8 layers by AUROC (decode, gated):")
                print(lm.head(8).to_string(index=False))
            else:
                print("No per-layer metrics available (decode_layer_probs missing).")

    if args.auc_sweep:
        print("\n=== AUROC sweep (post-hoc, same run) ===")
        # We'll reuse hallu_score_max as a baseline and try a few aggregations by recomputing from stored max score proxy.
        # Note: full per-layer probabilities are not persisted; this sweep focuses on layer-min & aggregation for hallu_score,
        # but requires rerun to change those. Here we at least show the max proxy (already printed).
        print("Tip: rerun with e.g. --score_layer_min 20 --score_agg topk_mean --score_top_k 3")

    if args.diagnose_decode and len(res_df) > 0:
        print("\n=== Decode diagnostics (exact_answer_last_token alignment) ===")
        try:
            has_target = res_df["expected_decode_step"].notna()
            print(f"Target span found: {int(has_target.sum())}/{len(res_df)}")
            if has_target.any():
                sub = res_df[has_target].copy()
                sub["pos_step_err"] = (sub["max_pos_step"].astype(float) - sub["expected_decode_step"].astype(float)).abs()
                exact_hit = (sub["max_pos_step"].astype(float) == sub["expected_decode_step"].astype(float)).mean()
                within1 = (sub["pos_step_err"] <= 1.0).mean()
                print(f"pos max-step == expected-step: {exact_hit:.3f}")
                print(f"pos max-step within ±1: {within1:.3f}")
                print(f"mean |step error|: {sub['pos_step_err'].mean():.3f}")

                # Streaming key-token detection stats for pos probe.
                # We evaluate timing relative to the exact-answer span: early if first_trigger < span_start_step.
                try:
                    tp = int(sub.get("pos_tp", 0).sum()) if "pos_tp" in sub else 0
                    fp = int(sub.get("pos_fp", 0).sum()) if "pos_fp" in sub else 0
                    fn = int(sub.get("pos_fn", 0).sum()) if "pos_fn" in sub else 0
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                    miss_rate = float((sub["first_pos_trigger_step"].isna()).mean()) if "first_pos_trigger_step" in sub else float("nan")
                    # Early relative to span start (if available); fall back to expected_decode_step (last token)
                    ref_start = sub["span_start_step"] if "span_start_step" in sub else sub["expected_decode_step"]
                    ref_end = sub["span_end_step"] if "span_end_step" in sub else sub["expected_decode_step"]
                    early_rate = float(
                        (sub["first_pos_trigger_step"].notna() & (sub["first_pos_trigger_step"].astype(float) < ref_start.astype(float))).mean()
                    ) if "first_pos_trigger_step" in sub else float("nan")
                    # delay-to-span-end (for intuition): negative means trigger before the end of the span.
                    delay_end = (sub["first_pos_trigger_step"].astype(float) - ref_end.astype(float)) if "first_pos_trigger_step" in sub else None
                    mode = str(getattr(args, "pos_trigger_mode", "threshold"))
                    print("\n=== Pos probe streaming key-token detection ===")
                    if mode == "threshold":
                        print(f"mode=threshold (span as positive) pos_threshold={float(args.pos_threshold):.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} (tp={tp}, fp={fp}, fn={fn})")
                    else:
                        print(f"mode=exit_span enter_thr={float(args.pos_enter_threshold):.3f} exit_thr={float(args.pos_exit_threshold):.3f} k={int(args.pos_enter_k)}")
                        print(f"(NOTE) tp/fp/fn are not token-level in exit_span mode (currently not accumulated); timing stats below are sample-level.")
                    # We report early relative to span-start; delay relative to span-end (last token in span).
                    mean_delay_end = float(delay_end.dropna().mean()) if (delay_end is not None and delay_end.notna().any()) else float("nan")
                    print(f"first_trigger miss_rate={miss_rate:.3f} early_trigger_rate={early_rate:.3f} mean_delay_to_span_end={mean_delay_end:.3f}")
                except Exception:
                    pass

                # AUROC if we score using decode probes at the true target token (only where we could compute it)
                mask2 = np.isfinite(sub["hallu_score_decode_at_target"].values.astype(float))
                if mask2.sum() >= 10:
                    try:
                        auc2 = roc_auc_score(sub.loc[mask2, "true_hallu"].astype(int), sub.loc[mask2, "hallu_score_decode_at_target"].astype(float))
                        print(f"ROC-AUC (decode_at_true_target): {auc2:.4f} (n={int(mask2.sum())})")
                    except Exception:
                        print("ROC-AUC (decode_at_true_target): N/A")
                else:
                    print("ROC-AUC (decode_at_true_target): N/A (insufficient samples)")

                # AUROC / accuracy if we score using decode probes across the entire exact-answer span (aggregated by max across span tokens)
                try:
                    mask3 = np.isfinite(sub["hallu_score_decode_on_span"].values.astype(float))
                    if mask3.sum() >= 10:
                        try:
                            auc3 = roc_auc_score(sub.loc[mask3, "true_hallu"].astype(int), sub.loc[mask3, "hallu_score_decode_on_span"].astype(float))
                            print(f"ROC-AUC (decode_on_span_max): {auc3:.4f} (n={int(mask3.sum())})")
                        except Exception:
                            print("ROC-AUC (decode_on_span_max): N/A")
                        # threshold-based accuracy at stop_threshold
                        try:
                            y_true3 = sub.loc[mask3, "true_hallu"].astype(int).values
                            y_pred3 = (sub.loc[mask3, "hallu_score_decode_on_span"].astype(float).values >= float(args.stop_threshold)).astype(int)
                            rep3 = classification_report(y_true3, y_pred3, zero_division=0, output_dict=True)
                            print(f"Accuracy (decode_on_span_max @ stop_threshold={float(args.stop_threshold):.2f}): {float(rep3.get('accuracy', np.nan)):.3f} (n={int(mask3.sum())})")
                        except Exception:
                            pass
                    else:
                        print("ROC-AUC (decode_on_span_max): N/A (insufficient samples)")
                except Exception:
                    pass
        except Exception:
            pass

    if args.pos_score_dist:
        # Print token-level pos score distribution (span positives vs answer negatives)
        try:
            import numpy as _np
            from sklearn.metrics import roc_auc_score as _roc_auc_score
            from sklearn.metrics import average_precision_score as _average_precision_score
            print("\n=== Pos probe score distribution (token-level, on answer tokens) ===")
            npos = len(pos_scores_pos)
            nneg = len(pos_scores_neg)
            print(f"Collected token scores: pos={npos} neg={nneg} (only samples with span found)")
            if npos > 0 and nneg > 0:
                pos_arr = _np.array(pos_scores_pos, dtype=float)
                neg_arr = _np.array(pos_scores_neg, dtype=float)
                # threshold-free metrics on token-level distribution
                try:
                    y_true = _np.concatenate([_np.ones_like(pos_arr, dtype=int), _np.zeros_like(neg_arr, dtype=int)])
                    y_score = _np.concatenate([pos_arr, neg_arr]).astype(float)
                    auc_tok = float(_roc_auc_score(y_true, y_score))
                    ap_tok = float(_average_precision_score(y_true, y_score))
                    print(f"Token-level ROC-AUC={auc_tok:.4f} PR-AUC(AP)={ap_tok:.4f}")
                except Exception:
                    pass
                qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
                def _q(arr):
                    return {q: float(_np.quantile(arr, q)) for q in qs}
                qpos = _q(pos_arr)
                qneg = _q(neg_arr)
                print("Quantiles (pos): " + " ".join([f"p{int(q*100):02d}={qpos[q]:.3f}" for q in qs]))
                print("Quantiles (neg): " + " ".join([f"p{int(q*100):02d}={qneg[q]:.3f}" for q in qs]))

                # small threshold sweep
                print("\nThreshold sweep (token-level, positives=span tokens):")
                thr_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
                for thr in thr_list:
                    tp = int((pos_arr >= thr).sum())
                    fn = int((pos_arr < thr).sum())
                    fp = int((neg_arr >= thr).sum())
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                    print(f"  thr={thr:.2f}  precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}  (tp={tp} fp={fp} fn={fn})")
            else:
                print("Not enough token scores collected to compute distribution.")
        except Exception:
            pass

    if args.pos_span_eval:
        try:
            print("\n=== Pos probe SINGLE-SPAN localization (answer tokens) ===")
            print(f"pos_span_threshold={float(args.pos_span_threshold):.3f} pos_span_min_len={int(args.pos_span_min_len)}")
            print(f"Span found (GT): {int(span_found)}/{len(res_df)}  | Span predicted: {int(span_predicted)}/{int(span_found) if span_found>0 else 0}")
            if span_iou_list:
                iou_arr = np.asarray(span_iou_list, dtype=float)
                se_arr = np.asarray(span_start_err_list, dtype=float)
                ee_arr = np.asarray(span_end_err_list, dtype=float)
                print(f"IoU: mean={float(iou_arr.mean()):.3f} median={float(np.median(iou_arr)):.3f}")
                print(f"start_err: mean={float(se_arr.mean()):.3f} median={float(np.median(se_arr)):.3f}")
                print(f"end_err:   mean={float(ee_arr.mean()):.3f} median={float(np.median(ee_arr)):.3f}")
                n = len(iou_arr)
                print(f"hit@end (exact): {float(span_hit_end_0/max(1,n)):.3f}  hit@end±1: {float(span_hit_end_1/max(1,n)):.3f} (n={n})")
            else:
                print("No spans were predicted; no stats.")
        except Exception:
            pass

if __name__ == "__main__":
    args = parse_args()
    run_e2e_realtime_eval(args)
