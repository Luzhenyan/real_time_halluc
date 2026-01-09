import argparse
import os
import torch
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from baukit import TraceDict

from probing_utils import (
    MODEL_FRIENDLY_NAMES,
    load_model_and_validate_gpu,
    get_indices_of_exact_answer,
    get_probing_layer_names,
    tokenize,
    find_prompt_end_in_full_ids,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--pos_probe_path', type=str, required=True)
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--max_samples', type=int, default=200)
    # Position selection strategy (replace plain argmax)
    parser.add_argument(
        '--pos_select_strategy',
        type=str,
        default='threshold_last',
        choices=[
            'threshold_last',
            'threshold_first',
            'last_run_last',
            'best_run_last',
            'causal_run_argmax',
            'causal_peak_patience',
            'causal_end_of_run',
            'causal_hysteresis_end',
            'causal_max_eps_last',
            'causal_hi_then_drop_last',
            'max_eps_last',
            'topk_last',
            'argmax',
        ],
        help="How to select a single token from per-token positive scores."
    )
    parser.add_argument(
        '--pos_threshold',
        type=float,
        default=0.5,
        help="Threshold on positive score (predict_proba[:,1]) for threshold/run strategies."
    )
    parser.add_argument(
        '--pos_calibrate',
        type=int,
        default=0,
        help="If 1, calibrate pos_threshold on a calibration dataset by maximizing position metric."
    )
    parser.add_argument(
        '--pos_calib_dataset',
        type=str,
        default=None,
        help="Dataset to calibrate threshold on (e.g., winobias). Defaults to dataset.replace('_test','')."
    )
    parser.add_argument(
        '--pos_calib_max_samples',
        type=int,
        default=500,
        help="Max samples for threshold calibration (0 means all)."
    )
    parser.add_argument(
        '--pos_calib_metric',
        type=str,
        default='exact',
        choices=['exact', 'offby1'],
        help="Metric to maximize when calibrating threshold."
    )
    parser.add_argument(
        '--pos_calib_max_fallback',
        type=float,
        default=0.5,
        help="During calibration, prefer thresholds with no_trigger_rate <= this value. Set >1 to disable."
    )
    parser.add_argument(
        '--pos_calib_lambda_fallback',
        type=float,
        default=0.0,
        help="If >0, optimize (metric - lambda*no_trigger_rate) instead of a hard no-trigger constraint."
    )
    parser.add_argument(
        '--pos_calib_grid',
        type=int,
        default=101,
        help="Number of thresholds in [0,1] to try when calibrating."
    )
    parser.add_argument(
        '--pos_min_run',
        type=int,
        default=1,
        help="Minimum length of a contiguous high-score run (score>=threshold) for best_run_last."
    )
    parser.add_argument(
        '--pos_hysteresis_delta',
        type=float,
        default=0.05,
        help="For causal_hysteresis_end: off_threshold = on_threshold - delta."
    )
    parser.add_argument(
        '--pos_hi',
        type=float,
        default=0.999,
        help="For causal_hi_then_drop_last: high-confidence threshold to enter a run."
    )
    parser.add_argument(
        '--pos_lo',
        type=float,
        default=0.5,
        help="For causal_hi_then_drop_last: low threshold; when score drops below this after entering hi-run, we emit."
    )
    parser.add_argument(
        '--pos_min_run_hi',
        type=int,
        default=1,
        help="For causal_hi_then_drop_last: minimum length of the hi-run before we allow emitting."
    )
    parser.add_argument(
        '--pos_fail_action',
        type=str,
        default='skip',
        choices=['skip', 'use_q_end', 'use_answer_end'],
        help="If the position strategy does not trigger, either skip the sample or use a fixed fallback position (NOT argmax)."
    )
    # Optional: more conservative hallucination scoring when a late high-score spike exists in position scores.
    parser.add_argument(
        '--hallu_agg',
        type=str,
        default='single',
        choices=['single', 'max_with_tail'],
        help="How to aggregate hallucination score. 'single' uses hallu probe at selected pos token. "
             "'max_with_tail' uses max(hallu@selected_pos, hallu@tail_pos) when a late spike is detected."
    )
    parser.add_argument(
        '--tail_gate_threshold',
        type=float,
        default=0.5,
        help="Threshold for defining 'above' in tail spike detection (applies to position probe scores)."
    )
    parser.add_argument(
        '--tail_gate_min_delta',
        type=int,
        default=16,
        help="Only treat as tail spike if (tail_pos - selected_pos) >= this many answer tokens."
    )
    parser.add_argument(
        '--tail_pos',
        type=str,
        default='last_above',
        choices=['last_above', 'answer_end'],
        help="Which tail position to use when hallu_agg=max_with_tail."
    )
    parser.add_argument(
        '--pos_eps',
        type=float,
        default=0.02,
        help="For max_eps_last: pick the LAST token with score >= (max_score - eps)."
    )
    parser.add_argument(
        '--pos_topk',
        type=int,
        default=5,
        help="For topk_last: pick the LAST token among the top-k scores."
    )
    parser.add_argument(
        '--pos_patience',
        type=int,
        default=16,
        help="For causal_peak_patience: after the best-so-far peak, wait this many answer tokens without a new peak, then emit the peak index."
    )
    parser.add_argument(
        '--pos_min_peak',
        type=float,
        default=0.5,
        help="For causal_peak_patience: require best-so-far peak score >= this before allowing early emit."
    )
    return parser.parse_args()

def get_activations(model, input_ids_1d, layer_idx, probe_at):
    input_ids = input_ids_1d.to(model.device).unsqueeze(0)
    layers_to_trace = get_probing_layer_names(probe_at, model.config._name_or_path)
    target_layer = layers_to_trace[layer_idx]
    with torch.no_grad():
        with TraceDict(model, [target_layer], retain_input=False) as ret:
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits[0].detach().cpu()
        residual_stream = [h[0].detach().cpu() for h in outputs.hidden_states]
        mlp_activations = ret[target_layer].output[0].detach().cpu()
    return logits, residual_stream, mlp_activations

def entropy_and_margin(logits_slice):
    probs = torch.softmax(logits_slice, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
    top2 = torch.topk(logits_slice, k=2, dim=-1).values
    margin = (top2[0] - top2[1]).item() if top2.numel() == 2 else 0.0
    p_top1 = torch.max(probs).item()
    return entropy, margin, p_top1

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_clf_and_scaler(path):
    """
    兼容两种 checkpoint 格式：
    - 旧：直接 pickle 一个 sklearn classifier
    - 新：pickle 一个 dict {"clf": clf, "scaler": StandardScaler}
    """
    obj = load_pkl(path)
    if isinstance(obj, dict) and ("clf" in obj):
        return obj.get("clf"), obj.get("scaler", None)
    return obj, None

def _argmax_abs(scores: list[float], start_abs: int) -> int:
    if len(scores) == 0:
        return start_abs
    return start_abs + int(np.argmax(scores))

def _runs_above_threshold(scores: list[float], threshold: float):
    """
    Return list of (start_idx_in_scores, end_idx_in_scores_inclusive) where scores[idx] >= threshold.
    """
    runs = []
    start = None
    for i, s in enumerate(scores):
        if s >= threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(scores) - 1))
    return runs

def _last_above_abs(scores: list[float], start_abs: int, threshold: float):
    """
    Return absolute index of the LAST score>=threshold within scores (scores are answer-local, start_abs=q_len).
    Returns None if none above threshold.
    """
    if len(scores) == 0:
        return None
    last = None
    for i, s in enumerate(scores):
        if float(s) >= float(threshold):
            last = i
    if last is None:
        return None
    return int(start_abs + last)

def select_pos_token_abs(scores: list[float], start_abs: int, strategy: str, threshold: float, min_run: int):
    """
    Select absolute token index using a dynamic strategy.
    Returns: (best_idx_abs_or_None, triggered_bool)
    """
    if len(scores) == 0:
        return None, False

    if strategy == 'argmax':
        return _argmax_abs(scores, start_abs), True

    above = [i for i, s in enumerate(scores) if s >= threshold]
    if strategy == 'threshold_first':
        if len(above) > 0:
            return start_abs + above[0], True
        return None, False

    if strategy == 'threshold_last':
        if len(above) > 0:
            return start_abs + above[-1], True
        return None, False

    if strategy == 'best_run_last':
        runs = _runs_above_threshold(scores, threshold)
        runs = [(a, b) for (a, b) in runs if (b - a + 1) >= max(1, int(min_run))]
        if len(runs) == 0:
            return None, False

        # Pick run with highest mean score; tie-break by later end index
        best = None
        for (a, b) in runs:
            mean_score = float(np.mean(scores[a:b+1]))
            cand = (mean_score, b, a)
            if best is None or cand > best:
                best = cand
        _, best_end, _ = best
        return start_abs + int(best_end), True

    if strategy == 'last_run_last':
        runs = _runs_above_threshold(scores, threshold)
        if len(runs) == 0:
            return None, False
        # pick the end of the last run (closest to the end of the answer)
        return start_abs + int(runs[-1][1]), True

    # Defensive fallback
    return None, False

def select_pos_token_abs_with_args(scores: list[float], start_abs: int, args):
    """
    Wrapper so strategies that need extra params (eps/topk) can use args cleanly.
    Returns: (best_idx_abs_or_None, triggered_bool)
    """
    if args.pos_select_strategy in ("max_eps_last", "topk_last"):
        if len(scores) == 0:
            return None, False
        if args.pos_select_strategy == "max_eps_last":
            max_s = float(np.max(scores))
            cand = [i for i, s in enumerate(scores) if s >= (max_s - float(args.pos_eps))]
            if len(cand) == 0:
                return None, False
            return start_abs + int(cand[-1]), True
        if args.pos_select_strategy == "topk_last":
            k = max(1, min(int(args.pos_topk), len(scores)))
            topk_idx = np.argpartition(np.array(scores), -k)[-k:]
            return start_abs + int(np.max(topk_idx)), True
    return select_pos_token_abs(scores, start_abs, args.pos_select_strategy, args.pos_threshold, args.pos_min_run)

def select_pos_token_abs_causal(scores: list[float], start_abs: int, threshold_on: float, min_run: int):
    """
    Causal strategy: scan left->right, detect a high-score run (>=threshold_on),
    and when the run ends, emit the last token of that run (if run length >= min_run).
    If no qualifying run ends, but we end inside a run, emit the last token (end-of-sequence).
    If never enters a qualifying run, return None.
    """
    if len(scores) == 0:
        return None, False

    in_run = False
    run_start = None
    last_in_run = None
    last_qualifying_end = None

    for i, s in enumerate(scores):
        if s >= threshold_on:
            if not in_run:
                in_run = True
                run_start = i
            last_in_run = i
        else:
            if in_run:
                run_len = (last_in_run - run_start + 1) if (run_start is not None and last_in_run is not None) else 0
                if run_len >= max(1, int(min_run)):
                    last_qualifying_end = last_in_run
                in_run = False
                run_start = None
                last_in_run = None

    # If ended in a run
    if in_run and run_start is not None and last_in_run is not None:
        run_len = last_in_run - run_start + 1
        if run_len >= max(1, int(min_run)):
            return start_abs + int(last_in_run), True

    # If we had a qualifying ended run, use the last one (closest to end)
    if last_qualifying_end is not None:
        return start_abs + int(last_qualifying_end), True

    return None, False

def select_pos_token_abs_causal_run_argmax(scores: list[float], start_abs: int, threshold_on: float, min_run: int):
    """
    Causal strategy: scan left->right. When we are inside a contiguous run of score>=threshold_on,
    track the argmax position within that run. When the run ENDS, emit argmax-in-run if run_len>=min_run.
    If we end inside a qualifying run, emit its argmax. Otherwise return None.
    """
    if len(scores) == 0:
        return None, False

    thr = float(threshold_on)
    min_run = max(1, int(min_run))

    in_run = False
    run_len = 0
    best_i = None
    best_s = -1e9

    for i, s in enumerate(scores):
        s = float(s)
        if s >= thr:
            if not in_run:
                in_run = True
                run_len = 0
                best_i = i
                best_s = s
            run_len += 1
            if s > best_s:
                best_s = s
                best_i = i
        else:
            if in_run:
                if run_len >= min_run and best_i is not None:
                    return int(start_abs + best_i), True
                in_run = False
                run_len = 0
                best_i = None
                best_s = -1e9

    if in_run and run_len >= min_run and best_i is not None:
        return int(start_abs + best_i), True
    return None, False

def select_pos_token_abs_causal_peak_patience(scores: list[float], start_abs: int, min_peak: float, patience: int):
    """
    Causal strategy for online early-stop:
    - Maintain best-so-far peak (argmax) over the answer token scores.
    - If we have not seen a NEW peak for `patience` answer tokens AND best_score>=min_peak, emit the peak index.
    - Otherwise, if we reach the end, emit the peak (if any).
    Returns: (best_idx_abs_or_None, triggered_bool)
    """
    if len(scores) == 0:
        return None, False

    patience = max(1, int(patience))
    min_peak = float(min_peak)

    best_i = 0
    best_s = float(scores[0])
    last_best_i = 0
    # scan left->right (answer-local indices)
    for i in range(1, len(scores)):
        s = float(scores[i])
        if s > best_s:
            best_s = s
            best_i = i
            last_best_i = i

        # if we're sufficiently past the last peak update, we can stop
        if (i - last_best_i) >= patience and best_s >= min_peak:
            return int(start_abs + best_i), True

    # end-of-sequence: emit best-so-far if it meets min_peak, else None
    if best_s >= min_peak:
        return int(start_abs + best_i), True
    return None, False

def select_pos_token_abs_causal_hysteresis(scores: list[float], start_abs: int, threshold_on: float, threshold_off: float, min_run: int):
    """
    Causal hysteresis: enter run when score>=threshold_on, exit when score<threshold_off.
    Emit run end similarly to select_pos_token_abs_causal.
    """
    if len(scores) == 0:
        return None, False

    in_run = False
    run_start = None
    last_in_run = None
    last_qualifying_end = None

    for i, s in enumerate(scores):
        if not in_run:
            if s >= threshold_on:
                in_run = True
                run_start = i
                last_in_run = i
        else:
            if s >= threshold_off:
                last_in_run = i
            else:
                # exit run
                run_len = (last_in_run - run_start + 1) if (run_start is not None and last_in_run is not None) else 0
                if run_len >= max(1, int(min_run)):
                    last_qualifying_end = last_in_run
                in_run = False
                run_start = None
                last_in_run = None

    if in_run and run_start is not None and last_in_run is not None:
        run_len = last_in_run - run_start + 1
        if run_len >= max(1, int(min_run)):
            return start_abs + int(last_in_run), True

    if last_qualifying_end is not None:
        return start_abs + int(last_qualifying_end), True

    return None, False

def select_pos_token_abs_causal_max_eps_last(scores: list[float], start_abs: int, eps: float):
    """
    Causal "running max" selector:
    - Track best_score seen so far.
    - Maintain candidate_end as the LAST index whose score is within eps of current best_score.
    - If a new best_score appears (> best_score + eps), reset candidate_end to that index.
    Returns: (best_idx_abs_or_None, triggered_bool)

    This avoids relying on score dropping below a threshold (which can cause always choosing answer_end).
    """
    if len(scores) == 0:
        return None, False
    best = -1.0
    cand_end = None
    eps = float(max(0.0, eps))
    for i, s in enumerate(scores):
        s = float(s)
        if s > best + eps:
            best = s
            cand_end = i
        elif s >= best - eps:
            # within eps of best so far -> push to the right
            cand_end = i
    if cand_end is None:
        return None, False
    return start_abs + int(cand_end), True

def select_pos_token_abs_causal_hi_then_drop_last(
    scores: list[float],
    start_abs: int,
    hi: float,
    lo: float,
    min_run_hi: int,
):
    """
    Causal rule:
    - Enter a "hi-run" when score >= hi.
    - Track the last index in that hi-run.
    - Once we've entered a hi-run and later observe score < lo ("sudden drop"),
      emit the last index of the most recent hi-run (if its length >= min_run_hi).
    - If we never observe a drop but have a hi-run till the end, emit the hi-run end.
    - If we never enter a hi-run, return None.

    Returns: (best_idx_abs_or_None, triggered_bool)
    """
    if len(scores) == 0:
        return None, False
    hi = float(hi)
    lo = float(lo)
    min_run_hi = max(1, int(min_run_hi))

    in_hi = False
    hi_start = None
    hi_last = None
    saw_hi = False

    for i, s in enumerate(scores):
        s = float(s)
        if s >= hi:
            if not in_hi:
                in_hi = True
                hi_start = i
            hi_last = i
            saw_hi = True
            continue

        # not >= hi
        if in_hi:
            # exiting hi-run
            in_hi = False

        # "drop" condition: after we have seen any hi, score < lo
        if saw_hi and s < lo:
            if hi_last is not None and hi_start is not None:
                run_len = hi_last - hi_start + 1
                if run_len >= min_run_hi:
                    return start_abs + int(hi_last), True
            # If hi-run too short, keep scanning (maybe a better hi-run later)

    # End of sequence: if we saw a hi-run and didn't drop, emit end of last hi-run if long enough
    if hi_last is not None and hi_start is not None:
        run_len = hi_last - hi_start + 1
        if run_len >= min_run_hi:
            return start_abs + int(hi_last), True

    return None, False

def calibrate_threshold(
    model,
    tokenizer,
    pos_clf,
    pos_pca,
    expected_pos_in_dim,
    dataset_name: str,
    model_key: str,
    max_samples: int,
    layer_idx: int,
    strategy: str,
    min_run: int,
    hysteresis_delta: float,
    metric: str,
    grid_n: int,
    max_fallback: float,
    lambda_fallback: float,
):
    """
    Calibrate pos_threshold by maximizing position metric (exact/offby1) on a calibration dataset.
    Uses ONLY position probe + GT exact_answer span (no hallucination labels), so it doesn't leak into hallucination probe.
    """
    friendly = MODEL_FRIENDLY_NAMES[model_key]
    source_file = f'../output/{friendly}-answers-{dataset_name}.csv'
    ids_file = f'../output/{friendly}-input_output_ids-{dataset_name}.pt'
    df = pd.read_csv(source_file)
    if max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=0)
    ids_all = torch.load(ids_file, map_location='cpu')

    # Collect per-sample (scores, q_len, gt_last) for valid gt_span only
    samples = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'calib_collect({dataset_name})'):
        try:
            input_ids_1d = ids_all[int(idx)]
            logits, residual_stream, _ = get_activations(model, input_ids_1d, layer_idx, 'mlp')
            # IMPORTANT: `input_output_ids` are generated from the *prompt* stored in `question`, not `raw_question`.
            # Using `raw_question` here will miscompute q_len and shift all token indices (notably for math where we append "Answer shortly.").
            prompt_col = "question" if ("question" in row and not pd.isna(row["question"])) else "raw_question"
            prompt_txt = str(row[prompt_col])
            q_len = len(tokenize(prompt_txt, tokenizer, model_key)[0])
            gt_span = get_indices_of_exact_answer(tokenizer, input_ids_1d, row['exact_answer'], model_key, prompt=prompt_txt)
            if not gt_span:
                continue
            gt_last = int(gt_span[-1])

            scores = []
            for t_idx in range(q_len, len(input_ids_1d)):
                ent, marg, p1 = entropy_and_margin(logits[t_idx-1]) if t_idx > 0 else (0, 0, 0)
                hs_resid = residual_stream[-1][t_idx].to(torch.float32).numpy()
                pure_feat = hs_resid
                rich_feat = np.concatenate(
                    [hs_resid, np.array([t_idx, t_idx - q_len, 0], dtype=np.float32), np.array([ent, marg, p1], dtype=np.float32)]
                )
                if expected_pos_in_dim is not None:
                    if pure_feat.shape[0] == expected_pos_in_dim:
                        feat = pure_feat
                    elif rich_feat.shape[0] == expected_pos_in_dim:
                        feat = rich_feat
                    else:
                        feat = pure_feat
                else:
                    feat = pure_feat
                feat_2d = feat.reshape(1, -1)
                if pos_pca:
                    feat_2d = pos_pca.transform(feat_2d)
                scores.append(float(pos_clf.predict_proba(feat_2d)[0, 1]))

            samples.append((scores, q_len, gt_last))
        except Exception:
            continue

    if len(samples) == 0:
        return 0.5

    thresholds = np.linspace(0.0, 1.0, int(grid_n))
    best_thr = 0.5
    best_val = -1.0
    best_no_tr = 1.0

    for thr in thresholds:
        hits = []
        no_triggers = []
        for scores, q_len, gt_last in samples:
            start_abs = q_len
            if strategy == 'causal_end_of_run':
                pred_abs, triggered = select_pos_token_abs_causal(scores, start_abs, threshold_on=float(thr), min_run=min_run)
            elif strategy == 'causal_hysteresis_end':
                off = max(0.0, float(thr) - float(hysteresis_delta))
                pred_abs, triggered = select_pos_token_abs_causal_hysteresis(scores, start_abs, threshold_on=float(thr), threshold_off=off, min_run=min_run)
            else:
                pred_abs, triggered = select_pos_token_abs(scores, start_abs, strategy=strategy, threshold=float(thr), min_run=min_run)

            no_triggers.append(0 if triggered else 1)

            if metric == 'exact':
                hits.append(1 if (pred_abs is not None and int(pred_abs) == int(gt_last)) else 0)
            else:
                hits.append(1 if (pred_abs is not None and abs(int(pred_abs) - int(gt_last)) <= 1) else 0)

        val = float(np.mean(hits)) if len(hits) > 0 else -1.0
        no_tr = float(np.mean(no_triggers)) if len(no_triggers) > 0 else 1.0

        if lambda_fallback and lambda_fallback > 0:
            obj = val - float(lambda_fallback) * no_tr
            best_obj = best_val - float(lambda_fallback) * best_no_tr
            if obj > best_obj:
                best_val = val
                best_no_tr = no_tr
                best_thr = float(thr)
        else:
            # Hard constraint: prefer thresholds that don't fail-to-trigger too often; within that set maximize val.
            if max_fallback <= 1.0:
                if no_tr <= float(max_fallback):
                    if (val > best_val) or (val == best_val and no_tr < best_no_tr):
                        best_val = val
                        best_no_tr = no_tr
                        best_thr = float(thr)
            else:
                # Constraint disabled
                if (val > best_val) or (val == best_val and no_tr < best_no_tr):
                    best_val = val
                    best_no_tr = no_tr
                    best_thr = float(thr)

    print(f'Calibrated threshold on {dataset_name}: best_thr={best_thr:.3f}, {metric}={best_val:.4f}, no_trigger={best_no_tr:.4f}, n={len(samples)}')
    return best_thr

def main():
    args = parse_args()
    friendly = MODEL_FRIENDLY_NAMES[args.model]
    
    pos_data = joblib.load(args.pos_probe_path)
    pos_clf, pos_pca = pos_data['clf'], pos_data['pca']

    base_clf_path = f'../checkpoints/clf_{friendly}_{args.dataset.replace("_test", "")}_layer-15_token-'
    hallu_exact_clf, hallu_exact_scaler = load_clf_and_scaler(base_clf_path + 'exact_answer_last_token.pkl')
    hallu_prefill_clf, hallu_prefill_scaler = load_clf_and_scaler(base_clf_path + 'last_q_token.pkl')
    hallu_full_last_clf, hallu_full_last_scaler = load_clf_and_scaler(base_clf_path + 'full_answer_last_token.pkl')

    source_file = f'../output/{friendly}-answers-{args.dataset}.csv'
    ids_file = f'../output/{friendly}-input_output_ids-{args.dataset}.pt'
    model, tokenizer = load_model_and_validate_gpu(args.model)
    df = pd.read_csv(source_file)
    if args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
    ids_all = torch.load(ids_file, map_location='cpu')
    
    y_true, y_probs_dynamic, y_preds_dynamic = [], [], []
    y_probs_static, y_preds_static = [], []
    y_probs_prefill, y_preds_prefill = [], []
    y_probs_full_last, y_preds_full_last = [], []

    # Position probe diagnostics (only for samples where GT exact answer span exists)
    pos_exact_matches = []
    pos_off_by_1 = []
    pos_abs_err = []
    pos_triggered = []

    # Infer what feature format the loaded position probe expects
    # - "pure": hidden_state only (dim = hidden_dim)
    # - "rich": hidden_state + [t_idx, rel_idx, 0] + [entropy, margin, p_top1] (dim = hidden_dim + 6)
    expected_pos_in_dim = None
    if pos_pca is not None and hasattr(pos_pca, "n_features_in_"):
        expected_pos_in_dim = int(pos_pca.n_features_in_)
    elif hasattr(pos_clf, "n_features_in_"):
        expected_pos_in_dim = int(pos_clf.n_features_in_)
    
    # Optional threshold calibration
    if args.pos_calibrate == 1:
        calib_ds = args.pos_calib_dataset or args.dataset.replace("_test", "")
        if calib_ds != args.dataset:
            print(f'Calibrating pos_threshold on dataset={calib_ds} (eval dataset={args.dataset})...')
        else:
            print(f'Calibrating pos_threshold on the same dataset={args.dataset} (note: this is not a clean holdout)...')
        args.pos_threshold = calibrate_threshold(
            model=model,
            tokenizer=tokenizer,
            pos_clf=pos_clf,
            pos_pca=pos_pca,
            expected_pos_in_dim=expected_pos_in_dim,
            dataset_name=calib_ds,
            model_key=args.model,
            max_samples=args.pos_calib_max_samples,
            layer_idx=args.layer,
            strategy=args.pos_select_strategy,
            min_run=args.pos_min_run,
            hysteresis_delta=args.pos_hysteresis_delta,
            metric=args.pos_calib_metric,
            grid_n=args.pos_calib_grid,
            max_fallback=args.pos_calib_max_fallback,
            lambda_fallback=args.pos_calib_lambda_fallback,
        )

    print(f'Running Diagnostic Evaluation on {args.dataset}...')
    print(f'Position selection: strategy={args.pos_select_strategy}, threshold={args.pos_threshold}, min_run={args.pos_min_run}, eps={args.pos_eps}, topk={args.pos_topk}, fail_action={args.pos_fail_action}')
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            input_ids_1d = ids_all[int(idx)]
            logits, residual_stream, mlp_activations = get_activations(model, input_ids_1d, args.layer, 'mlp')
            # IMPORTANT: use the actual prompt (`question`) to align with `input_output_ids`
            prompt_col = "question" if ("question" in row and not pd.isna(row["question"])) else "raw_question"
            prompt_txt = str(row[prompt_col])
            prompt_ids_1d = tokenize(prompt_txt, tokenizer, args.model)[0]
            q_len = find_prompt_end_in_full_ids(input_ids_1d, prompt_ids_1d, allow_bos_mismatch=True)
            
            # Prefill
            prefill_hs = mlp_activations[max(0, q_len - 1)].to(torch.float32).numpy()
            prefill_feat = prefill_hs.reshape(1, -1)
            if hallu_prefill_scaler is not None:
                prefill_feat = hallu_prefill_scaler.transform(prefill_feat)
            y_probs_prefill.append(hallu_prefill_clf.predict_proba(prefill_feat)[0, 1])
            y_preds_prefill.append(hallu_prefill_clf.predict(prefill_feat)[0])
            
            # Full Last
            fl_hs = mlp_activations[len(input_ids_1d) - 1].to(torch.float32).numpy()
            fl_feat = fl_hs.reshape(1, -1)
            if hallu_full_last_scaler is not None:
                fl_feat = hallu_full_last_scaler.transform(fl_feat)
            y_probs_full_last.append(hallu_full_last_clf.predict_proba(fl_feat)[0, 1])
            y_preds_full_last.append(hallu_full_last_clf.predict(fl_feat)[0])
            
            # Position
            ans_token_scores = []
            for t_idx in range(q_len, len(input_ids_1d)):
                ent, marg, p1 = entropy_and_margin(logits[t_idx-1]) if t_idx > 0 else (0,0,0)
                hs_resid = residual_stream[-1][t_idx].to(torch.float32).numpy()

                # Build candidate feature vectors (pure vs rich), then pick the one that matches
                pure_feat = hs_resid
                rich_feat = np.concatenate(
                    [hs_resid, np.array([t_idx, t_idx - q_len, 0], dtype=np.float32), np.array([ent, marg, p1], dtype=np.float32)]
                )

                if expected_pos_in_dim is not None:
                    if pure_feat.shape[0] == expected_pos_in_dim:
                        feat = pure_feat
                    elif rich_feat.shape[0] == expected_pos_in_dim:
                        feat = rich_feat
                    else:
                        # Fallback: try pure first; if it doesn't work, try rich
                        feat = pure_feat
                else:
                    # If we cannot infer expected dim, default to pure (safer for "token_probes_pure")
                    feat = pure_feat

                feat_2d = feat.reshape(1, -1)
                if pos_pca:
                    feat_2d = pos_pca.transform(feat_2d)

                ans_token_scores.append(pos_clf.predict_proba(feat_2d)[0, 1])
            
            if args.pos_select_strategy == 'causal_peak_patience':
                best_idx_abs, triggered = select_pos_token_abs_causal_peak_patience(
                    ans_token_scores,
                    start_abs=q_len,
                    min_peak=args.pos_min_peak,
                    patience=args.pos_patience,
                )
            elif args.pos_select_strategy == 'causal_run_argmax':
                best_idx_abs, triggered = select_pos_token_abs_causal_run_argmax(
                    ans_token_scores, start_abs=q_len, threshold_on=args.pos_threshold, min_run=args.pos_min_run
                )
            elif args.pos_select_strategy == 'causal_end_of_run':
                best_idx_abs, triggered = select_pos_token_abs_causal(
                    ans_token_scores, start_abs=q_len, threshold_on=args.pos_threshold, min_run=args.pos_min_run
                )
            elif args.pos_select_strategy == 'causal_hysteresis_end':
                off = max(0.0, float(args.pos_threshold) - float(args.pos_hysteresis_delta))
                best_idx_abs, triggered = select_pos_token_abs_causal_hysteresis(
                    ans_token_scores, start_abs=q_len, threshold_on=args.pos_threshold, threshold_off=off, min_run=args.pos_min_run
                )
            elif args.pos_select_strategy == 'causal_max_eps_last':
                best_idx_abs, triggered = select_pos_token_abs_causal_max_eps_last(
                    ans_token_scores, start_abs=q_len, eps=args.pos_eps
                )
            elif args.pos_select_strategy == 'causal_hi_then_drop_last':
                best_idx_abs, triggered = select_pos_token_abs_causal_hi_then_drop_last(
                    ans_token_scores,
                    start_abs=q_len,
                    hi=args.pos_hi,
                    lo=args.pos_lo,
                    min_run_hi=args.pos_min_run_hi,
                )
            else:
                best_idx_abs, triggered = select_pos_token_abs_with_args(ans_token_scores, start_abs=q_len, args=args)

            if not triggered or best_idx_abs is None:
                if args.pos_fail_action == 'use_q_end':
                    best_idx_abs = max(0, q_len - 1)
                    triggered = True
                elif args.pos_fail_action == 'use_answer_end':
                    best_idx_abs = int(len(input_ids_1d) - 1)
                    triggered = True
                else:
                    # skip this sample entirely (keep y_true alignment)
                    y_probs_prefill.pop(); y_preds_prefill.pop()
                    y_probs_full_last.pop(); y_preds_full_last.pop()
                    continue
            
            # Dynamic
            dynamic_hs = mlp_activations[best_idx_abs].to(torch.float32).numpy()
            dyn_feat = dynamic_hs.reshape(1, -1)
            if hallu_exact_scaler is not None:
                dyn_feat = hallu_exact_scaler.transform(dyn_feat)
            dyn_prob = float(hallu_exact_clf.predict_proba(dyn_feat)[0, 1])

            # Optional conservative aggregation when a late spike exists.
            if args.hallu_agg == 'max_with_tail':
                tail_abs = None
                if args.tail_pos == 'answer_end':
                    tail_abs = int(len(input_ids_1d) - 1)
                else:
                    # last_above: last token with pos_score >= tail_gate_threshold
                    tail_abs = _last_above_abs(ans_token_scores, start_abs=q_len, threshold=args.tail_gate_threshold)

                if tail_abs is not None and int(tail_abs) >= 0 and int(tail_abs) < int(len(input_ids_1d)):
                    # gate: only apply if tail is sufficiently after the chosen position
                    if int(tail_abs) - int(best_idx_abs) >= int(args.tail_gate_min_delta):
                        tail_hs = mlp_activations[int(tail_abs)].to(torch.float32).numpy()
                        tail_feat = tail_hs.reshape(1, -1)
                        if hallu_exact_scaler is not None:
                            tail_feat = hallu_exact_scaler.transform(tail_feat)
                        tail_prob = float(hallu_exact_clf.predict_proba(tail_feat)[0, 1])
                        dyn_prob = float(max(dyn_prob, tail_prob))

            y_probs_dynamic.append(dyn_prob)
            # keep prediction threshold consistent with sklearn default decision boundary (0.5 for LogisticRegression)
            y_preds_dynamic.append(1 if dyn_prob >= 0.5 else 0)
            
            # Static
            gt_span = get_indices_of_exact_answer(tokenizer, input_ids_1d, row['exact_answer'], args.model, prompt=prompt_txt)
            if gt_span:
                gt_last = gt_span[-1]
                pos_exact_matches.append(1 if best_idx_abs == gt_last else 0)
                pos_off_by_1.append(1 if abs(int(best_idx_abs) - int(gt_last)) <= 1 else 0)
                pos_abs_err.append(abs(int(best_idx_abs) - int(gt_last)))
                pos_triggered.append(1 if triggered else 0)

                gt_hs = mlp_activations[gt_span[-1]].to(torch.float32).numpy()
                gt_feat = gt_hs.reshape(1, -1)
                if hallu_exact_scaler is not None:
                    gt_feat = hallu_exact_scaler.transform(gt_feat)
                y_probs_static.append(hallu_exact_clf.predict_proba(gt_feat)[0, 1])
                y_preds_static.append(hallu_exact_clf.predict(gt_feat)[0])
                y_true.append(row['automatic_correctness'])
            else:
                y_probs_prefill.pop(); y_preds_prefill.pop()
                y_probs_full_last.pop(); y_preds_full_last.pop()
                y_probs_dynamic.pop(); y_preds_dynamic.pop()
                
        except Exception: continue

    print('\n=== Comparison Table ===')
    headers = ['Position', 'Accuracy', 'AUC', 'F1 Score']
    results_list = [
        ('Prefill Last Token', y_preds_prefill, y_probs_prefill),
        ('Full Answer Last', y_preds_full_last, y_probs_full_last),
        ('Dynamic (Detected)', y_preds_dynamic, y_probs_dynamic),
        ('Static (GT Position)', y_preds_static, y_probs_static),
    ]
    
    print(f"{headers[0]:<25} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<10}")
    print("-" * 65)
    for name, preds, probs in results_list:
        acc = accuracy_score(y_true, preds)
        auc = roc_auc_score(y_true, probs)
        f1 = f1_score(y_true, preds)
        print(f"{name:<25} | {acc:.4f}     | {auc:.4f}     | {f1:.4f}")
    
    if len(pos_exact_matches) > 0:
        print(f'\nPosition Exact Match: {np.mean(pos_exact_matches):.4f}')
        print(f'Position Off-by-1:    {np.mean(pos_off_by_1):.4f}')
        print(f'Position Abs Error:   {np.mean(pos_abs_err):.4f}')
        if len(pos_triggered) > 0:
            print(f'Position Trigger(%):  {100.0 * np.mean(pos_triggered):.2f}')
    else:
        print('\nPosition metrics: N/A (no samples with valid exact_answer span)')

"""
NOTE: The block below is a duplicated fragment accidentally pasted into this file (starts with an indented line).
It is intentionally kept here as a comment to avoid breaking execution.


            input_ids_1d = ids_all[int(idx)]
            logits, residual_stream, mlp_activations = get_activations(model, input_ids_1d, args.layer, 'mlp')
            # IMPORTANT: use the actual prompt (`question`) to align with `input_output_ids`
            prompt_col = "question" if ("question" in row and not pd.isna(row["question"])) else "raw_question"
            prompt_txt = str(row[prompt_col])
            q_len = len(tokenize(prompt_txt, tokenizer, args.model)[0])
            
            # Prefill
            prefill_hs = mlp_activations[max(0, q_len - 1)].to(torch.float32).numpy()
            y_probs_prefill.append(hallu_clf_prefill.predict_proba(prefill_hs.reshape(1, -1))[0, 1])
            y_preds_prefill.append(hallu_clf_prefill.predict(prefill_hs.reshape(1, -1))[0])
            
            # Full Last
            fl_hs = mlp_activations[len(input_ids_1d) - 1].to(torch.float32).numpy()
            y_probs_full_last.append(hallu_clf_full_last.predict_proba(fl_hs.reshape(1, -1))[0, 1])
            y_preds_full_last.append(hallu_clf_full_last.predict(fl_hs.reshape(1, -1))[0])
            
            # Position
            ans_token_scores = []
            for t_idx in range(q_len, len(input_ids_1d)):
                ent, marg, p1 = entropy_and_margin(logits[t_idx-1]) if t_idx > 0 else (0,0,0)
                hs_resid = residual_stream[-1][t_idx].to(torch.float32).numpy()

                # Build candidate feature vectors (pure vs rich), then pick the one that matches
                pure_feat = hs_resid
                rich_feat = np.concatenate(
                    [hs_resid, np.array([t_idx, t_idx - q_len, 0], dtype=np.float32), np.array([ent, marg, p1], dtype=np.float32)]
                )

                if expected_pos_in_dim is not None:
                    if pure_feat.shape[0] == expected_pos_in_dim:
                        feat = pure_feat
                    elif rich_feat.shape[0] == expected_pos_in_dim:
                        feat = rich_feat
                    else:
                        # Fallback: try pure first; if it doesn't work, try rich
                        feat = pure_feat
                else:
                    # If we cannot infer expected dim, default to pure (safer for "token_probes_pure")
                    feat = pure_feat

                feat_2d = feat.reshape(1, -1)
                if pos_pca:
                    feat_2d = pos_pca.transform(feat_2d)

                ans_token_scores.append(pos_clf.predict_proba(feat_2d)[0, 1])
            
            if args.pos_select_strategy == 'causal_peak_patience':
                best_idx_abs, triggered = select_pos_token_abs_causal_peak_patience(
                    ans_token_scores,
                    start_abs=q_len,
                    min_peak=args.pos_min_peak,
                    patience=args.pos_patience,
                )
            elif args.pos_select_strategy == 'causal_run_argmax':
                best_idx_abs, triggered = select_pos_token_abs_causal_run_argmax(
                    ans_token_scores, start_abs=q_len, threshold_on=args.pos_threshold, min_run=args.pos_min_run
                )
            elif args.pos_select_strategy == 'causal_end_of_run':
                best_idx_abs, triggered = select_pos_token_abs_causal(
                    ans_token_scores, start_abs=q_len, threshold_on=args.pos_threshold, min_run=args.pos_min_run
                )
            elif args.pos_select_strategy == 'causal_hysteresis_end':
                off = max(0.0, float(args.pos_threshold) - float(args.pos_hysteresis_delta))
                best_idx_abs, triggered = select_pos_token_abs_causal_hysteresis(
                    ans_token_scores, start_abs=q_len, threshold_on=args.pos_threshold, threshold_off=off, min_run=args.pos_min_run
                )
            elif args.pos_select_strategy == 'causal_max_eps_last':
                best_idx_abs, triggered = select_pos_token_abs_causal_max_eps_last(
                    ans_token_scores, start_abs=q_len, eps=args.pos_eps
                )
            elif args.pos_select_strategy == 'causal_hi_then_drop_last':
                best_idx_abs, triggered = select_pos_token_abs_causal_hi_then_drop_last(
                    ans_token_scores,
                    start_abs=q_len,
                    hi=args.pos_hi,
                    lo=args.pos_lo,
                    min_run_hi=args.pos_min_run_hi,
                )
            else:
                best_idx_abs, triggered = select_pos_token_abs_with_args(ans_token_scores, start_abs=q_len, args=args)

            if not triggered or best_idx_abs is None:
                if args.pos_fail_action == 'use_q_end':
                    best_idx_abs = max(0, q_len - 1)
                    triggered = True
                elif args.pos_fail_action == 'use_answer_end':
                    best_idx_abs = int(len(input_ids_1d) - 1)
                    triggered = True
                else:
                    # skip this sample entirely (keep y_true alignment)
                    y_probs_prefill.pop(); y_preds_prefill.pop()
                    y_probs_full_last.pop(); y_preds_full_last.pop()
                    continue
            
            # Dynamic
            dynamic_hs = mlp_activations[best_idx_abs].to(torch.float32).numpy()
            dyn_prob = float(hallu_clf_exact.predict_proba(dynamic_hs.reshape(1, -1))[0, 1])

            # Optional conservative aggregation when a late spike exists.
            if args.hallu_agg == 'max_with_tail':
                tail_abs = None
                if args.tail_pos == 'answer_end':
                    tail_abs = int(len(input_ids_1d) - 1)
                else:
                    # last_above: last token with pos_score >= tail_gate_threshold
                    tail_abs = _last_above_abs(ans_token_scores, start_abs=q_len, threshold=args.tail_gate_threshold)

                if tail_abs is not None and int(tail_abs) >= 0 and int(tail_abs) < int(len(input_ids_1d)):
                    # gate: only apply if tail is sufficiently after the chosen position
                    if int(tail_abs) - int(best_idx_abs) >= int(args.tail_gate_min_delta):
                        tail_hs = mlp_activations[int(tail_abs)].to(torch.float32).numpy()
                        tail_prob = float(hallu_clf_exact.predict_proba(tail_hs.reshape(1, -1))[0, 1])
                        dyn_prob = float(max(dyn_prob, tail_prob))

            y_probs_dynamic.append(dyn_prob)
            # keep prediction threshold consistent with sklearn default decision boundary (0.5 for LogisticRegression)
            y_preds_dynamic.append(1 if dyn_prob >= 0.5 else 0)
            
            # Static
            gt_span = get_indices_of_exact_answer(tokenizer, input_ids_1d, row['exact_answer'], args.model, prompt=prompt_txt)
            if gt_span:
                gt_last = gt_span[-1]
                pos_exact_matches.append(1 if best_idx_abs == gt_last else 0)
                pos_off_by_1.append(1 if abs(int(best_idx_abs) - int(gt_last)) <= 1 else 0)
                pos_abs_err.append(abs(int(best_idx_abs) - int(gt_last)))
                pos_triggered.append(1 if triggered else 0)

                gt_hs = mlp_activations[gt_span[-1]].to(torch.float32).numpy()
                y_probs_static.append(hallu_clf_exact.predict_proba(gt_hs.reshape(1, -1))[0, 1])
                y_preds_static.append(hallu_clf_exact.predict(gt_hs.reshape(1, -1))[0])
                y_true.append(row['automatic_correctness'])
            else:
                y_probs_prefill.pop(); y_preds_prefill.pop()
                y_probs_full_last.pop(); y_preds_full_last.pop()
                y_probs_dynamic.pop(); y_preds_dynamic.pop()
                
        except Exception: continue

    print('\n=== Comparison Table ===')
    headers = ['Position', 'Accuracy', 'AUC', 'F1 Score']
    results_list = [
        ('Prefill Last Token', y_preds_prefill, y_probs_prefill),
        ('Full Answer Last', y_preds_full_last, y_probs_full_last),
        ('Dynamic (Detected)', y_preds_dynamic, y_probs_dynamic),
        ('Static (GT Position)', y_preds_static, y_probs_static),
    ]
    
    print(f"{headers[0]:<25} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<10}")
    print("-" * 65)
    for name, preds, probs in results_list:
        acc = accuracy_score(y_true, preds)
        auc = roc_auc_score(y_true, probs)
        f1 = f1_score(y_true, preds)
        print(f"{name:<25} | {acc:.4f}     | {auc:.4f}     | {f1:.4f}")
    
    if len(pos_exact_matches) > 0:
        print(f'\nPosition Exact Match: {np.mean(pos_exact_matches):.4f}')
        print(f'Position Off-by-1:    {np.mean(pos_off_by_1):.4f}')
        print(f'Position Abs Error:   {np.mean(pos_abs_err):.4f}')
        if len(pos_triggered) > 0:
            print(f'Position Trigger(%):  {100.0 * np.mean(pos_triggered):.2f}')
    else:
        print('\nPosition metrics: N/A (no samples with valid exact_answer span)')

"""
if __name__ == '__main__':
    main()
