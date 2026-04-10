import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch
from baukit import TraceDict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

from probing_utils import (
    LIST_OF_DATASETS,
    LIST_OF_TEST_DATASETS,
    MODEL_FRIENDLY_NAMES,
    get_indices_of_exact_answer,
    get_probing_layer_names,
    load_model_and_validate_gpu,
    tokenize,
    find_prompt_end_in_full_ids,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=str, default='last')
    parser.add_argument(
        '--probe_at',
        type=str,
        default='resid',
        choices=['resid', 'mlp'],
        help="Feature source for position probe: resid uses outputs.hidden_states; mlp uses MLP output activations via TraceDict.",
    )
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--neg_pos_ratio', type=float, default=5.0)
    parser.add_argument(
        '--neg_strategy',
        type=str,
        default='mixed',
        choices=['random_answer', 'hard_window', 'mixed'],
        help="How to sample negatives (always from answer tokens excluding span): "
             "random_answer=random from answer; hard_window=only near span; mixed=half near span, half random.",
    )
    parser.add_argument(
        '--hard_window',
        type=int,
        default=8,
        help="When --neg_strategy hard_window/mixed: negatives are sampled from answer tokens within +/-hard_window "
             "of any span token (excluding the span).",
    )
    parser.add_argument('--use_pca', type=int, default=256)
    parser.add_argument('--use_scaler', action='store_true', help="If set, standardize features (fit on train) before PCA/LogReg.")
    parser.add_argument('--val_ratio', type=float, default=0.2, help="Hold-out ratio for reporting AUC (ROW-level split).")
    parser.add_argument('--localize_eval', action='store_true', help="Also report localization metrics on val split (argmax token position).")
    parser.add_argument('--pos_threshold', type=float, default=0.5, help="When --localize_eval, threshold for token-level key-token detection metrics.")
    parser.add_argument('--max_search_tokens', type=int, default=256, help="When --localize_eval, scan at most this many output tokens for argmax position.")
    parser.add_argument('--out_dir', type=str, default='../output/token_probes_pure')
    parser.add_argument(
        '--out_tag',
        type=str,
        default='',
        help="Optional tag appended to the output base name (keeps legacy naming when empty). "
             "Example: --out_tag layer-15 -> ..._pure_layer-15.joblib",
    )
    parser.add_argument(
        '--pos_mode',
        type=str,
        default='all',
        choices=['first', 'last', 'all'],
        help="Which tokens in the exact-answer span to use as positives: "
             "first=only first token; last=only last token; all=all tokens in span.",
    )
    return parser.parse_args()

def forward_on_ids(model, input_ids_1d: torch.Tensor, *, probe_at: str, layer_names: list[str] | None):
    """
    Returns per-layer features as a list of [seq_len, hidden_dim] on CPU.
    - probe_at='resid': residual stream hidden states (outputs.hidden_states)
    - probe_at='mlp': MLP output activations traced by module names (layer_names)
    """
    input_ids = input_ids_1d.to(model.device).unsqueeze(0)
    with torch.no_grad():
        if probe_at == "mlp":
            assert layer_names is not None and len(layer_names) > 0
            with TraceDict(model, layers=layer_names) as td:
                _ = model(input_ids=input_ids, use_cache=False, output_hidden_states=False)
            feats = []
            for name in layer_names:
                act = td[name].output  # [1, seq_len, hidden]
                feats.append(act[0].detach().cpu())
            return feats
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hiddens = [h[0].detach().cpu() for h in outputs.hidden_states]
        return hiddens

def main():
    args = parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    friendly = MODEL_FRIENDLY_NAMES[args.model]
    
    model, tokenizer = load_model_and_validate_gpu(args.model)
    if args.layer == 'last':
        layer_idx = -1
    else:
        layer_idx = int(args.layer)
    layer_names = None
    if args.probe_at == "mlp":
        # Names for MLP outputs per layer (same convention as probing scripts)
        layer_names = get_probing_layer_names('mlp', model.config._name_or_path)

    df = pd.read_csv(f'../output/{friendly}-answers-{args.dataset}.csv')
    df = df[(df['valid_exact_answer'] == 1) & (df['exact_answer'] != 'NO ANSWER')]
    if args.max_samples > 0: df = df.sample(n=min(args.max_samples, len(df)), random_state=0)
    ids_all = torch.load(f'../output/{friendly}-input_output_ids-{args.dataset}.pt', map_location='cpu')

    # Split by row (example) to avoid leakage across tokens from the same sample.
    all_row_indices = df.index.to_numpy()
    val_ratio = float(args.val_ratio)
    if not (0.0 < val_ratio < 0.9):
        raise ValueError(f"--val_ratio must be in (0, 0.9), got {args.val_ratio}")
    train_rows, val_rows = train_test_split(all_row_indices, test_size=val_ratio, random_state=0)
    train_row_set = set(int(i) for i in train_rows)
    val_row_set = set(int(i) for i in val_rows)

    X_train, y_train = [], []
    X_val, y_val = [], []
    print(f'Extracting pure hidden states for {args.dataset}...')
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            ids = ids_all[int(idx)]
            hiddens = forward_on_ids(model, ids, probe_at=args.probe_at, layer_names=layer_names)
            # Strict alignment: find prompt end by matching IDs, then search exact answer within the output segment.
            prompt = str(row["question"])
            prompt_ids_1d = tokenize(prompt, tokenizer, args.model)[0]
            q_len = find_prompt_end_in_full_ids(ids, prompt_ids_1d, allow_bos_mismatch=True)
            output_ids_1d = ids[int(q_len):]
            span = get_indices_of_exact_answer(tokenizer, ids, row['exact_answer'], args.model, output_ids=output_ids_1d)
            if not span: continue

            # Positives: tokens inside the exact-answer span, selected by pos_mode
            all_span_indices = [int(i) for i in span]
            all_span_indices = [i for i in all_span_indices if 0 <= i < len(ids)]
            if not all_span_indices:
                continue
            if args.pos_mode == 'first':
                pos_indices = [all_span_indices[0]]
            elif args.pos_mode == 'last':
                pos_indices = [all_span_indices[-1]]
            else:  # 'all'
                pos_indices = all_span_indices
            if not pos_indices:
                continue

            # Negatives: ONLY from the answer segment (output tokens), excluding span tokens.
            # This matches online usage better: we only need to distinguish key tokens within decode.
            span_set = set(pos_indices)
            ans_start = int(q_len)
            ans_end = len(ids)
            if ans_end <= ans_start:
                continue
            all_answer_negs = [i for i in range(ans_start, ans_end) if i not in span_set]
            if not all_answer_negs:
                continue
            # Hard negatives: within a window around span tokens (excluding span)
            hw = int(max(0, args.hard_window))
            hard_negs = []
            if hw > 0:
                hard_set = set()
                for p in pos_indices:
                    for t in range(max(ans_start, p - hw), min(ans_end, p + hw + 1)):
                        if t not in span_set:
                            hard_set.add(int(t))
                hard_negs = sorted(hard_set)
            # Sample negatives proportional to #positives (keeps dataset size reasonable)
            num_pos = len(pos_indices)
            num_neg = int(max(1, round(float(args.neg_pos_ratio) * float(num_pos))))
            sampled_negs = []
            if args.neg_strategy == "random_answer":
                sampled_negs = random.sample(all_answer_negs, min(len(all_answer_negs), num_neg))
            elif args.neg_strategy == "hard_window":
                pool = hard_negs if hard_negs else all_answer_negs
                sampled_negs = random.sample(pool, min(len(pool), num_neg))
            else:  # mixed
                n_hard = int(num_neg // 2)
                n_rand = int(num_neg - n_hard)
                hard_pool = hard_negs if hard_negs else all_answer_negs
                sampled_negs.extend(random.sample(hard_pool, min(len(hard_pool), n_hard)))
                # random negatives from answer excluding already chosen
                remain = [i for i in all_answer_negs if i not in set(sampled_negs)]
                if remain and n_rand > 0:
                    sampled_negs.extend(random.sample(remain, min(len(remain), n_rand)))

            is_train = int(idx) in train_row_set
            X_dst, y_dst = (X_train, y_train) if is_train else (X_val, y_val)
            for p_idx in pos_indices:
                X_dst.append(hiddens[layer_idx][p_idx].to(torch.float32).numpy())
                y_dst.append(1)
            for n_idx in sampled_negs:
                X_dst.append(hiddens[layer_idx][n_idx].to(torch.float32).numpy())
                y_dst.append(0)
        except: continue

    if not X_train or not X_val:
        raise RuntimeError(f"Not enough data after filtering: X_train={len(X_train)}, X_val={len(X_val)}")
    X_train = np.stack(X_train); y_train = np.array(y_train)
    X_val = np.stack(X_val); y_val = np.array(y_val)
    scaler = None
    if args.use_scaler:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    pca = None
    if args.use_pca > 0:
        pca = PCA(n_components=min(args.use_pca, X_train.shape[0], X_train.shape[1]), random_state=0)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Report AUCs on row-level split (still token-level labels, but examples are disjoint).
    train_scores = clf.predict_proba(X_train)[:, 1]
    val_scores = clf.predict_proba(X_val)[:, 1]
    train_auc = roc_auc_score(y_train, train_scores) if len(np.unique(y_train)) == 2 else float("nan")
    val_auc = roc_auc_score(y_val, val_scores) if len(np.unique(y_val)) == 2 else float("nan")
    train_ap = average_precision_score(y_train, train_scores) if len(np.unique(y_train)) == 2 else float("nan")
    val_ap = average_precision_score(y_val, val_scores) if len(np.unique(y_val)) == 2 else float("nan")
    print(f"[pos_probe] train ROC-AUC: {train_auc:.4f} (n={len(y_train)})")
    print(f"[pos_probe]   val ROC-AUC: {val_auc:.4f} (val_ratio={val_ratio}, n={len(y_val)})")
    print(f"[pos_probe] train PR-AUC(AP): {train_ap:.4f}")
    print(f"[pos_probe]   val PR-AUC(AP): {val_ap:.4f}")

    if args.localize_eval:
        # Token-level key-token detection metrics on val:
        # iterate over answer tokens and treat tokens in the exact-answer span as positives.
        # This matches online usage: classify each decode step as key-token or not.
        tp = fp = fn = tn = 0
        n_found = 0
        n_total = 0
        max_search = int(args.max_search_tokens)
        for idx, row in tqdm(df.loc[list(val_row_set)].iterrows(), total=len(val_row_set), desc="Localization (val)"):
            try:
                ids = ids_all[int(idx)]
                hiddens = forward_on_ids(model, ids, probe_at=args.probe_at, layer_names=layer_names)
                prompt = str(row["question"])
                prompt_ids_1d = tokenize(prompt, tokenizer, args.model)[0]
                q_len = find_prompt_end_in_full_ids(ids, prompt_ids_1d, allow_bos_mismatch=True)
                output_ids_1d = ids[int(q_len):]
                span = get_indices_of_exact_answer(tokenizer, ids, row['exact_answer'], args.model, output_ids=output_ids_1d)
                n_total += 1
                if not span:
                    continue
                pos_indices = [int(i) for i in span]
                pos_indices = [i for i in pos_indices if 0 <= i < len(ids)]
                if not pos_indices:
                    continue
                span_set = set(pos_indices)
                # sanity: span should lie in answer segment for decode-only usage
                if min(pos_indices) < int(q_len):
                    continue
                # scan output segment up to max_search tokens
                start = int(q_len)
                end = min(len(ids), int(q_len) + max_search)
                if end <= start:
                    continue
                feats = []
                y_true = []
                for t in range(start, end):
                    feats.append(hiddens[layer_idx][t].to(torch.float32).numpy())
                    y_true.append(1 if int(t) in span_set else 0)
                feats = np.stack(feats)
                if scaler is not None:
                    feats = scaler.transform(feats)
                if pca is not None:
                    feats = pca.transform(feats)
                scores = clf.predict_proba(feats)[:, 1]
                y_pred = (scores >= float(args.pos_threshold)).astype(int)
                y_true = np.array(y_true, dtype=int)
                # confusion counts
                tp += int(((y_pred == 1) & (y_true == 1)).sum())
                fp += int(((y_pred == 1) & (y_true == 0)).sum())
                fn += int(((y_pred == 0) & (y_true == 1)).sum())
                tn += int(((y_pred == 0) & (y_true == 0)).sum())
                n_found += 1
            except Exception:
                continue
        if n_found > 0:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            print(
                f"[pos_probe] token-metrics (val): span_found={n_found}/{n_total} "
                f"pos_threshold={float(args.pos_threshold):.3f} "
                f"precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} "
                f"(tp={tp}, fp={fp}, fn={fn}, tn={tn}) (max_search={max_search})"
            )
            # quick threshold sweep for calibration (token-level)
            try:
                thr_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
                # recompute from accumulated (tp/fp/fn/tn) is not possible; re-score on X_val
                # So sweep on token-level val set directly:
                y_score = val_scores.astype(float)
                y_true2 = y_val.astype(int)
                best = {"thr": None, "f1": -1.0, "prec": 0.0, "rec": 0.0}
                print("[pos_probe] threshold sweep (val, token-level):")
                for thr in thr_list:
                    yp = (y_score >= float(thr)).astype(int)
                    tp2 = int(((yp == 1) & (y_true2 == 1)).sum())
                    fp2 = int(((yp == 1) & (y_true2 == 0)).sum())
                    fn2 = int(((yp == 0) & (y_true2 == 1)).sum())
                    prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0.0
                    rec2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0
                    f12 = (2 * prec2 * rec2 / (prec2 + rec2)) if (prec2 + rec2) > 0 else 0.0
                    print(f"  thr={thr:.2f} precision={prec2:.3f} recall={rec2:.3f} f1={f12:.3f} (tp={tp2} fp={fp2} fn={fn2})")
                    if f12 > best["f1"]:
                        best = {"thr": float(thr), "f1": float(f12), "prec": float(prec2), "rec": float(rec2)}
                if best["thr"] is not None:
                    print(f"[pos_probe] best@val_f1: thr={best['thr']:.2f} f1={best['f1']:.3f} precision={best['prec']:.3f} recall={best['rec']:.3f}")
            except Exception:
                pass
        else:
            print(f"[pos_probe] token-metrics (val): span_found=0/{n_total} (no stats)")

    base = f'{friendly}_{args.dataset}_pure'
    if args.out_tag:
        base = f'{base}_{args.out_tag}'
    out_base = os.path.join(args.out_dir, base)
    joblib.dump(
        {
            'clf': clf,
            'pca': pca,
            'scaler': scaler,
            'probe_at': args.probe_at,
            'layer': args.layer,
            # training recipe metadata (optional, backward compatible)
            'pos_label': 'exact_answer_span_all_tokens',
            'neg_source': 'answer_tokens_excluding_span',
            'neg_pos_ratio': float(args.neg_pos_ratio),
            'neg_strategy': str(args.neg_strategy),
            'hard_window': int(args.hard_window),
            'use_scaler': bool(args.use_scaler),
        },
        out_base + '.joblib'
    )
    print(f'Saved pure probe to {out_base}.joblib')

if __name__ == '__main__':
    main()
