import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import set_seed

from eval_dynamic_pipeline import get_activations
from probing_utils import MODEL_FRIENDLY_NAMES, load_model_and_validate_gpu, tokenize, get_indices_of_exact_answer, exact_answer_is_valid


def parse_args():
    p = argparse.ArgumentParser(description="Train/eval hallucination probes at key token positions.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, help="Training dataset name (e.g. winobias)")
    p.add_argument("--test_dataset", type=str, required=True, help="Test dataset name (e.g. winobias_test)")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--probe_at", type=str, default="mlp", choices=["mlp"])
    p.add_argument("--max_samples", type=int, default=2000, help="0 means all")
    p.add_argument("--max_samples_test", type=int, default=2000, help="0 means all")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 5, 26, 42, 63])
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--out_dir", type=str, default="../output/hallu_keypos")
    p.add_argument("--save_dir", type=str, default="../checkpoints/hallu_keypos")
    p.add_argument(
        "--threshold_mode",
        type=str,
        default="fixed",
        choices=["fixed", "val_opt"],
        help="How to choose probability threshold for turning probs into class predictions.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Used when --threshold_mode=fixed.",
    )
    p.add_argument(
        "--threshold_metric",
        type=str,
        default="balanced_acc",
        choices=["acc", "balanced_acc", "f1_pos1", "f1_pos0", "youden"],
        help="Metric to optimize on the validation set when --threshold_mode=val_opt.",
    )
    p.add_argument(
        "--threshold_grid",
        type=int,
        default=201,
        help="Number of thresholds in [0,1] to scan when --threshold_mode=val_opt.",
    )
    p.add_argument(
        "--positions",
        type=str,
        default="first_answer_token,exact_answer_before_first_token,exact_answer_first_token,exact_answer_last_token,full_answer_last_token",
        help="Comma-separated positions.",
    )
    return p.parse_args()


def load_artifacts(friendly: str, dataset: str, max_samples: int):
    answers_path = f"../output/{friendly}-answers-{dataset}.csv"
    ids_path = f"../output/{friendly}-input_output_ids-{dataset}.pt"
    df = pd.read_csv(answers_path)
    if max_samples and max_samples > 0:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    ids_all = torch.load(ids_path, map_location="cpu")
    return df, ids_all


def safe_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    # NOTE: y=automatic_correctness (1=correct, 0=incorrect). We report F1 for both pos_label=1 and pos_label=0.
    f1_pos1 = float(f1_score(y_true, y_pred, pos_label=1))
    f1_pos0 = float(f1_score(y_true, y_pred, pos_label=0))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return acc, bacc, auc, f1_pos1, f1_pos0


def majority_baseline_acc(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return float("nan")
    p1 = float(y.mean())
    return float(max(p1, 1.0 - p1))


def pick_threshold(y_true, prob, metric: str, n_grid: int):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if len(y_true) == 0:
        return 0.5, float("nan")
    if n_grid < 2:
        n_grid = 2
    thrs = np.linspace(0.0, 1.0, int(n_grid))
    best_thr = 0.5
    best_score = -1e9
    for thr in thrs:
        pred = (prob >= thr).astype(int)
        if metric == "acc":
            score = accuracy_score(y_true, pred)
        elif metric == "balanced_acc":
            score = balanced_accuracy_score(y_true, pred)
        elif metric == "f1_pos1":
            score = f1_score(y_true, pred, pos_label=1)
        elif metric == "f1_pos0":
            score = f1_score(y_true, pred, pos_label=0)
        elif metric == "youden":
            # Youden J = TPR - FPR for pos_label=1
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr
        else:
            raise ValueError(f"Unknown threshold metric: {metric}")
        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)
    return best_thr, best_score


def compute_position_index(
    pos: str,
    tokenizer,
    model_key: str,
    question: str,
    input_ids_1d,
    q_len: int,
    exact_answer,
    exact_valid,
):
    seq_len = len(input_ids_1d)
    ans_len = seq_len - q_len
    if ans_len <= 0:
        return None

    if pos == "first_answer_token":
        return q_len
    if pos == "full_answer_last_token":
        return seq_len - 1

    # exact-answer derived positions: require valid exact answer AND find a span; otherwise return None (no fallback)
    if pos.startswith("exact_answer_"):
        if not exact_answer_is_valid(exact_valid, exact_answer):
            return None
        span = get_indices_of_exact_answer(tokenizer, input_ids_1d, exact_answer, model_key, prompt=question)
        if not span:
            return None
        if pos == "exact_answer_first_token":
            return int(span[0])
        if pos == "exact_answer_last_token":
            return int(min(seq_len - 1, span[-1]))
        if pos == "exact_answer_before_first_token":
            t = int(span[0]) - 1
            return t if t >= 0 else None
        # allow extension if needed later
        return None

    return None


def main():
    args = parse_args()
    positions = [x.strip() for x in args.positions.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    friendly = MODEL_FRIENDLY_NAMES[args.model]
    df_tr, ids_tr = load_artifacts(friendly, args.dataset, args.max_samples)
    df_te, ids_te = load_artifacts(friendly, args.test_dataset, args.max_samples_test)

    model, tokenizer = load_model_and_validate_gpu(args.model)

    # Collect features (once per sample, for all requested positions)
    X_tr = {p: [] for p in positions}
    y_tr = {p: [] for p in positions}
    X_te = {p: [] for p in positions}
    y_te = {p: [] for p in positions}

    def collect(df, ids_all, X, y, tag):
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"collect_{tag}({args.dataset if tag=='train' else args.test_dataset})"):
            try:
                input_ids_1d = ids_all[int(idx)]
                _, _, mlp_acts = get_activations(model, input_ids_1d, args.layer, args.probe_at)
                # IMPORTANT: align with `input_output_ids` generated from the prompt stored in `question`
                q_col = "question" if ("question" in row and not pd.isna(row["question"])) else "raw_question"
                question = str(row[q_col])
                q_len = len(tokenize(question, tokenizer, args.model)[0])
                y0 = int(row["automatic_correctness"])

                exact_answer = row["exact_answer"] if "exact_answer" in row and not pd.isna(row["exact_answer"]) else None
                exact_valid = int(row["valid_exact_answer"]) if "valid_exact_answer" in row and not pd.isna(row["valid_exact_answer"]) else None

                for p in positions:
                    t_idx = compute_position_index(
                        pos=p,
                        tokenizer=tokenizer,
                        model_key=args.model,
                        question=question,
                        input_ids_1d=input_ids_1d,
                        q_len=q_len,
                        exact_answer=exact_answer,
                        exact_valid=exact_valid,
                    )
                    if t_idx is None or t_idx < 0 or t_idx >= len(input_ids_1d):
                        continue
                    feat = mlp_acts[int(t_idx)].to(torch.float32).numpy()
                    X[p].append(feat)
                    y[p].append(y0)
            except Exception:
                continue

    collect(df_tr, ids_tr, X_tr, y_tr, "train")
    collect(df_te, ids_te, X_te, y_te, "test")

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        for p in positions:
            if len(y_tr[p]) < 50 or len(y_te[p]) < 50:
                continue
            X = np.stack(X_tr[p], axis=0)
            y = np.asarray(y_tr[p], dtype=int)
            idxs = np.arange(len(y))
            tr_idx, va_idx = train_test_split(
                idxs,
                test_size=float(args.val_frac),
                random_state=seed,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            clf = LogisticRegression(random_state=seed, max_iter=2000).fit(X[tr_idx], y[tr_idx])

            prob_va = clf.predict_proba(X[va_idx])[:, 1]
            if args.threshold_mode == "val_opt":
                thr, thr_score = pick_threshold(y[va_idx], prob_va, args.threshold_metric, args.threshold_grid)
            else:
                thr, thr_score = float(args.threshold), float("nan")
            pred_va = (prob_va >= thr).astype(int)
            va_acc, va_bacc, va_auc, va_f1_pos1, va_f1_pos0 = safe_metrics(y[va_idx], pred_va, prob_va)

            Xt = np.stack(X_te[p], axis=0)
            yt = np.asarray(y_te[p], dtype=int)
            prob_t = clf.predict_proba(Xt)[:, 1]
            pred_t = (prob_t >= thr).astype(int)
            te_acc, te_bacc, te_auc, te_f1_pos1, te_f1_pos0 = safe_metrics(yt, pred_t, prob_t)

            save_path = os.path.join(args.save_dir, f"clf_{friendly}_{args.dataset}_keypos-{p}_layer-{args.layer}_seed-{seed}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(clf, f)

            y_te_mean = float(np.mean(yt))
            rows.append(
                {
                    "seed": int(seed),
                    "position": p,
                    "train_n": int(len(y)),
                    "val_n": int(len(va_idx)),
                    "val_acc": va_acc,
                    "val_balanced_acc": va_bacc,
                    "val_auc": va_auc,
                    "val_f1_pos1": va_f1_pos1,
                    "val_f1_pos0": va_f1_pos0,
                    "threshold_mode": args.threshold_mode,
                    "threshold_metric": args.threshold_metric,
                    "threshold": float(thr),
                    "threshold_metric_score": float(thr_score),
                    "test_dataset": args.test_dataset,
                    "test_n": int(len(yt)),
                    "test_acc": te_acc,
                    "test_balanced_acc": te_bacc,
                    "test_auc": te_auc,
                    "test_f1_pos1": te_f1_pos1,
                    "test_f1_pos0": te_f1_pos0,
                    "test_pos_rate_y1": y_te_mean,
                    "test_majority_baseline_acc": majority_baseline_acc(yt),
                    "clf_path": save_path,
                }
            )

    out_csv = os.path.join(args.out_dir, f"{friendly}_{args.dataset}_keypos_probes.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved results: {out_csv}")
    print(f"Saved clfs to: {args.save_dir}")


if __name__ == "__main__":
    main()



