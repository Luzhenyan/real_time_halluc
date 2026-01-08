import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from probing_utils import (
    MODEL_FRIENDLY_NAMES,
    load_model_and_validate_gpu,
    extract_internal_reps_single_sample,
    compile_probing_indices,
    tokenize,
    find_prompt_end_in_full_ids,
    get_indices_of_exact_answer,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a token-level hallucination probe for every layer (supports last_q_token & key decode tokens)."
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument(
        "--probe_at",
        type=str,
        default="mlp",
        choices=["mlp", "mlp_last_layer_only", "mlp_last_layer_only_input", "attention_output"],
    )
    p.add_argument("--token", type=str, default="last_q_token")
    p.add_argument(
        "--n_samples",
        type=str,
        default="all",
        help="Use 'all' or an int (e.g. 2000). Matches probe.py semantics.",
    )
    p.add_argument(
        "--n_per_class",
        type=int,
        default=None,
        help="If set, sample exactly this many positives and negatives for training (total train = 2*n_per_class).",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--n_validation_samples", type=int, default=200)
    p.add_argument("--layer_start", type=int, default=0)
    p.add_argument("--layer_end", type=int, default=-1, help="-1 means 'last layer index'")
    p.add_argument("--out_dir", type=str, default="../output/prefill_all_layers")
    p.add_argument("--save_clf_dir", type=str, default="../checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.save_clf_dir, exist_ok=True)

    friendly = MODEL_FRIENDLY_NAMES[args.model]
    answers_path = f"../output/{friendly}-answers-{args.dataset}.csv"
    ids_path = f"../output/{friendly}-input_output_ids-{args.dataset}.pt"

    data = pd.read_csv(answers_path).reset_index()
    input_output_ids = torch.load(ids_path, map_location="cpu")

    model, tokenizer = load_model_and_validate_gpu(args.model)

    # 获取层数（list 的长度）
    sample_ids = input_output_ids[0]
    sample_out = extract_internal_reps_single_sample(model, sample_ids, args.probe_at, args.model)
    n_layers = len(sample_out)

    layer_end = (n_layers - 1) if args.layer_end == -1 else int(args.layer_end)
    layers = list(range(int(args.layer_start), layer_end + 1))

    for seed in args.seeds:
        train_idx, val_idx = compile_probing_indices(
            data,
            args.n_samples,
            seed,
            n_validation_samples=args.n_validation_samples,
        )
        # NOTE: keep behavior aligned with legacy `probe.py`:
        # - do NOT drop invalid exact answers; instead fall back to last_q_token when exact span is missing.

        # 可选：严格指定每类训练样本数（从非验证集池中抽样）
        if args.n_per_class is not None:
            rng = np.random.RandomState(seed)

            all_idx = np.arange(len(data))
            val_set = set(int(x) for x in np.asarray(val_idx).tolist())
            pool_idx = np.array([i for i in all_idx if int(i) not in val_set], dtype=np.int64)

            pool_labels = data.iloc[pool_idx]["automatic_correctness"].values.astype(int)
            pool_pos = pool_idx[pool_labels == 1]
            pool_neg = pool_idx[pool_labels == 0]

            need = int(args.n_per_class)
            if len(pool_pos) < need or len(pool_neg) < need:
                raise RuntimeError(
                    f"Not enough samples for n_per_class={need}: "
                    f"pool_pos={len(pool_pos)}, pool_neg={len(pool_neg)} (after holding out val={len(val_idx)})"
                )

            sel_pos = rng.choice(pool_pos, need, replace=False)
            sel_neg = rng.choice(pool_neg, need, replace=False)
            train_idx = np.concatenate([sel_pos, sel_neg])
            rng.shuffle(train_idx)

        # 训练集均衡（只对 train 做，val 保持原分布）
        # - 若指定 n_per_class，则 train_idx 已严格均衡，此处跳过
        if args.n_per_class is None:
            train_labels = data.iloc[train_idx]["automatic_correctness"].values
            pos_indices = train_idx[train_labels == 1]
            neg_indices = train_idx[train_labels == 0]
            min_count = min(len(pos_indices), len(neg_indices))
            if min_count == 0:
                raise RuntimeError("Training split has only one class after sampling; cannot balance.")

            np.random.seed(seed)
            balanced_pos = np.random.choice(pos_indices, min_count, replace=False)
            balanced_neg = np.random.choice(neg_indices, min_count, replace=False)
            train_idx = np.concatenate([balanced_pos, balanced_neg])
            np.random.shuffle(train_idx)

        print(f"Extracting features (Train: {len(train_idx)}, Valid: {len(val_idx)})...")

        def collect_features(indices, desc):
            X = {l: [] for l in layers}
            y = []
            for i in tqdm(indices, desc=desc):
                row = data.iloc[int(i)]
                # Align ids with the original CSV row id (legacy `probe.py` uses original indices)
                orig_idx = int(row["index"]) if "index" in row else int(i)
                full_ids = input_output_ids[orig_idx]

                # Legacy safety guard (see `prepare_for_probing`)
                if len(full_ids) > 10000:
                    continue

                y.append(int(row["automatic_correctness"]))
                prompt_text = str(row["question"])  # NOTE: 生成时真实喂给模型的 prompt（含 preprocess）
                
                # 确定截断位置（q_len）
                if args.token == "full_answer_last_token":
                    # Legacy definition: last token of the whole sequence
                    q_len = len(full_ids)
                elif args.token == "exact_answer_last_token":
                    # Legacy definition: last token of GT exact answer span, falling back to last_q_token if missing/invalid
                    exact_answer = row["exact_answer"] if "exact_answer" in row else None
                    valid_exact = int(row["valid_exact_answer"]) if "valid_exact_answer" in row else 0
                    span = None
                    if valid_exact == 1 and isinstance(exact_answer, str) and exact_answer != "NO ANSWER":
                        span = get_indices_of_exact_answer(
                            tokenizer, full_ids, exact_answer, args.model, prompt=prompt_text
                        )
                    if span is None or len(span) == 0:
                        prompt_ids = tokenize(prompt_text, tokenizer, args.model)[0]
                        q_len = find_prompt_end_in_full_ids(full_ids, prompt_ids, allow_bos_mismatch=True)
                    else:
                        t = min(len(full_ids) - 1, int(span[-1]))
                        q_len = t + 1
                else:
                    # Prefill definition: last token of the prompt (strict id match)
                    prompt_ids = tokenize(prompt_text, tokenizer, args.model)[0]
                    q_len = find_prompt_end_in_full_ids(full_ids, prompt_ids, allow_bos_mismatch=True)
                
                q_len = max(1, int(q_len))

                # 只前向到 q_len，确保特征纯净（不包含后续 token）
                out = extract_internal_reps_single_sample(model, full_ids[:q_len], args.probe_at, args.model)
                for l in layers:
                    # 存储为 float16 降低内存占用，后续 stack 时再转换为 float32
                    X[l].append(out[l][-1].to(torch.float16).numpy())
            return X, np.asarray(y, dtype=np.int64)

        X_train_all, y_train = collect_features(train_idx, "Train Features")
        X_val_all, y_val = collect_features(val_idx, "Valid Features")

        results = []
        for l in layers:
            print(f"Training layer {l} (MLP + Scaler)...")
            X_train = np.stack(X_train_all[l])
            X_val = np.stack(X_val_all[l])
            X_train = X_train.astype(np.float32, copy=False)
            X_val = X_val.astype(np.float32, copy=False)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            clf = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                alpha=0.01,  # 正则化，抑制过拟合/极端概率
                random_state=seed,
                early_stopping=True,
            )
            clf.fit(X_train_scaled, y_train)

            y_prob = clf.predict_proba(X_val_scaled)[:, 1]
            y_pred = clf.predict(X_val_scaled)

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5

            print(
                f"Layer {l} | Acc: {acc:.4f} | AUC: {auc:.4f} | "
                f"Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
            )
            results.append({"layer": l, "acc": acc, "auc": auc, "precision": prec, "recall": rec, "f1": f1})

            if args.token == "last_q_token":
                out_name = f"clf_{friendly}_{args.dataset}_prefill_seed-{seed}_layer-{l}_token-{args.token}.pkl"
            else:
                # keep non-prefill tokens separate from prefill naming
                out_name = f"clf_{friendly}_{args.dataset}_seed-{seed}_layer-{l}_token-{args.token}.pkl"
            with open(os.path.join(args.save_clf_dir, out_name), "wb") as f:
                pickle.dump({"clf": clf, "scaler": scaler}, f)

        res_df = pd.DataFrame(results)
        res_df.to_csv(
            os.path.join(args.out_dir, f"{friendly}_{args.dataset}_mlp_summary_seed-{seed}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
