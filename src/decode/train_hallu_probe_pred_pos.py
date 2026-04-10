#!/usr/bin/env python
"""
训练 Hallucination Probe，使用 Position Probe 预测的位置，而不是 GT 位置。
这样训练和推理条件更一致。
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from baukit import TraceDict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from probing_utils import (
    load_model_and_validate_gpu,
    LIST_OF_MODELS,
    MODEL_FRIENDLY_NAMES,
    LAYERS_TO_TRACE,
    get_indices_of_exact_answer,
    tokenize,
    find_prompt_end_in_full_ids,
    get_probing_layer_names,
)

def find_final_token_with_lookahead(probs, threshold=0.9, lookahead=6):
    """使用 lookahead 策略找到最后一个关键 token 的位置"""
    tmp = None
    for i, p in enumerate(probs):
        if p > threshold:
            tmp = i
            found_higher = False
            for j in range(i + 1, min(i + lookahead + 1, len(probs))):
                if probs[j] > threshold:
                    found_higher = True
                    break
            if not found_higher:
                return tmp
    return tmp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/var/wangyicheng/models/Qwen3-8B")
    parser.add_argument("--train_dataset", type=str, default="triviaqa")
    parser.add_argument("--test_dataset", type=str, default="triviaqa_test")
    parser.add_argument("--pos_probe_path", type=str,
                        default="../../probe/qwen3-8b_triviaqa_pure.joblib",
                        help="Position Probe 路径（用于预测位置）")
    parser.add_argument("--pos_probe_layer", type=int, default=15,
                        help="Position Probe 使用的层")
    parser.add_argument("--hallu_layer", type=int, default=21,
                        help="Hallucination Probe 使用的层")
    parser.add_argument("--pos_threshold", type=float, default=0.9)
    parser.add_argument("--pos_lookahead", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Maximum samples to process (0=all)")
    parser.add_argument("--out_dir", type=str, default="../../probe/decode")
    return parser.parse_args()

def forward_on_ids(model, input_ids_1d: torch.Tensor, layer_names: list[str]):
    """
    返回每层的 MLP 输出激活，格式为 list of [seq_len, hidden_dim] (CPU)。
    """
    input_ids = input_ids_1d.to(model.device).unsqueeze(0)
    with torch.no_grad():
        with TraceDict(model, layers=layer_names) as td:
            _ = model(input_ids=input_ids, use_cache=False, output_hidden_states=False)
        feats = []
        for name in layer_names:
            act = td[name].output  # [1, seq_len, hidden]
            feats.append(act[0].detach().cpu())
        return feats

def main():
    args = parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    friendly = MODEL_FRIENDLY_NAMES.get(args.model, args.model.split("/")[-1])
    print(f"Model: {friendly}")
    print(f"Train dataset: {args.train_dataset}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Position Probe: {args.pos_probe_path}")
    print(f"Position Probe Layer: {args.pos_probe_layer}")
    print(f"Hallucination Layer: {args.hallu_layer}")
    print(f"Position threshold: {args.pos_threshold}, lookahead: {args.pos_lookahead}")

    # 1. 加载 Position Probe
    print("\n加载 Position Probe...")
    pos_probe_bundle = joblib.load(args.pos_probe_path)
    pos_clf = pos_probe_bundle['clf']
    pos_pca = pos_probe_bundle.get('pca')
    pos_scaler = pos_probe_bundle.get('scaler')
    print(f"  Position Probe loaded: clf={type(pos_clf).__name__}, pca={pos_pca is not None}, scaler={pos_scaler is not None}")

    # 2. 加载模型
    print("\n加载模型...")
    model, tokenizer = load_model_and_validate_gpu(args.model)
    model.eval()

    # 获取层配置
    layer_names = get_probing_layer_names('mlp', args.model)
    pos_probe_layer_idx = args.pos_probe_layer
    hallu_layer_idx = args.hallu_layer

    print(f"  Position Probe layer index: {pos_probe_layer_idx}")
    print(f"  Hallucination Probe layer index: {hallu_layer_idx}")

    # 3. 处理训练数据
    print(f"\n处理训练数据 ({args.train_dataset})...")
    train_df = pd.read_csv(f"../output/{friendly}-answers-{args.train_dataset}.csv")
    train_ids_all = torch.load(f"../output/{friendly}-input_output_ids-{args.train_dataset}.pt", map_location='cpu')

    # 过滤有效样本
    train_df = train_df[(train_df['valid_exact_answer'] == 1) & (train_df['exact_answer'] != 'NO ANSWER')]
    if args.max_samples > 0:
        train_df = train_df.sample(n=min(args.max_samples, len(train_df)), random_state=0)
    print(f"  有效样本数: {len(train_df)}")

    train_features = []
    train_labels = []
    train_pred_positions = []
    train_gt_positions = []
    n_pos_failed = 0

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="提取训练集特征"):
        try:
            # 获取 IDs
            ids = train_ids_all[int(idx)]
            if isinstance(ids, dict):
                input_ids = ids.get('all_input_ids', ids.get('input_ids'))
                output_ids = ids.get('all_output_ids', ids.get('output_ids'))
                ids = torch.cat([input_ids.squeeze(), output_ids.squeeze()], dim=0) if output_ids is not None else input_ids.squeeze()

            label = int(row['automatic_correctness'] == 1)

            # 找到 prompt 结束位置
            prompt = str(row["question"])
            prompt_ids_1d = tokenize(prompt, tokenizer, args.model)[0]
            q_len = find_prompt_end_in_full_ids(ids, prompt_ids_1d, allow_bos_mismatch=True)
            output_ids_1d = ids[int(q_len):]

            # 计算 GT span
            span = get_indices_of_exact_answer(tokenizer, ids, row['exact_answer'], args.model, output_ids=output_ids_1d)
            if not span or len(span) == 0:
                continue
            gt_last_pos_abs = int(span[-1])  # 绝对位置
            gt_last_pos_rel = gt_last_pos_abs - int(q_len)  # 相对于 answer 的位置

            # 提取 activations
            hiddens = forward_on_ids(model, ids, layer_names)

            # 准备 Position Probe 特征（只用 answer 部分）
            n_answer_tokens = len(ids) - int(q_len)
            if n_answer_tokens <= 0:
                continue

            # 提取 Position Probe 层的特征
            pos_feats = hiddens[pos_probe_layer_idx][int(q_len):].to(torch.float32).numpy()

            # 应用 PCA 和 Scaler
            if pos_scaler is not None:
                pos_feats = pos_scaler.transform(pos_feats)
            if pos_pca is not None:
                pos_feats = pos_pca.transform(pos_feats)

            # 预测每个 token 的置信度
            pos_probs = pos_clf.predict_proba(pos_feats)[:, 1]

            # 用 lookahead 策略找到预测的位置（相对位置）
            pred_pos_rel = find_final_token_with_lookahead(pos_probs, args.pos_threshold, args.pos_lookahead)

            if pred_pos_rel is None:
                n_pos_failed += 1
                continue

            # 提取 Hallucination Probe 特征（在预测位置）
            if pred_pos_rel >= n_answer_tokens:
                pred_pos_rel = n_answer_tokens - 1

            pred_pos_abs = pred_pos_rel + int(q_len)
            hallu_feat = hiddens[hallu_layer_idx][pred_pos_abs].to(torch.float32).numpy()

            train_features.append(hallu_feat)
            train_labels.append(label)
            train_pred_positions.append(pred_pos_rel)
            train_gt_positions.append(gt_last_pos_rel)

        except Exception as e:
            continue

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print(f"  提取到 {len(train_features)} 个训练样本")
    print(f"  正样本: {sum(train_labels)}, 负样本: {len(train_labels) - sum(train_labels)}")
    print(f"  Position Probe 预测失败: {n_pos_failed}")

    # 位置预测准确率
    if len(train_pred_positions) > 0:
        pos_errors = np.array(train_pred_positions) - np.array(train_gt_positions)
        exact_match = np.mean(pos_errors == 0) * 100
        within_1 = np.mean(np.abs(pos_errors) <= 1) * 100
        print(f"  训练集位置预测准确率: 精确={exact_match:.1f}%, ±1内={within_1:.1f}%")

    # 4. 训练 Hallucination Probe
    print("\n训练 Hallucination Probe...")

    # 标准化
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    # 训练 Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(train_features_scaled, train_labels)

    # 训练集评估
    train_probs = clf.predict_proba(train_features_scaled)[:, 1]
    train_auc = roc_auc_score(train_labels, train_probs)
    train_acc = accuracy_score(train_labels, clf.predict(train_features_scaled))
    train_bal_acc = balanced_accuracy_score(train_labels, clf.predict(train_features_scaled))

    print(f"  训练集 AUC: {train_auc:.4f}")
    print(f"  训练集 Accuracy: {train_acc:.4f}")
    print(f"  训练集 Balanced Accuracy: {train_bal_acc:.4f}")

    # 5. 测试集评估
    print(f"\n处理测试数据 ({args.test_dataset})...")
    test_df = pd.read_csv(f"../output/{friendly}-answers-{args.test_dataset}.csv")
    test_ids_all = torch.load(f"../output/{friendly}-input_output_ids-{args.test_dataset}.pt", map_location='cpu')

    test_df = test_df[(test_df['valid_exact_answer'] == 1) & (test_df['exact_answer'] != 'NO ANSWER')]
    print(f"  有效样本数: {len(test_df)}")

    test_features = []
    test_labels = []
    test_pred_positions = []
    test_gt_positions = []
    n_test_pos_failed = 0

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="提取测试集特征"):
        try:
            ids = test_ids_all[int(idx)]
            if isinstance(ids, dict):
                input_ids = ids.get('all_input_ids', ids.get('input_ids'))
                output_ids = ids.get('all_output_ids', ids.get('output_ids'))
                ids = torch.cat([input_ids.squeeze(), output_ids.squeeze()], dim=0) if output_ids is not None else input_ids.squeeze()

            label = int(row['automatic_correctness'] == 1)

            prompt = str(row["question"])
            prompt_ids_1d = tokenize(prompt, tokenizer, args.model)[0]
            q_len = find_prompt_end_in_full_ids(ids, prompt_ids_1d, allow_bos_mismatch=True)
            output_ids_1d = ids[int(q_len):]

            span = get_indices_of_exact_answer(tokenizer, ids, row['exact_answer'], args.model, output_ids=output_ids_1d)
            if not span or len(span) == 0:
                continue
            gt_last_pos_abs = int(span[-1])
            gt_last_pos_rel = gt_last_pos_abs - int(q_len)

            hiddens = forward_on_ids(model, ids, layer_names)

            n_answer_tokens = len(ids) - int(q_len)
            if n_answer_tokens <= 0:
                continue

            pos_feats = hiddens[pos_probe_layer_idx][int(q_len):].to(torch.float32).numpy()

            if pos_scaler is not None:
                pos_feats = pos_scaler.transform(pos_feats)
            if pos_pca is not None:
                pos_feats = pos_pca.transform(pos_feats)

            pos_probs = pos_clf.predict_proba(pos_feats)[:, 1]
            pred_pos_rel = find_final_token_with_lookahead(pos_probs, args.pos_threshold, args.pos_lookahead)

            if pred_pos_rel is None:
                n_test_pos_failed += 1
                continue

            if pred_pos_rel >= n_answer_tokens:
                pred_pos_rel = n_answer_tokens - 1

            pred_pos_abs = pred_pos_rel + int(q_len)
            hallu_feat = hiddens[hallu_layer_idx][pred_pos_abs].to(torch.float32).numpy()

            test_features.append(hallu_feat)
            test_labels.append(label)
            test_pred_positions.append(pred_pos_rel)
            test_gt_positions.append(gt_last_pos_rel)

        except Exception as e:
            continue

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    print(f"  提取到 {len(test_features)} 个测试样本")
    print(f"  正样本: {sum(test_labels)}, 负样本: {len(test_labels) - sum(test_labels)}")
    print(f"  Position Probe 预测失败: {n_test_pos_failed}")

    # 位置预测准确率
    if len(test_pred_positions) > 0:
        pos_errors = np.array(test_pred_positions) - np.array(test_gt_positions)
        exact_match = np.mean(pos_errors == 0) * 100
        within_1 = np.mean(np.abs(pos_errors) <= 1) * 100
        print(f"  测试集位置预测准确率: 精确={exact_match:.1f}%, ±1内={within_1:.1f}%")

    # 测试集评估
    if len(test_features) > 0:
        test_features_scaled = scaler.transform(test_features)
        test_probs = clf.predict_proba(test_features_scaled)[:, 1]
        test_auc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) == 2 else float('nan')
        test_acc = accuracy_score(test_labels, clf.predict(test_features_scaled))
        test_bal_acc = balanced_accuracy_score(test_labels, clf.predict(test_features_scaled))

        print(f"\n  测试集 AUC: {test_auc:.4f}")
        print(f"  测试集 Accuracy: {test_acc:.4f}")
        print(f"  测试集 Balanced Accuracy: {test_bal_acc:.4f}")
    else:
        test_auc = float('nan')
        test_acc = float('nan')
        test_bal_acc = float('nan')
        print("\n  测试集无有效样本")

    # 6. 保存模型
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"clf_{friendly}_{args.train_dataset}_pred_pos_layer-{hallu_layer_idx}.pkl")

    save_dict = {
        'clf': clf,
        'scaler': scaler,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'pos_threshold': args.pos_threshold,
        'pos_lookahead': args.pos_lookahead,
        'hallu_layer': hallu_layer_idx,
        'pos_probe_path': args.pos_probe_path,
    }
    joblib.dump(save_dict, out_path)
    print(f"\n模型已保存到: {out_path}")

    # 7. 对比总结
    print("\n" + "=" * 60)
    print("训练结果总结")
    print("=" * 60)
    print(f"Pred-Pos Hallucination Probe (Layer {hallu_layer_idx}):")
    print(f"  - 训练集 AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
    print(f"  - 测试集 AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}")
    print()
    print("与 GT-Probe 对比:")
    print("  - GT-Probe: 在 GT 位置训练和测试，AUC ≈ 0.82")
    print(f"  - Pred-Probe: 在预测位置训练和测试，AUC = {test_auc:.4f}")
    print(f"  - 差距: {0.82 - test_auc:.4f}" if not np.isnan(test_auc) else "  - 差距: N/A")
    print()
    print("说明:")
    print("  Pred-Probe 的训练和推理条件一致（都用预测位置），")
    print("  更接近实际部署时的表现。")

if __name__ == "__main__":
    main()
