"""
为每个样本绘制 Position Probe 置信度图
- 标注 GT span（红色）和预测位置（绿色边框）
- 按定位准确度分目录保存
- 使用 lookahead 策略预测关键 token 位置

用法:
    python plot_pos_probe_confidence.py \
        --model /var/wangyicheng/models/Qwen3-8B \
        --dataset winobias \
        --pos_probe_path ../../probe/qwen3-8b_winobias_pure.joblib \
        --output_dir ../../output/pos_probe_plots \
        --n_samples 100 \
        --enter_threshold 0.9 \
        --lookahead 6
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from tqdm import tqdm
from baukit import TraceDict

from probing_utils import (
    MODEL_FRIENDLY_NAMES, LAYERS_TO_TRACE,
    load_model_and_validate_gpu, tokenize,
    get_indices_of_exact_answer, find_prompt_end_in_full_ids,
    get_probing_layer_names,
)


def simulate_lookahead(probs, enter_threshold=0.9, lookahead=6):
    """
    Lookahead 策略：
    - 只要分数 >= 阈值，就更新候选位置
    - 连续 N 步 < 阈值时，返回最后的候选位置
    """
    candidate_pos = None
    steps_since_last_high = 0

    for i, score in enumerate(probs):
        if score >= enter_threshold:
            candidate_pos = i
            steps_since_last_high = 0
        else:
            if candidate_pos is not None:
                steps_since_last_high += 1
                if steps_since_last_high >= lookahead:
                    return candidate_pos

    return candidate_pos  # 序列结束，返回最后候选


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Position Probe confidence per sample")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--pos_probe_path", type=str, required=True, help="Position Probe joblib path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: ../../output/pos_probe_plots/<friendly>_<dataset>)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples per class (correct/wrong)")
    parser.add_argument("--enter_threshold", type=float, default=0.9,
                        help="Lookahead enter threshold")
    parser.add_argument("--lookahead", type=int, default=6,
                        help="Lookahead steps")
    parser.add_argument("--dpi", type=int, default=120, help="Plot DPI")
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置字体
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

    friendly = MODEL_FRIENDLY_NAMES.get(args.model, args.model.split("/")[-1])

    # 输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output = os.path.join(script_dir, '..', '..', 'output')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_output, f"pos_probe_plots/{friendly}_{args.dataset}")

    os.makedirs(f"{output_dir}/loc_correct", exist_ok=True)
    os.makedirs(f"{output_dir}/loc_wrong", exist_ok=True)
    os.makedirs(f"{output_dir}/no_gt", exist_ok=True)

    # 加载数据
    print("Loading data...")
    data_dir = os.path.join(script_dir, '..', 'output')
    df = pd.read_csv(os.path.join(data_dir, f"{friendly}-answers-{args.dataset}.csv"))
    ids_all = torch.load(os.path.join(data_dir, f"{friendly}-input_output_ids-{args.dataset}.pt"), map_location="cpu")

    # 加载模型
    print("Loading model...")
    model, tokenizer = load_model_and_validate_gpu(args.model)

    # 加载 Position Probe
    print("Loading position probe...")
    probe_dict = joblib.load(args.pos_probe_path)
    clf = probe_dict['clf']
    pca = probe_dict.get('pca')
    scaler = probe_dict.get('scaler')
    use_scaler = probe_dict.get('use_scaler', True)
    probe_at = probe_dict.get('probe_at', 'resid')
    print(f"Probe at: {probe_at}")

    # 获取 layer index
    layer_val = probe_dict.get('layer', 12)
    if isinstance(layer_val, str) and layer_val == 'last':
        layer_idx = -1
    elif isinstance(layer_val, str) and layer_val in ['first', 'all']:
        layer_idx = 0
    else:
        layer_idx = int(layer_val)
    print(f"Probe layer index: {layer_idx}")

    # 根据 probe_at 决定提取什么特征
    if probe_at == 'mlp':
        layer_names = get_probing_layer_names('mlp', args.model)
        layer_name = layer_names[layer_idx]
        print(f"Using MLP layer: {layer_name}")
    else:
        layer_name = None
        print(f"Using residual stream at layer index: {layer_idx}")

    # 取样本：正确和错误各 n_samples 个
    df_wrong = df[df["automatic_correctness"] == 0].head(args.n_samples)
    df_correct = df[df["automatic_correctness"] == 1].head(args.n_samples)
    df_sample = pd.concat([df_wrong, df_correct])
    print(f"Processing {len(df_sample)} samples ({len(df_wrong)} wrong, {len(df_correct)} correct)")

    results = {
        'correct': {'exact_match': 0, 'within_1': 0, 'within_3': 0, 'total': 0},
        'wrong': {'exact_match': 0, 'within_1': 0, 'within_3': 0, 'total': 0},
        'no_gt': 0
    }

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Processing"):
        try:
            correctness = int(row['automatic_correctness'])

            # 获取问题和 token
            q_col = "question" if ("question" in row and not pd.isna(row["question"])) else "raw_question"
            question = str(row[q_col])

            input_ids_1d = ids_all[int(idx)]

            # 使用 find_prompt_end_in_full_ids 获取准确的 q_len
            prompt_ids = tokenize(question, tokenizer, args.model)[0]
            q_len = find_prompt_end_in_full_ids(input_ids_1d, prompt_ids, allow_bos_mismatch=True)
            q_len = int(q_len)

            # 获取 answer tokens
            answer_ids = input_ids_1d[q_len:]
            answer_tokens = [tokenizer.decode([tid]) for tid in answer_ids]
            max_tokens = len(answer_tokens)

            if max_tokens <= 0:
                results['no_gt'] += 1
                continue

            # 获取特征
            input_tensor = input_ids_1d.clone().detach().unsqueeze(0).to(model.device)

            with torch.no_grad():
                if probe_at == 'mlp':
                    with TraceDict(model, [layer_name], retain_input=False, retain_output=True) as traces:
                        model(input_tensor)

                    mlp_out = traces[layer_name].output
                    if isinstance(mlp_out, tuple):
                        mlp_out = mlp_out[0]

                    features = mlp_out[0, q_len:q_len+max_tokens, :].cpu().float().numpy()
                else:
                    outputs = model(input_tensor, output_hidden_states=True)
                    hs = outputs.hidden_states[layer_idx]
                    features = hs[0, q_len:q_len+max_tokens, :].cpu().float().numpy()

            # 处理特征: scaler -> pca -> clf
            if use_scaler and scaler is not None:
                feat = scaler.transform(features)
            else:
                feat = features

            if pca is not None:
                feat = pca.transform(feat)

            # 用 Position Probe 预测置信度
            probs = clf.predict_proba(feat)[:, 1]

            # 预测位置 (lookahead 策略)
            pred_pos = simulate_lookahead(probs, args.enter_threshold, args.lookahead)
            if pred_pos is None:
                pred_pos = int(np.argmax(probs))

            # 获取 GT span
            exact_answer = row.get("exact_answer", None)
            valid_exact = row.get("valid_exact_answer", None)

            gt_span = []
            gt_last_token = None
            if exact_answer and valid_exact == 1:
                output_ids = input_ids_1d[q_len:]
                span = get_indices_of_exact_answer(tokenizer, input_ids_1d, exact_answer, args.model, output_ids=output_ids)
                if span:
                    gt_span = [s - q_len for s in span if 0 <= s - q_len < max_tokens]
                    if gt_span:
                        gt_last_token = gt_span[-1]

            # 确定分类目录和统计
            if not gt_span:
                loc_category = "no_gt"
                results['no_gt'] += 1
                error = None
            else:
                error = abs(pred_pos - gt_last_token)
                loc_category = "loc_correct" if error <= 1 else "loc_wrong"

                category = "correct" if correctness == 1 else "wrong"
                results[category]['total'] += 1

                if error == 0:
                    results[category]['exact_match'] += 1
                if error <= 1:
                    results[category]['within_1'] += 1
                if error <= 3:
                    results[category]['within_3'] += 1

            # 绘图
            fig, ax = plt.subplots(figsize=(16, 6))

            x = np.arange(max_tokens)
            colors = ['steelblue'] * max_tokens
            edgecolors = ['none'] * max_tokens
            linewidths = [0] * max_tokens

            # 标注 GT span (红色)
            for pos in gt_span:
                if pos < max_tokens:
                    colors[pos] = 'red'

            # 标注预测位置 (绿色边框)
            if pred_pos < max_tokens:
                edgecolors[pred_pos] = 'green'
                linewidths[pred_pos] = 3

            ax.bar(x, probs, color=colors, alpha=0.7, edgecolor=edgecolors, linewidth=linewidths)

            # 阈值线
            ax.axhline(y=args.enter_threshold, color='orange', linestyle='--', alpha=0.5,
                        label=f'Threshold {args.enter_threshold}')

            # 设置 x 轴标签
            labels = [f"{i}:{tok[:8]}" for i, tok in enumerate(answer_tokens)]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

            ax.set_xlabel('Token Position: Token Text', fontsize=10)
            ax.set_ylabel('Position Probe Confidence', fontsize=10)
            ax.set_ylim(0, 1.05)

            # 标题
            match_status = "N/A"
            if gt_last_token is not None:
                err = abs(pred_pos - gt_last_token)
                if err == 0:
                    match_status = "EXACT"
                elif err <= 1:
                    match_status = f"~{err}"
                else:
                    match_status = f"X diff={err}"

            title = f"Sample {idx} | Correctness={correctness} | exact_answer='{exact_answer}'\n"
            title += f"GT last={gt_last_token} | Pred={pred_pos} | {match_status}"
            ax.set_title(title, fontsize=10)

            # 图例
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label='GT span'),
                Patch(facecolor='steelblue', edgecolor='green', linewidth=2, alpha=0.7,
                      label=f'Lookahead pred (thr={args.enter_threshold}, la={args.lookahead})'),
                plt.Line2D([0], [0], color='orange', linestyle='--', label=f'Threshold {args.enter_threshold}')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/{loc_category}/sample_{idx}.png", dpi=args.dpi, bbox_inches='tight')
            plt.close()

        except Exception as e:
            import traceback
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            continue

    # 打印统计结果
    print("\n" + "=" * 60)
    print("Position Probe Localization Results")
    print("=" * 60)

    for cat in ['correct', 'wrong']:
        r = results[cat]
        if r['total'] > 0:
            print(f"\n{cat.upper()} samples (n={r['total']}):")
            print(f"  Exact match: {r['exact_match']} ({100*r['exact_match']/r['total']:.1f}%)")
            print(f"  Within +/-1: {r['within_1']} ({100*r['within_1']/r['total']:.1f}%)")
            print(f"  Within +/-3: {r['within_3']} ({100*r['within_3']/r['total']:.1f}%)")

    print(f"\nNo GT samples: {results['no_gt']}")
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
