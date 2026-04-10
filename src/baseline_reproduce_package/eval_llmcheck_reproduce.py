"""
LLM-Check 复现脚本 (NeurIPS 2024)

复现论文: LLM-Check: Investigating Detection of Hallucinations in Large Language Models
在 TriviaQA / Winobias 数据集 + Qwen3-8B / Llama3-8B 模型上评估

LLM-Check 提出了三类无需训练的幻觉检测方法：
1. Logit-based: Perplexity, Window Entropy, Logit Entropy (Top-K)
2. Hidden-based: 对隐藏状态进行 SVD 分析，计算特征值
3. Attention-based: 对注意力矩阵计算对角元素的对数均值

用法:
    python eval_llmcheck_reproduce.py --model qwen3 --dataset triviaqa_test --max_samples 200
    python eval_llmcheck_reproduce.py --model llama3 --dataset winobias_test --max_samples 200
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle as pkl

sys.path.insert(0, '..')
from probing_utils import MODEL_FRIENDLY_NAMES, find_prompt_end_in_full_ids, tokenize

# 模型路径映射
MODEL_PATHS = {
    'qwen3': '/var/wangyicheng/models/Qwen3-8B',
    'llama3': '/var/luzhenyan/Meta-Llama-3-8B-Instruct',
}

FRIENDLY_NAMES = {
    'qwen3': 'qwen3-8b',
    'llama3': 'llama3-8b-instruct',
}


# ============================================
# LLM-Check 核心评分函数 (来自原论文代码)
# ============================================

def get_model_vals(model, tok_in):
    """获取模型的 logits, hidden states, attentions"""
    kwargs = {
        "input_ids": tok_in,
        "use_cache": False,
        "past_key_values": None,
        "output_attentions": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    with torch.no_grad():
        output = model(**kwargs)
    return output.logits, output.hidden_states, output.attentions


def centered_svd_val(Z, alpha=0.001):
    """计算中心化协方差矩阵的对数奇异值均值
    
    这是 LLM-Check 的核心 Hidden Score 计算方法
    """
    # Z: [hidden_dim, seq_len] 转置后的隐藏状态
    J = torch.eye(Z.shape[0], device=Z.device, dtype=Z.dtype) - \
        (1 / Z.shape[0]) * torch.ones(Z.shape[0], Z.shape[0], device=Z.device, dtype=Z.dtype)
    Sigma = torch.matmul(torch.matmul(Z.t(), J), Z)
    Sigma = Sigma + alpha * torch.eye(Sigma.shape[0], device=Z.device, dtype=Z.dtype)
    svdvals = torch.linalg.svdvals(Sigma)
    eigscore = torch.log(svdvals).mean()
    return eigscore.item()


def get_svd_eval(hidden_act, layer_num, tok_start, tok_end):
    """对指定层的隐藏状态进行 SVD 评估
    
    Args:
        hidden_act: 模型某一层的隐藏状态 [seq_len, hidden_dim]
        layer_num: 层索引（仅用于日志）
        tok_start: 开始 token 位置
        tok_end: 结束 token 位置
    
    Returns:
        float: SVD-based score
    """
    Z = hidden_act[tok_start:tok_end, :]  # 只取答案部分
    if Z.shape[0] < 2:
        return 0.0
    Z = Z.T.float().cpu()  # [hidden_dim, seq_len]
    return centered_svd_val(Z)


def get_attn_eig_prod(attns_layer, tok_start, tok_end):
    """计算注意力矩阵的对角元素对数均值
    
    这是 LLM-Check 的 Attention Score 计算方法
    
    Args:
        attns_layer: 某层的注意力矩阵 [num_heads, seq_len, seq_len]
        tok_start: 开始位置
        tok_end: 结束位置
    
    Returns:
        float: Attention eigenvalue score
    """
    eigscore = 0.0
    num_heads = attns_layer.shape[0]
    
    for head_idx in range(num_heads):
        Sigma = attns_layer[head_idx, tok_start:tok_end, tok_start:tok_end]
        if Sigma.shape[0] > 0:
            # 取对角元素的对数均值
            diag = torch.diagonal(Sigma, 0)
            # 避免 log(0)
            diag = torch.clamp(diag, min=1e-10)
            eigscore += torch.log(diag).mean().item()
    
    return eigscore / max(num_heads, 1)


def perplexity(logits, tok_in, tok_start, tok_end):
    """计算困惑度
    
    Args:
        logits: 模型输出 [seq_len, vocab_size]
        tok_in: 输入 token IDs [1, seq_len]
        tok_start: 答案开始位置
        tok_end: 答案结束位置
    
    Returns:
        float: Perplexity score
    """
    softmax = torch.nn.Softmax(dim=-1)
    
    # 对答案部分计算困惑度
    # logits[i] 预测的是 tok_in[i+1]
    pr = torch.log(softmax(logits))[torch.arange(tok_start, tok_end) - 1, tok_in[0, tok_start:tok_end]]
    ppl = torch.exp(-pr.mean()).item()
    
    return ppl


def logit_entropy(logits, tok_start, tok_end, top_k=50):
    """计算 Top-K Logit Entropy
    
    Args:
        logits: 模型输出 [seq_len, vocab_size]
        tok_start: 答案开始位置
        tok_end: 答案结束位置
        top_k: 只考虑 top-k 个 token
    
    Returns:
        float: Entropy score
    """
    softmax = torch.nn.Softmax(dim=-1)
    
    l = logits[tok_start:tok_end]
    l = softmax(torch.topk(l, top_k, 1).values)
    entropy = (-l * torch.log(l + 1e-10)).mean().item()
    
    return entropy


def window_logit_entropy(logits, tok_start, tok_end, w=1):
    """计算窗口化最大熵
    
    Args:
        logits: 模型输出 [seq_len, vocab_size]
        tok_start: 答案开始位置
        tok_end: 答案结束位置
        w: 窗口大小
    
    Returns:
        float: Max windowed entropy
    """
    softmax = torch.nn.Softmax(dim=-1)
    
    l = softmax(logits[tok_start:tok_end])
    if l.shape[0] < w:
        return (-l * torch.log(l + 1e-10)).mean().item()
    
    # 每个 token 的熵
    per_token_entropy = (-l * torch.log(l + 1e-10)).mean(dim=1)
    
    # 滑动窗口取最大
    if per_token_entropy.shape[0] >= w:
        windows = per_token_entropy.unfold(0, w, w).mean(1)
        return windows.max().item()
    else:
        return per_token_entropy.mean().item()


def get_roc_scores(scores, labels):
    """计算 ROC-AUC 和准确率"""
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 处理 NaN
    valid = ~np.isnan(scores)
    if valid.sum() < len(scores) * 0.5:
        return 0.5, 0.5, 0.0
    
    scores = scores[valid]
    labels = labels[valid]
    
    fpr, tpr, _ = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    
    # TPR at 5% FPR
    low_fpr_idx = np.where(fpr < 0.05)[0]
    low = tpr[low_fpr_idx[-1]] if len(low_fpr_idx) > 0 else 0.0
    
    return arc, acc, low


# ============================================
# 主程序
# ============================================

def load_model_and_tokenizer(model_path, use_eager_attention=False):
    """加载模型和分词器"""
    print(f"Loading model from {model_path}...")
    
    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    
    # 如果需要获取 attention 输出，使用 eager 实现
    if use_eager_attention:
        kwargs["attn_implementation"] = "eager"
        print("Using eager attention implementation for attention output")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    model.requires_grad_(False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="LLM-Check Reproduction on TriviaQA/Winobias")
    parser.add_argument("--model", type=str, required=True, choices=['qwen3', 'llama3'],
                        help="Model to use")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., triviaqa_test, winobias_test)")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--methods", type=str, nargs='+', 
                        default=['logit', 'hidden', 'attns'],
                        choices=['logit', 'hidden', 'attns'],
                        help="Methods to evaluate")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=None,
                        help="Specific hidden layers to evaluate (default: all)")
    parser.add_argument("--attn_layers", type=int, nargs='+', default=None,
                        help="Specific attention layers to evaluate (default: all)")
    parser.add_argument("--output_dir", type=str, default="../output/llmcheck_scores",
                        help="Output directory for scores")
    parser.add_argument("--balanced", action="store_true",
                        help="Use balanced sampling (equal correct/incorrect)")
    args = parser.parse_args()
    
    model_path = MODEL_PATHS[args.model]
    friendly = FRIENDLY_NAMES[args.model]
    
    print("=" * 70)
    print(f"LLM-Check 幻觉检测评估")
    print(f"模型: {args.model.upper()} ({friendly})")
    print(f"数据集: {args.dataset}")
    print(f"方法: {args.methods}")
    print("=" * 70)
    
    # 加载数据
    print("\n加载数据...")
    csv_file = f"../output/{friendly}-answers-{args.dataset}.csv"
    ids_file = f"../output/{friendly}-input_output_ids-{args.dataset}.pt"
    
    if not os.path.exists(csv_file):
        print(f"错误: 找不到数据文件 {csv_file}")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    ids_all = torch.load(ids_file, map_location="cpu")
    
    # 可选：平衡采样
    if args.balanced and args.max_samples > 0:
        df_pos = df[df["automatic_correctness"] == 1]
        df_neg = df[df["automatic_correctness"] == 0]
        half = args.max_samples // 2
        n = min(half, len(df_pos), len(df_neg))
        df = pd.concat([
            df_pos.sample(n=n, random_state=42),
            df_neg.sample(n=n, random_state=42)
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"📊 平衡采样: {n} 正确 + {n} 错误 = {2*n} 样本")
    elif args.max_samples > 0:
        df = df.head(args.max_samples)
    
    print(f"评估样本数: {len(df)}")
    
    # 加载模型（如果需要 attention 输出，使用 eager 实现）
    use_eager = 'attns' in args.methods
    model, tokenizer = load_model_and_tokenizer(model_path, use_eager_attention=use_eager)
    num_layers = model.config.num_hidden_layers
    print(f"模型层数: {num_layers}")
    
    # 初始化评分存储
    scores_dict = {
        'logit': {
            'perplexity': [],
            'window_entropy': [],
            'logit_entropy': [],
        },
        'hidden': {f'H_layer{l}': [] for l in range(num_layers)},
        'attns': {f'A_layer{l}': [] for l in range(num_layers)},
    }
    labels = []
    
    # 确定要评估的层
    hidden_layers = args.hidden_layers if args.hidden_layers else list(range(1, num_layers))
    attn_layers = args.attn_layers if args.attn_layers else list(range(1, num_layers))
    
    print(f"\n开始评估...")
    print(f"Hidden 层: {hidden_layers[:5]}... (共 {len(hidden_layers)} 层)" if len(hidden_layers) > 5 else f"Hidden 层: {hidden_layers}")
    print(f"Attention 层: {attn_layers[:5]}... (共 {len(attn_layers)} 层)" if len(attn_layers) > 5 else f"Attention 层: {attn_layers}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        try:
            # 获取 input_ids
            input_ids = ids_all[idx]
            if isinstance(input_ids, torch.Tensor):
                input_ids_1d = input_ids
            else:
                input_ids_1d = torch.tensor(input_ids)
            
            # 确定 prompt 结束位置
            q = str(row.get("question") or row.get("raw_question", ""))
            prompt_ids = tokenize(q, tokenizer, model_path)[0]
            q_len = find_prompt_end_in_full_ids(input_ids_1d, prompt_ids, allow_bos_mismatch=True)
            q_len = int(q_len)
            
            # 答案范围
            tok_start = q_len
            tok_end = len(input_ids_1d)
            
            if tok_end <= tok_start + 1:
                # 答案太短，跳过
                continue
            
            # 前向传播获取模型输出
            tok_in = input_ids_1d.unsqueeze(0).to(model.device)
            
            # 根据需要的方法决定获取哪些输出
            need_attns = 'attns' in args.methods
            need_hidden = 'hidden' in args.methods
            
            kwargs = {
                "input_ids": tok_in,
                "use_cache": False,
                "output_attentions": need_attns,
                "output_hidden_states": need_hidden or 'logit' in args.methods,
                "return_dict": True,
            }
            
            with torch.no_grad():
                output = model(**kwargs)
            
            logits = output.logits[0].float().cpu()
            
            # 记录标签 (1 = 正确/非幻觉, 0 = 错误/幻觉)
            # LLM-Check 论文中，高分表示幻觉，所以我们需要反转
            label = 1 - int(row['automatic_correctness'])  # 1 = 幻觉, 0 = 正确
            labels.append(label)
            
            # === 1. Logit-based Scores ===
            if 'logit' in args.methods:
                ppl = perplexity(logits, tok_in.cpu(), tok_start, tok_end)
                w_ent = window_logit_entropy(logits, tok_start, tok_end, w=1)
                l_ent = logit_entropy(logits, tok_start, tok_end, top_k=50)
                
                scores_dict['logit']['perplexity'].append(ppl)
                scores_dict['logit']['window_entropy'].append(w_ent)
                scores_dict['logit']['logit_entropy'].append(l_ent)
            
            # === 2. Hidden-based Scores (SVD) ===
            if 'hidden' in args.methods and output.hidden_states is not None:
                for layer_idx in hidden_layers:
                    if layer_idx < len(output.hidden_states):
                        hidden_act = output.hidden_states[layer_idx][0].float().cpu()
                        svd_score = get_svd_eval(hidden_act, layer_idx, tok_start, tok_end)
                        scores_dict['hidden'][f'H_layer{layer_idx}'].append(svd_score)
            
            # === 3. Attention-based Scores ===
            if 'attns' in args.methods and output.attentions is not None:
                for layer_idx in attn_layers:
                    if layer_idx < len(output.attentions):
                        attn = output.attentions[layer_idx][0].float().cpu()  # [num_heads, seq, seq]
                        attn_score = get_attn_eig_prod(attn, tok_start, tok_end)
                        scores_dict['attns'][f'A_layer{layer_idx}'].append(attn_score)
            
            # 清理显存
            del output, logits
            if idx % 50 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n⚠️ 样本 {idx} 处理失败: {e}")
            # 填充 NaN
            labels.append(np.nan)
            for method in args.methods:
                for key in scores_dict.get(method, {}):
                    scores_dict[method][key].append(np.nan)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"scores_{args.dataset}_{args.model}_{len(labels)}samp.pkl")
    with open(output_file, 'wb') as f:
        pkl.dump([scores_dict, labels], f)
    print(f"\n✅ 评分已保存到: {output_file}")
    
    # ============================================
    # 结果分析
    # ============================================
    print("\n" + "=" * 70)
    print("LLM-Check 评估结果")
    print("=" * 70)
    
    labels_arr = np.array(labels)
    valid_mask = ~np.isnan(labels_arr)
    labels_valid = labels_arr[valid_mask].astype(int)
    
    print(f"\n有效样本: {valid_mask.sum()} / {len(labels)}")
    print(f"幻觉样本: {labels_valid.sum()} ({100*labels_valid.mean():.1f}%)")
    print(f"正确样本: {(1-labels_valid).sum()} ({100*(1-labels_valid.mean()):.1f}%)")
    
    results = []
    
    # 1. Logit-based 结果
    if 'logit' in args.methods:
        print("\n📊 Logit-based Scores:")
        print("-" * 50)
        for name, scores in scores_dict['logit'].items():
            scores_arr = np.array(scores)[valid_mask]
            arc, acc, low = get_roc_scores(scores_arr, labels_valid)
            print(f"  {name:<20} AUC={arc:.4f}  Acc={acc:.4f}  TPR@5%FPR={low:.4f}")
            results.append({'method': name, 'auc': arc, 'acc': acc, 'tpr_at_5fpr': low})
    
    # 2. Hidden-based 结果 (找最佳层)
    if 'hidden' in args.methods:
        print("\n📊 Hidden-based Scores (SVD):")
        print("-" * 50)
        
        best_hidden_auc = 0
        best_hidden_layer = None
        
        for layer_idx in hidden_layers:
            key = f'H_layer{layer_idx}'
            if key in scores_dict['hidden'] and len(scores_dict['hidden'][key]) > 0:
                scores_arr = np.array(scores_dict['hidden'][key])[valid_mask]
                if len(scores_arr) > 0:
                    arc, acc, low = get_roc_scores(scores_arr, labels_valid)
                    if arc > best_hidden_auc:
                        best_hidden_auc = arc
                        best_hidden_layer = layer_idx
        
        if best_hidden_layer is not None:
            key = f'H_layer{best_hidden_layer}'
            scores_arr = np.array(scores_dict['hidden'][key])[valid_mask]
            arc, acc, low = get_roc_scores(scores_arr, labels_valid)
            print(f"  Best Layer {best_hidden_layer:<10} AUC={arc:.4f}  Acc={acc:.4f}  TPR@5%FPR={low:.4f}")
            results.append({'method': f'hidden_layer{best_hidden_layer}', 'auc': arc, 'acc': acc, 'tpr_at_5fpr': low})
            
            # 显示其他一些层的结果
            sample_layers = [num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
            for l in sample_layers:
                if l != best_hidden_layer and f'H_layer{l}' in scores_dict['hidden']:
                    scores_arr = np.array(scores_dict['hidden'][f'H_layer{l}'])[valid_mask]
                    if len(scores_arr) > 0:
                        arc, _, _ = get_roc_scores(scores_arr, labels_valid)
                        print(f"  Layer {l:<15} AUC={arc:.4f}")
    
    # 3. Attention-based 结果 (找最佳层)
    if 'attns' in args.methods:
        print("\n📊 Attention-based Scores:")
        print("-" * 50)
        
        best_attn_auc = 0
        best_attn_layer = None
        
        for layer_idx in attn_layers:
            key = f'A_layer{layer_idx}'
            if key in scores_dict['attns'] and len(scores_dict['attns'][key]) > 0:
                scores_arr = np.array(scores_dict['attns'][key])[valid_mask]
                if len(scores_arr) > 0:
                    arc, acc, low = get_roc_scores(scores_arr, labels_valid)
                    if arc > best_attn_auc:
                        best_attn_auc = arc
                        best_attn_layer = layer_idx
        
        if best_attn_layer is not None:
            key = f'A_layer{best_attn_layer}'
            scores_arr = np.array(scores_dict['attns'][key])[valid_mask]
            arc, acc, low = get_roc_scores(scores_arr, labels_valid)
            print(f"  Best Layer {best_attn_layer:<10} AUC={arc:.4f}  Acc={acc:.4f}  TPR@5%FPR={low:.4f}")
            results.append({'method': f'attn_layer{best_attn_layer}', 'auc': arc, 'acc': acc, 'tpr_at_5fpr': low})
            
            # 显示其他一些层的结果
            sample_layers = [num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
            for l in sample_layers:
                if l != best_attn_layer and f'A_layer{l}' in scores_dict['attns']:
                    scores_arr = np.array(scores_dict['attns'][f'A_layer{l}'])[valid_mask]
                    if len(scores_arr) > 0:
                        arc, _, _ = get_roc_scores(scores_arr, labels_valid)
                        print(f"  Layer {l:<15} AUC={arc:.4f}")
    
    # ============================================
    # 组合评分 (LLM-Check 的核心)
    # ============================================
    print("\n📊 组合评分 (LLM-Check Suite):")
    print("-" * 50)
    
    # 收集所有可用的评分
    all_scores = []
    score_names = []
    
    if 'logit' in args.methods:
        for name, scores in scores_dict['logit'].items():
            scores_arr = np.array(scores)[valid_mask]
            if not np.all(np.isnan(scores_arr)):
                all_scores.append(scores_arr)
                score_names.append(name)
    
    if 'hidden' in args.methods and best_hidden_layer is not None:
        scores_arr = np.array(scores_dict['hidden'][f'H_layer{best_hidden_layer}'])[valid_mask]
        all_scores.append(scores_arr)
        score_names.append(f'hidden_L{best_hidden_layer}')
    
    if 'attns' in args.methods and best_attn_layer is not None:
        scores_arr = np.array(scores_dict['attns'][f'A_layer{best_attn_layer}'])[valid_mask]
        all_scores.append(scores_arr)
        score_names.append(f'attn_L{best_attn_layer}')
    
    if len(all_scores) > 1:
        # 归一化每个评分到 [0, 1]
        normalized_scores = []
        for scores in all_scores:
            s_min, s_max = np.nanmin(scores), np.nanmax(scores)
            if s_max > s_min:
                normalized = (scores - s_min) / (s_max - s_min)
            else:
                normalized = np.zeros_like(scores)
            normalized_scores.append(normalized)
        
        # 简单平均组合
        combined_avg = np.nanmean(normalized_scores, axis=0)
        arc, acc, low = get_roc_scores(combined_avg, labels_valid)
        print(f"  Average Ensemble       AUC={arc:.4f}  Acc={acc:.4f}")
        results.append({'method': 'ensemble_avg', 'auc': arc, 'acc': acc, 'tpr_at_5fpr': low})
        
        # 最大值组合
        combined_max = np.nanmax(normalized_scores, axis=0)
        arc, acc, low = get_roc_scores(combined_max, labels_valid)
        print(f"  Max Ensemble           AUC={arc:.4f}  Acc={acc:.4f}")
        results.append({'method': 'ensemble_max', 'auc': arc, 'acc': acc, 'tpr_at_5fpr': low})
    
    # 保存汇总结果
    results_df = pd.DataFrame(results)
    results_file = os.path.join(args.output_dir, f"results_{args.dataset}_{args.model}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ 汇总结果已保存到: {results_file}")
    
    # ============================================
    # 与 Probe 方法对比
    # ============================================
    print("\n" + "=" * 70)
    print("📈 与 Probe 方法的对比（来自 README_probe_fix.md）")
    print("=" * 70)
    
    if args.model == 'qwen3' and 'triviaqa' in args.dataset:
        print("""
Qwen3-8B + TriviaQA 参考结果:
  Probe 方法:
    - Prefill Probe (L20):        AUC ≈ 0.65
    - Decode Probe (Lookahead):   AUC ≈ 0.84
    - Decode Probe (GT):          AUC ≈ 0.85
    - Combined (0.3P + 0.7D):     AUC ≈ 0.86
""")
    elif args.model == 'qwen3' and 'winobias' in args.dataset:
        print("""
Qwen3-8B + Winobias 参考结果:
  Probe 方法:
    - Prefill Probe (L22):        AUC ≈ 0.77
    - Decode Probe (Lookahead):   AUC ≈ 0.86
    - Decode Probe (GT):          AUC ≈ 0.86
    - Combined (0.3P + 0.7D):     AUC ≈ 0.87
""")
    elif args.model == 'llama3' and 'winobias' in args.dataset:
        print("""
Llama3-8B + Winobias 参考结果:
  Probe 方法:
    - Prefill Probe (L25):        AUC ≈ 0.63
    - Decode Probe (Lookahead):   AUC ≈ 0.74
    - Decode Probe (GT):          AUC ≈ 0.76
    - Combined (0.7P + 0.3D):     AUC ≈ 0.75
""")
    
    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()

