"""
EigenScore (INSIDE) 复现脚本 (ICLR 2024)

复现论文: INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection
在 TriviaQA / Winobias 数据集 + Qwen3-8B / Llama3-8B 模型上评估

EigenScore 的核心思想：
1. 对同一问题生成多个回复（通过采样）
2. 提取每个回复的隐藏状态 embedding
3. 计算这些 embedding 的协方差矩阵
4. 对协方差矩阵进行 SVD 分解
5. 用特征值的对数均值作为幻觉分数（高散度 = 可能幻觉）

与 LLM-Check 的区别：
- LLM-Check: 单次生成，分析输出不确定性
- EigenScore: 多次采样，分析回复间的语义一致性

用法:
    python eval_eigenscore_reproduce.py --model qwen3 --dataset triviaqa_test --max_samples 100 --num_generations 5
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
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
# EigenScore 核心评分函数 (来自原论文代码)
# ============================================

def get_perplexity_score(logits_list):
    """计算困惑度
    
    Args:
        logits_list: list of logits tensors, 每个是 [vocab_size]
    """
    perplexity = 0.0
    for logits in logits_list:
        if logits.dim() == 1:
            conf = torch.max(logits.softmax(0)).cpu().item()
        else:
            conf = torch.max(logits.softmax(-1)).cpu().item()
        perplexity += np.log(conf + 1e-10)
    perplexity = -1.0 * perplexity / max(len(logits_list), 1)
    return perplexity


def get_energy_score(logits_list):
    """计算 Energy Score（基于 logsumexp）"""
    avg_energy = 0.0
    for logits in logits_list:
        if logits.dim() > 1:
            logits = logits[0]
        energy = -torch.logsumexp(logits, dim=0, keepdim=False).item()
        avg_energy += energy
    return avg_energy / max(len(logits_list), 1)


def get_eigenscore_from_embeddings(embeddings, alpha=1e-3):
    """计算 EigenScore：embedding 协方差矩阵的特征值对数均值
    
    Args:
        embeddings: numpy array [num_samples, hidden_dim]
        alpha: 正则化系数
    
    Returns:
        float: EigenScore (特征值对数均值)
    """
    if embeddings.shape[0] < 2:
        return 0.0
    
    # 计算协方差矩阵
    CovMatrix = np.cov(embeddings)  # [num_samples, num_samples]
    
    # 确保协方差矩阵是方阵
    if CovMatrix.ndim == 0:
        return 0.0
    
    # 添加正则化
    CovMatrix = CovMatrix + alpha * np.eye(CovMatrix.shape[0])
    
    # SVD 分解
    try:
        u, s, vT = np.linalg.svd(CovMatrix)
        # 特征值对数均值
        eigenScore = np.mean(np.log10(s + 1e-10))
    except:
        eigenScore = 0.0
    
    return eigenScore


def get_lexical_similarity(generated_texts):
    """计算生成文本之间的词汇相似度（使用简化的 ROUGE-L）"""
    if len(generated_texts) < 2:
        return 0.0
    
    def lcs_length(s1, s2):
        """计算最长公共子序列长度"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    def rouge_l(s1, s2):
        """计算 ROUGE-L F1"""
        words1 = s1.lower().split()
        words2 = s2.lower().split()
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        lcs = lcs_length(words1, words2)
        precision = lcs / len(words1)
        recall = lcs / len(words2)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    total_sim = 0
    count = 0
    for i in range(len(generated_texts)):
        for j in range(i + 1, len(generated_texts)):
            total_sim += rouge_l(generated_texts[i], generated_texts[j])
            count += 1
    
    return total_sim / max(count, 1)


def get_roc_scores(scores, labels):
    """计算 ROC-AUC 和准确率"""
    scores = np.array(scores)
    labels = np.array(labels)
    
    valid = ~np.isnan(scores) & ~np.isnan(labels)
    if valid.sum() < 10:
        return 0.5, 0.5, 0.0
    
    scores = scores[valid]
    labels = labels[valid]
    
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        arc = auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr)) / 2)
        low_fpr_idx = np.where(fpr < 0.05)[0]
        low = tpr[low_fpr_idx[-1]] if len(low_fpr_idx) > 0 else 0.0
    except:
        arc, acc, low = 0.5, 0.5, 0.0
    
    return arc, acc, low


# ============================================
# 主程序
# ============================================

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    print(f"Loading model from {model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    model.requires_grad_(False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_multiple_responses(model, tokenizer, input_ids, num_generations=5, 
                                 temperature=0.7, top_p=0.9, max_new_tokens=64):
    """生成多个回复并提取隐藏状态
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        input_ids: 输入 token IDs [seq_len]
        num_generations: 生成回复数量
        temperature: 采样温度
        top_p: nucleus sampling 参数
        max_new_tokens: 最大生成 token 数
    
    Returns:
        generated_texts: list of generated text strings
        embeddings: numpy array [num_generations, hidden_dim]
        perplexities: list of perplexity scores
    """
    device = next(model.parameters()).device
    input_ids_2d = input_ids.unsqueeze(0).to(device)
    input_length = input_ids_2d.shape[1]
    
    generated_texts = []
    embeddings = []
    perplexities = []
    energies = []
    
    for _ in range(num_generations):
        with torch.no_grad():
            outputs = model.generate(
                input_ids_2d,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                output_hidden_states=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # 解码生成的文本
        gen_ids = outputs.sequences[0, input_length:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generated_texts.append(gen_text)
        
        # 提取隐藏状态 - 使用最后一个 token 的中间层
        # outputs.hidden_states 是 tuple of tuples: (step, layer, [batch, seq, hidden])
        if outputs.hidden_states:
            # 取每个生成步骤的隐藏状态，然后平均
            num_layers = len(outputs.hidden_states[0])
            selected_layer = num_layers // 2  # 使用中间层
            
            # 收集所有生成 token 的隐藏状态
            token_embeddings = []
            for step_hidden in outputs.hidden_states:
                # step_hidden[selected_layer] 是 [batch, 1, hidden_dim]
                emb = step_hidden[selected_layer][0, -1, :].float().cpu().numpy()
                token_embeddings.append(emb)
            
            # 平均所有 token 的隐藏状态
            avg_embedding = np.mean(token_embeddings, axis=0)
            embeddings.append(avg_embedding)
        
        # 计算 perplexity
        if outputs.scores:
            ppl = get_perplexity_score([s[0] for s in outputs.scores])
            energy = get_energy_score([s[0] for s in outputs.scores])
            perplexities.append(ppl)
            energies.append(energy)
    
    if embeddings:
        embeddings = np.array(embeddings)
    else:
        embeddings = np.zeros((num_generations, 1))
    
    return generated_texts, embeddings, perplexities, energies


def main():
    parser = argparse.ArgumentParser(description="EigenScore Reproduction on TriviaQA/Winobias")
    parser.add_argument("--model", type=str, required=True, choices=['qwen3', 'llama3'])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=5,
                        help="Number of generations per sample for EigenScore")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="../output/eigenscore_results")
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    
    model_path = MODEL_PATHS[args.model]
    friendly = FRIENDLY_NAMES[args.model]
    
    print("=" * 70)
    print(f"EigenScore (INSIDE) 幻觉检测评估")
    print(f"模型: {args.model.upper()} ({friendly})")
    print(f"数据集: {args.dataset}")
    print(f"每样本生成数: {args.num_generations}")
    print(f"温度: {args.temperature}, Top-p: {args.top_p}")
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
    
    # 平衡采样
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
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 评分存储
    scores_dict = {
        'eigenscore': [],           # 核心：多次生成的隐藏状态协方差特征值
        'lexical_similarity': [],   # 多次生成的文本相似度
        'avg_perplexity': [],       # 平均困惑度
        'std_perplexity': [],       # 困惑度标准差
        'avg_energy': [],           # 平均 energy
    }
    labels = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n开始评估 (每样本 {args.num_generations} 次生成)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            # 获取输入 - 只用问题部分，不用原始答案
            q = str(row.get("question") or row.get("raw_question", ""))
            
            # 重新 tokenize 问题作为输入
            input_ids = tokenize(q, tokenizer, model_path)[0]
            
            # 生成多个回复
            gen_texts, embeddings, ppls, energies = generate_multiple_responses(
                model, tokenizer, input_ids,
                num_generations=args.num_generations,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens
            )
            
            # 计算 EigenScore
            eigenscore = get_eigenscore_from_embeddings(embeddings)
            
            # 计算词汇相似度
            lex_sim = get_lexical_similarity(gen_texts)
            
            # 标签：1 = 幻觉（原始答案错误）, 0 = 正确
            label = 1 - int(row['automatic_correctness'])
            
            # 存储
            scores_dict['eigenscore'].append(eigenscore)
            scores_dict['lexical_similarity'].append(lex_sim)
            scores_dict['avg_perplexity'].append(np.mean(ppls) if ppls else 0)
            scores_dict['std_perplexity'].append(np.std(ppls) if len(ppls) > 1 else 0)
            scores_dict['avg_energy'].append(np.mean(energies) if energies else 0)
            labels.append(label)
            
            # 定期清理显存
            if idx % 20 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n⚠️ 样本 {idx} 处理失败: {e}")
            for key in scores_dict:
                scores_dict[key].append(np.nan)
            labels.append(np.nan)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, f"scores_{args.dataset}_{args.model}_{len(labels)}samp.pkl")
    with open(output_file, 'wb') as f:
        pkl.dump([scores_dict, labels], f)
    print(f"\n✅ 评分已保存到: {output_file}")
    
    # ============================================
    # 结果分析
    # ============================================
    print("\n" + "=" * 70)
    print("EigenScore (INSIDE) 评估结果")
    print("=" * 70)
    
    labels_arr = np.array(labels)
    valid_mask = ~np.isnan(labels_arr)
    labels_valid = labels_arr[valid_mask].astype(int)
    
    print(f"\n有效样本: {valid_mask.sum()} / {len(labels)}")
    print(f"幻觉样本: {labels_valid.sum()} ({100*labels_valid.mean():.1f}%)")
    print(f"正确样本: {(1-labels_valid).sum()} ({100*(1-labels_valid.mean()):.1f}%)")
    
    results = []
    
    print("\n📊 各方法评分:")
    print("-" * 60)
    
    for name, scores in scores_dict.items():
        scores_arr = np.array(scores)[valid_mask]
        
        # EigenScore 和 Lexical Similarity: 高值 = 高一致性 = 低幻觉概率
        # 需要反转来检测幻觉
        if name in ['lexical_similarity']:
            # 高相似度 = 低幻觉，需要取负
            scores_for_auc = -scores_arr
        else:
            scores_for_auc = scores_arr
        
        arc, acc, low = get_roc_scores(scores_for_auc, labels_valid)
        
        # 如果 AUC < 0.5，说明方向反了
        if arc < 0.5:
            arc = 1 - arc
            scores_for_auc = -scores_for_auc
        
        print(f"  {name:<25} AUC={arc:.4f}  Acc={acc:.4f}")
        results.append({'method': name, 'auc': arc, 'acc': acc, 'tpr_at_5fpr': low})
    
    # 保存汇总
    results_df = pd.DataFrame(results)
    results_file = os.path.join(args.output_dir, f"results_{args.dataset}_{args.model}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ 汇总结果已保存到: {results_file}")
    
    # 对比参考
    print("\n" + "=" * 70)
    print("📈 与其他方法的对比参考")
    print("=" * 70)
    
    print("""
┌────────────────────────────────────────────────────────────────┐
│                    方法对比 (AUC)                               │
├────────────────────────────────────────────────────────────────┤
│ 方法                    │ TriviaQA  │ Winobias (Q) │ Winobias (L)│
├────────────────────────────────────────────────────────────────┤
│ EigenScore (本次复现)   │   ?       │      ?       │      ?      │
│ LLM-Check Window Entropy│  0.51     │     0.51     │    0.41     │
│ LLM-Check Hidden SVD    │  0.43     │     0.46     │    0.61     │
├────────────────────────────────────────────────────────────────┤
│ Prefill Probe           │  0.65     │     0.77     │    0.63     │
│ Decode Probe (Lookahead)│  0.84     │     0.86     │    0.74     │
│ 端到端实时              │  0.81     │      -       │     -       │
└────────────────────────────────────────────────────────────────┘

注: EigenScore 通过分析多次采样的语义一致性来检测幻觉
    低一致性（高 EigenScore）= 高幻觉概率
""")
    
    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()

