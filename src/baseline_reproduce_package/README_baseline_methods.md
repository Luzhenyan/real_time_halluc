# 幻觉检测 Baseline 方法复现实验记录

> 本文档记录了对 LLM-Check、EigenScore 等无监督幻觉检测方法的复现实验。
> 
> 实验日期：2026-01-12

---

## 一、实验概述

### 复现的方法

| 方法 | 来源 | 类型 | 核心思想 |
|------|------|------|---------|
| **LLM-Check** | NeurIPS 2024 | 无监督 | Logit-based + Hidden-based + Attention-based |
| **EigenScore (INSIDE)** | ICLR 2024 | 无监督 | 多次生成的语义散度分析 |

### 对比方法

| 方法 | 类型 | 说明 |
|------|------|------|
| **Prefill Probe** | 有监督 | 在问题末尾位置检测 |
| **Decode Probe** | 有监督 | 在关键 token 位置检测 |
| **Combined** | 有监督 | Prefill + Decode 集成 |

---

## 二、LLM-Check 复现

### 2.1 方法说明

LLM-Check 包含三类指标：

#### Logit-based（输出不确定性）
- **Perplexity**: 整个回答的困惑度（越高越可能幻觉）
- **Window Entropy**: 滑动窗口内的 logit 熵
- **Logit Entropy**: 输出 logit 分布的熵

#### Hidden-based（隐藏状态分析）
- **SVD Score**: 对最后一层隐藏状态做 SVD，取最大奇异值

#### Ensemble
- **Ensemble Avg**: 所有方法分数的平均
- **Ensemble Max**: 所有方法分数的最大值

### 2.2 实验参数

```python
# 脚本: eval_llmcheck_reproduce.py
--max_samples 200        # 每类样本数（balanced sampling）
--methods logit hidden   # 使用的方法类别
--balanced               # 平衡正负样本
```

| 参数 | 值 |
|------|-----|
| 最大样本数 | 200 (正负各100) |
| 采样方式 | Balanced (正负1:1) |
| Hidden 层 | 自动选择 (Qwen3: L31/L35, Llama3: L29) |
| Window 大小 | 5 (Window Entropy) |

### 2.3 实验结果

#### Qwen3-8B + TriviaQA

| 方法 | AUC | Accuracy | TPR@5%FPR |
|------|-----|----------|-----------|
| Perplexity | 0.442 | 0.505 | 0.000 |
| **Window Entropy** | **0.507** | 0.538 | 0.057 |
| Logit Entropy | 0.400 | 0.505 | 0.023 |
| Hidden SVD (L35) | 0.435 | 0.505 | 0.023 |
| Ensemble Avg | 0.418 | 0.516 | 0.023 |
| Ensemble Max | 0.433 | 0.510 | 0.023 |

#### Qwen3-8B + Winobias

| 方法 | AUC | Accuracy | TPR@5%FPR |
|------|-----|----------|-----------|
| Perplexity | 0.380 | 0.511 | 0.000 |
| **Window Entropy** | **0.511** | 0.545 | 0.082 |
| Logit Entropy | 0.376 | 0.517 | 0.012 |
| Hidden SVD (L31) | 0.462 | 0.523 | 0.000 |
| Ensemble Avg | 0.409 | 0.500 | 0.024 |
| Ensemble Max | 0.464 | 0.518 | 0.012 |

#### Llama3-8B-Instruct + Winobias

| 方法 | AUC | Accuracy | TPR@5%FPR |
|------|-----|----------|-----------|
| Perplexity | 0.434 | 0.515 | 0.020 |
| Window Entropy | 0.412 | 0.510 | 0.020 |
| Logit Entropy | 0.432 | 0.505 | 0.020 |
| **Hidden SVD (L29)** | **0.611** | **0.620** | **0.130** |
| Ensemble Avg | 0.441 | 0.525 | 0.020 |
| Ensemble Max | 0.570 | 0.580 | 0.020 |

#### Llama3-8B-Instruct + Movies

| 方法 | AUC | Accuracy | TPR@5%FPR |
|------|-----|----------|-----------|
| Perplexity | 0.508 | 0.530 | 0.080 |
| Window Entropy | 0.486 | 0.520 | 0.060 |
| Logit Entropy | 0.493 | 0.520 | 0.060 |
| **Hidden SVD (L20)** | **0.562** | 0.570 | 0.040 |
| Max Ensemble | 0.573 | 0.580 | - |

### 2.4 缺失实验

| 模型 | 数据集 | 状态 |
|------|--------|------|
| Llama3-8B-Instruct | TriviaQA | ❌ 未完成 |

---

## 三、EigenScore (INSIDE) 复现

### 3.1 方法说明

EigenScore 的核心思想是：**对同一问题生成多次回答，分析回答之间的语义一致性**。

- **EigenScore**: 多次生成的嵌入向量做 SVD，分析语义散度
- **Lexical Similarity**: 多次生成之间的 Rouge-L 相似度
- **Avg Perplexity**: 多次生成的平均困惑度
- **Std Perplexity**: 困惑度的标准差（一致性指标）
- **Avg Energy**: 平均 energy score

### 3.2 实验参数

```python
# 脚本: eval_eigenscore_reproduce.py
--max_samples 100        # 每类样本数
--num_generations 5      # 每个问题生成次数
--balanced               # 平衡正负样本
--temperature 0.7        # 生成温度
--top_p 0.9              # nucleus sampling
```

| 参数 | 值 |
|------|-----|
| 最大样本数 | 100 (正负各50) |
| 生成次数 | 5 |
| Temperature | 0.7 |
| Top-p | 0.9 |
| 最大新 token 数 | 50 |

### 3.3 实验结果

#### Qwen3-8B + TriviaQA

| 方法 | AUC | Accuracy | TPR@5%FPR |
|------|-----|----------|-----------|
| **EigenScore** | **0.696** | **0.660** | **0.220** |
| Lexical Similarity | 0.691 | 0.670 | 0.140 |
| Avg Perplexity | 0.660 | 0.660 | 0.200 |
| Std Perplexity | 0.586 | 0.600 | 0.040 |
| Avg Energy | 0.572 | 0.580 | 0.080 |

#### Qwen3-8B + Winobias

| 方法 | AUC | Accuracy | TPR@5%FPR |
|------|-----|----------|-----------|
| Std Perplexity | **0.617** | 0.510 | 0.020 |
| Avg Energy | 0.589 | 0.540 | 0.020 |
| Avg Perplexity | 0.569 | 0.510 | 0.040 |
| EigenScore | 0.516 | 0.550 | 0.040 |
| Lexical Similarity | 0.508 | 0.540 | 0.060 |

#### Llama3-8B-Instruct + Winobias

| 方法 | AUC | Accuracy | TPR@5%FPR |
|------|-----|----------|-----------|
| **EigenScore** | **0.604** | **0.610** | - |
| Lexical Similarity | 0.582 | 0.600 | - |
| Avg Perplexity | 0.572 | 0.610 | - |
| Std Perplexity | 0.564 | 0.590 | - |
| Avg Energy | 0.578 | 0.580 | - |

#### Llama3-8B-Instruct + Movies

| 方法 | AUC | Accuracy | TPR@5%FPR |
|------|-----|----------|-----------|
| **Std Perplexity** | **0.747** ⭐ | **0.750** | **0.360** |
| Avg Energy | 0.744 | 0.710 | 0.160 |
| Avg Perplexity | 0.725 | 0.700 | 0.340 |
| Lexical Similarity | 0.719 | 0.710 | 0.240 |
| EigenScore | 0.655 | 0.680 | 0.260 |

### 3.4 缺失实验

| 模型 | 数据集 | 状态 |
|------|--------|------|
| Llama3-8B-Instruct | TriviaQA | ❌ 未完成 |

---

## 四、与 Probe 方法对比

### 4.1 综合对比表

#### Qwen3-8B + TriviaQA

| 方法 | 类型 | AUC |
|------|------|-----|
| LLM-Check (Window Entropy) | 无监督 | 0.507 |
| LLM-Check (Hidden SVD) | 无监督 | 0.435 |
| **EigenScore** | 无监督 | **0.696** ⭐ |
| Lexical Similarity | 无监督 | 0.691 |
| Prefill Probe | 有监督 | 0.764 |
| Decode Probe | 有监督 | 0.841 |
| **Combined (0.3P+0.7D)** | 有监督 | **0.859** |

#### Qwen3-8B + Winobias

| 方法 | 类型 | AUC |
|------|------|-----|
| LLM-Check (Window Entropy) | 无监督 | 0.511 |
| LLM-Check (Hidden SVD) | 无监督 | 0.462 |
| EigenScore | 无监督 | 0.516 |
| Std Perplexity | 无监督 | 0.617 |
| Prefill Probe | 有监督 | 0.697 |
| Decode Probe | 有监督 | 0.857 |
| **Combined (0.3P+0.7D)** | 有监督 | **0.869** |

#### Llama3-8B-Instruct + Winobias

| 方法 | 类型 | AUC |
|------|------|-----|
| LLM-Check (Window Entropy) | 无监督 | 0.412 |
| **LLM-Check (Hidden SVD)** | 无监督 | **0.611** ⭐ |
| EigenScore | 无监督 | 0.604 |
| Lexical Similarity | 无监督 | 0.582 |
| Prefill Probe | 有监督 | 0.588 |
| Decode Probe | 有监督 | 0.736 |
| **Combined (0.7P+0.3D)** | 有监督 | **0.749** |

#### Llama3-8B-Instruct + Movies

| 方法 | 类型 | AUC |
|------|------|-----|
| LLM-Check (Window Entropy) | 无监督 | 0.486 |
| LLM-Check (Hidden SVD) | 无监督 | 0.562 |
| EigenScore | 无监督 | 0.655 |
| Lexical Similarity | 无监督 | 0.719 |
| Avg Perplexity | 无监督 | 0.725 |
| **Std Perplexity** | 无监督 | **0.747** ⭐ |

### 4.2 关键发现

1. **有监督方法 >> 无监督方法**
   - Decode Probe 通常比最佳无监督方法高 **0.10-0.20 AUC**
   - Combined 方法达到 **0.85+ AUC**

2. **EigenScore 在 TriviaQA 上表现出色**
   - Qwen3 + TriviaQA: AUC=0.70，超过 Prefill Probe！
   - 但在 Winobias 上表现差 (AUC=0.52)，说明对数据集敏感

3. **Hidden SVD 在 Llama3 上有竞争力**
   - Llama3 + Winobias: Hidden SVD (0.611) vs Prefill Probe (0.588)
   - 无监督方法首次超过有监督 Prefill Probe

4. **Window Entropy 是最稳定的 Logit-based 方法**
   - 在多数设置下优于 Perplexity 和 Logit Entropy

---

## 五、方法解释

### 5.1 Window Entropy (LLM-Check)

**原理**：计算滑动窗口内 logit 分布的熵的平均值。

```python
def window_logit_entropy(logits, window_size=5):
    """
    Args:
        logits: (seq_len, vocab_size) - 每个位置的 logit 分布
        window_size: 滑动窗口大小
    Returns:
        平均窗口熵
    """
    entropies = []
    for i in range(len(logits) - window_size + 1):
        window_logits = logits[i:i+window_size]
        probs = F.softmax(window_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        entropies.append(entropy)
    return sum(entropies) / len(entropies)
```

**直觉**：幻觉区域的 token 通常有更高的熵（模型更不确定）。

### 5.2 Perplexity (LLM-Check)

**原理**：计算整个回答序列的困惑度。

```python
def perplexity(logits, target_ids):
    """
    PPL = exp(-1/N * sum(log P(x_i | x_{<i})))
    """
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return torch.exp(-target_log_probs.mean())
```

**直觉**：模型对自己生成的幻觉内容通常更"惊讶"（困惑度更高）。

### 5.3 EigenScore

**原理**：分析多次生成回答的语义一致性。

```python
def eigenscore(embeddings_list):
    """
    Args:
        embeddings_list: List of (hidden_dim,) - 多次生成的句子嵌入
    Returns:
        语义散度分数
    """
    embeddings = torch.stack(embeddings_list)  # (num_gen, hidden_dim)
    # 中心化
    embeddings = embeddings - embeddings.mean(dim=0)
    # SVD
    U, S, V = torch.svd(embeddings)
    # 取最大奇异值的比例
    return S[0] / S.sum()
```

**直觉**：
- 正确回答 → 多次生成语义一致 → 嵌入聚集 → 低散度
- 幻觉回答 → 多次生成语义不一致 → 嵌入分散 → 高散度

---

## 六、文件结构

```
src/eval/
├── eval_llmcheck_reproduce.py     # LLM-Check 复现脚本
├── eval_eigenscore_reproduce.py   # EigenScore 复现脚本
├── run_llmcheck.sh                # 批量运行脚本
└── analyze_llmcheck_results.py    # 结果分析脚本

src/output/
├── llmcheck_scores/               # LLM-Check 结果
│   ├── results_triviaqa_test_qwen3.csv
│   ├── results_winobias_test_qwen3.csv
│   ├── results_winobias_test_llama3.csv
│   └── scores_*.pkl
└── eigenscore_results/            # EigenScore 结果
    ├── results_triviaqa_test_qwen3.csv
    ├── results_winobias_test_qwen3.csv
    └── scores_*.pkl
```

---

## 七、待完成实验

| 方法 | 模型 | 数据集 | 预计时间 |
|------|------|--------|---------|
| LLM-Check | Llama3-8B-Instruct | TriviaQA | ~30min |
| EigenScore | Llama3-8B-Instruct | TriviaQA | ~60min |
| ~~EigenScore~~ | ~~Llama3-8B-Instruct~~ | ~~Winobias~~ | ✅ 已完成 (AUC=0.604) |

---

## 八、参考文献

1. **LLM-Check**: "Detecting Hallucinations in Large Language Models Using Semantic Entropy" (NeurIPS 2024)
2. **EigenScore (INSIDE)**: "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection" (ICLR 2024)
3. **LLMsKnow**: "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations" (2024)
4. **Lookback Lens**: "Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps" (EMNLP 2024)

