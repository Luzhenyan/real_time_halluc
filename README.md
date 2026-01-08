# Real-Time Hallucination Detection

基于 Position Probe 的实时幻觉检测系统。

## 目录结构

```
├── src/
│   ├── train_token_probe.py      # 训练 Position Probe
│   ├── eval_pos_probe_span.py    # 评估 Probe 的 Span 检测性能
│   ├── visualize_pos_scores.py   # 可视化 token 级置信度
│   └── probing_utils.py          # 依赖工具函数
├── data/
│   ├── llama-3-8b-instruct-local-answers-triviaqa_test.csv
│   └── llama-3-8b-instruct-local-input_output_ids-triviaqa_test.pt
├── probe/
│   └── llama-3-8b-instruct-local_triviaqa_pure_mlp_last_spanpos_ansneg_mixedHard8_scaler_pca256.joblib
├── output/                       # 输出目录
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用流程

### 1. 训练 Position Probe（如需重新训练）

```bash
cd src
python train_token_probe.py \
  --model /path/to/llama3-8b-instruct \
  --dataset triviaqa \
  --probe_at mlp \
  --layer last \
  --neg_strategy mixed \
  --hard_window 8 \
  --use_scaler \
  --pca_dim 256 \
  --output_dir ../probe
```

### 2. 评估 Probe 的 Span 检测性能

```bash
cd src
python eval_pos_probe_span.py \
  --dataset triviaqa_test \
  --model /path/to/llama3-8b-instruct \
  --pos_probe_path ../probe/llama-3-8b-instruct-local_triviaqa_pure_mlp_last_spanpos_ansneg_mixedHard8_scaler_pca256.joblib \
  --max_samples 200 \
  --balanced_eval \
  --thr 0.8 \
  --pick first
```

### 3. 可视化每个样本的置信度曲线

```bash
cd src
python visualize_pos_scores.py \
  --dataset triviaqa_test \
  --model /path/to/llama3-8b-instruct \
  --pos_probe_path ../probe/llama-3-8b-instruct-local_triviaqa_pure_mlp_last_spanpos_ansneg_mixedHard8_scaler_pca256.joblib \
  --max_samples 200 \
  --balanced_eval \
  --output_dir ../output
```

## 输出

- **评估指标**: miss_rate, IoU, hit@end, first_trigger_hit_rate
- **可视化图片**: `output/sample_XXX.png` (每个样本的 token 级置信度曲线)
- **详细数据**: `output/all_samples.json`

## GPU 需求

- 约需 16GB GPU 内存加载 Llama-3-8B
- 支持 `device_map='auto'` 自动分配多卡

## 模型

需要自行下载 LLM 模型：
- HuggingFace: `meta-llama/Meta-Llama-3-8B-Instruct`
- 或指定本地路径
