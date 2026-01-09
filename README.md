# Real-Time Hallucination Detection

基于 Position Probe 和 Hallucination Probe 的实时幻觉检测系统。

## 目录结构

```
├── src/
│   ├── probing_utils.py              # 通用工具函数
│   ├── data_gen/                     # 数据生成
│   │   ├── generate_model_answers.py # 生成模型答案
│   │   ├── compute_correctness.py    # 计算正确性标签
│   │   └── extract_exact_answer.py   # 提取精确答案
│   ├── prefill/                      # Prefill 阶段
│   │   └── train_prefill_probes_all_layers.py  # 训练 prefill 幻觉 probe
│   ├── decode/                       # Decode 阶段
│   │   ├── train_token_probe.py      # 训练位置 probe (检测关键 token)
│   │   ├── train_hallu_probes_at_key_positions.py  # 训练 decode 幻觉 probe
│   │   └── eval_dynamic_pipeline.py  # 动态位置检测 + 幻觉检测评估
│   └── eval/                         # 评估/测试
│       ├── eval_pos_probe_span.py    # 评估位置 probe 的 span 检测
│       ├── eval_end_to_end_realtime.py  # 端到端实时评估
│       ├── eval_token_probe.py       # 评估位置 probe
│       └── visualize_pos_scores.py   # 可视化 token 级置信度
├── scripts/                          # Pipeline 脚本
│   ├── triviaqa_pipeline.sh          # TriviaQA 完整流程
│   └── run_all_layers_train_and_e2e_triviaqa.sh
├── data/                             # 数据文件
├── probe/                            # 已训练的 probe
├── output/                           # 输出目录
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 完整工作流程

### Phase 1: 数据生成

```bash
cd src/data_gen

# 1. 生成模型答案
python generate_model_answers.py \
  --model /path/to/llama3-8b-instruct \
  --dataset triviaqa

# 2. 计算正确性标签
python compute_correctness.py \
  --model llama-3-8b-instruct-local \
  --dataset triviaqa

# 3. 提取精确答案
python extract_exact_answer.py \
  --model llama-3-8b-instruct-local \
  --dataset triviaqa
```

### Phase 2: 训练 Prefill 阶段 Probe

```bash
cd src/prefill

# 训练 prefill 幻觉 probe (在 last_q_token 位置)
python train_prefill_probes_all_layers.py \
  --model /path/to/llama3-8b-instruct \
  --dataset triviaqa \
  --target last_q_token \
  --n_per_class 500
```

### Phase 3: 训练 Decode 阶段 Probe

```bash
cd src/decode

# 1. 训练位置 probe (检测关键 token 位置)
python train_token_probe.py \
  --model /path/to/llama3-8b-instruct \
  --dataset triviaqa \
  --probe_at mlp \
  --layer last \
  --neg_strategy mixed \
  --hard_window 8 \
  --use_scaler \
  --pca_dim 256

# 2. 训练 decode 幻觉 probe (在关键 token 位置)
python train_hallu_probes_at_key_positions.py \
  --model /path/to/llama3-8b-instruct \
  --dataset triviaqa \
  --target exact_answer_last_token
```

### Phase 4: 评估

```bash
cd src/eval

# 1. 评估位置 probe 的 span 检测性能
python eval_pos_probe_span.py \
  --dataset triviaqa_test \
  --model /path/to/llama3-8b-instruct \
  --pos_probe_path ../../probe/xxx.joblib \
  --max_samples 200 \
  --balanced_eval \
  --thr 0.8 \
  --pick first

# 2. 端到端实时评估
python eval_end_to_end_realtime.py \
  --dataset triviaqa_test \
  --pos_probe_path ../../probe/pos_probe.joblib \
  --hallu_probe_base ../../probe/hallu_probe \
  --max_samples 100 \
  --balanced_eval

# 3. 动态 Pipeline 评估 (支持多种位置检测策略)
cd ../decode
python eval_dynamic_pipeline.py \
  --dataset triviaqa_test \
  --model /path/to/llama3-8b-instruct \
  --pos_probe_path ../../probe/pos_probe.joblib \
  --pos_select_strategy threshold_last \
  --pos_threshold 0.8 \
  --max_samples 200

# 3. 可视化 token 级置信度
python visualize_pos_scores.py \
  --dataset triviaqa_test \
  --model /path/to/llama3-8b-instruct \
  --pos_probe_path ../../probe/xxx.joblib \
  --max_samples 20 \
  --output_dir ../../output
```

## 输出指标

### 位置 Probe 评估
- `miss_rate`: 未检测到 span 的比例
- `overlap_hit_rate`: 预测 span 与 GT 有交集的比例
- `IoU`: 预测与 GT 的交并比
- `hit@end`: 预测 span 结束位置准确率
- `first_trigger_hit_rate`: 第一个触发的 token 在 GT 内的比例

### 端到端评估
- `Prefill Accuracy`: Prefill 阶段准确率
- `Decode Accuracy`: Decode 阶段准确率
- `AUROC`: 幻觉检测的 ROC-AUC

## GPU 需求

- 约需 16GB GPU 内存加载 Llama-3-8B
- 支持 `device_map='auto'` 自动分配多卡

## 模型

需要自行下载 LLM 模型：
- HuggingFace: `meta-llama/Meta-Llama-3-8B-Instruct`
- 或指定本地路径
