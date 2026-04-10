#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B TriviaQA Pipeline
# 使用 Qwen3-8B 在 TriviaQA 上训练和评估 probe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# HuggingFace 镜像加速
export HF_ENDPOINT="https://hf-mirror.com"

ROOT="/home/luzhenyan/tmp/real_time_halluc-main"
cd "${ROOT}/src"

# 添加 src 目录到 PYTHONPATH，让子目录能找到 probing_utils
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# 模型路径
MODEL_PATH="${MODEL_PATH:-/var/wangyicheng/models/Qwen3-8B}"
DATASET_TRAIN="${DATASET_TRAIN:-triviaqa}"
DATASET_TEST="${DATASET_TEST:-triviaqa_test}"

# 样本数量 (可调整以加快测试)
MAX_SAMPLES_TRAIN="${MAX_SAMPLES_TRAIN:-500}"
MAX_SAMPLES_TEST="${MAX_SAMPLES_TEST:-100}"

# Position probe 参数
LAYER="${LAYER:-last}"
NEG_POS_RATIO="${NEG_POS_RATIO:-5.0}"
PCA_COMPONENTS="${PCA_COMPONENTS:-256}"

echo "=============================================="
echo "Qwen3-8B TriviaQA Pipeline"
echo "=============================================="
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET_TRAIN=${DATASET_TRAIN}"
echo "DATASET_TEST=${DATASET_TEST}"
echo "MAX_SAMPLES_TRAIN=${MAX_SAMPLES_TRAIN}"
echo "MAX_SAMPLES_TEST=${MAX_SAMPLES_TEST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Step 1: 生成模型答案
echo ""
echo "=== Step 1: Generate model answers ==="
cd "${ROOT}/src/data_gen"
python generate_model_answers.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_TRAIN}" \
    --n_samples "${MAX_SAMPLES_TRAIN}"

python generate_model_answers.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_TEST}" \
    --n_samples "${MAX_SAMPLES_TEST}"

# Step 2: 提取精确答案 (如果需要)
echo ""
echo "=== Step 2: Extract exact answers ==="
python extract_exact_answer.py \
    --model "${MODEL_PATH}" \
    --extraction_model "${MODEL_PATH}" \
    --dataset "${DATASET_TRAIN}"

python extract_exact_answer.py \
    --model "${MODEL_PATH}" \
    --extraction_model "${MODEL_PATH}" \
    --dataset "${DATASET_TEST}"

# Step 3: 训练位置 probe
echo ""
echo "=== Step 3: Train position probe ==="
cd "${ROOT}/src/decode"
POS_PROBE_OUT_DIR="${ROOT}/probe"
mkdir -p "${POS_PROBE_OUT_DIR}"

python train_token_probe.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_TRAIN}" \
    --probe_at mlp \
    --layer "${LAYER}" \
    --neg_strategy mixed \
    --hard_window 8 \
    --neg_pos_ratio "${NEG_POS_RATIO}" \
    --use_pca "${PCA_COMPONENTS}" \
    --use_scaler \
    --max_samples "${MAX_SAMPLES_TRAIN}" \
    --out_dir "${POS_PROBE_OUT_DIR}"

# Step 4: 评估位置 probe
echo ""
echo "=== Step 4: Evaluate position probe ==="
cd "${ROOT}/src/eval"
POS_PROBE_PATH="${POS_PROBE_OUT_DIR}/qwen3-8b_${DATASET_TRAIN}_pure_mlp_${LAYER}_spanpos_ansneg_mixedHard8_scaler_pca${PCA_COMPONENTS}.joblib"

# 如果找不到带完整后缀的文件，尝试简单命名
if [ ! -f "${POS_PROBE_PATH}" ]; then
    POS_PROBE_PATH="${POS_PROBE_OUT_DIR}/qwen3-8b_${DATASET_TRAIN}_pure.joblib"
fi

echo "Using pos_probe: ${POS_PROBE_PATH}"

python eval_pos_probe_span.py \
    --dataset "${DATASET_TEST}" \
    --model "${MODEL_PATH}" \
    --pos_probe_path "${POS_PROBE_PATH}" \
    --max_samples "${MAX_SAMPLES_TEST}" \
    --balanced_eval \
    --thr 0.8 \
    --pick first

echo ""
echo "=== ALL DONE ==="
echo "Probe saved to: ${POS_PROBE_OUT_DIR}"

