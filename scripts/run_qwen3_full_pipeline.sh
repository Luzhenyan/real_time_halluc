#!/bin/bash
# Qwen3-8B 完整 Pipeline - 2000 均匀样本
# 从数据生成到端到端评估

set -eo pipefail

# ========== 配置 ==========
MODEL_PATH="/var/wangyicheng/models/Qwen3-8B"
DATASET_TRAIN="triviaqa"
DATASET_TEST="triviaqa_test"
PROJECT_ROOT="/home/luzhenyan/tmp/real_time_halluc-main"

# 样本数量配置
N_TRAIN_SAMPLES=3000      # 训练集总样本数 (生成更多以确保每类有足够样本)
N_TEST_SAMPLES=1000       # 测试集样本数
N_PER_CLASS=1000          # 每类样本数 (1000 正确 + 1000 错误 = 2000 均匀样本)

# Probe 层配置
POSITION_PROBE_LAYER=15           # Position Probe: 第 15 层
PREFILL_LAYER_START=0             # Prefill Probe: 从第 0 层开始 (所有层)
DECODE_HALLU_LAYER_START=15       # Decode Hallu Probe: 从第 15 层开始

# 输出目录 (大文件存到 /var/luzhenyan)
VAR_ROOT="/var/luzhenyan/real_time_halluc"
PROBE_DIR="${VAR_ROOT}/probe"
CHECKPOINT_DIR="${VAR_ROOT}/checkpoints"
OUTPUT_DIR="${VAR_ROOT}/output"

# 环境设置
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export WANDB_MODE=offline

# 创建目录
mkdir -p "${PROBE_DIR}" "${CHECKPOINT_DIR}" "${OUTPUT_DIR}"
mkdir -p "${PROJECT_ROOT}/src/output"

echo "=============================================="
echo "Qwen3-8B Full Pipeline (2000 均匀样本)"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Train samples: ${N_TRAIN_SAMPLES} (${N_PER_CLASS} per class)"
echo "Test samples: ${N_TEST_SAMPLES}"
echo ""
echo "Probe 配置:"
echo "  - Prefill Probe: 所有层 (0 ~ last)"
echo "  - Position Probe: 第 ${POSITION_PROBE_LAYER} 层"
echo "  - Decode Hallu Probe: ${DECODE_HALLU_LAYER_START} ~ last 层"
echo ""

# ========== Phase 1: 数据生成 ==========
echo "=============================================="
echo "Phase 1: 数据生成"
echo "=============================================="

# Step 1.1: 生成训练集模型答案
echo ""
echo "=== Step 1.1: Generate model answers (train) ==="
cd "${PROJECT_ROOT}/src/data_gen"

python generate_model_answers.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_TRAIN}" \
    --n_samples ${N_TRAIN_SAMPLES}

echo "✅ 训练集答案生成完成"

# Step 1.2: 生成测试集模型答案
echo ""
echo "=== Step 1.2: Generate model answers (test) ==="
python generate_model_answers.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_TEST}" \
    --n_samples ${N_TEST_SAMPLES}

echo "✅ 测试集答案生成完成"

# Step 1.3: 提取精确答案 (训练集)
echo ""
echo "=== Step 1.3: Extract exact answers (train) ==="
python extract_exact_answer.py \
    --model "${MODEL_PATH}" \
    --extraction_model "${MODEL_PATH}" \
    --dataset "${DATASET_TRAIN}"

echo "✅ 训练集精确答案提取完成"

# Step 1.4: 提取精确答案 (测试集)
echo ""
echo "=== Step 1.4: Extract exact answers (test) ==="
python extract_exact_answer.py \
    --model "${MODEL_PATH}" \
    --extraction_model "${MODEL_PATH}" \
    --dataset "${DATASET_TEST}"

echo "✅ 测试集精确答案提取完成"

# ========== Phase 2: 训练 Position Probe (第15层) ==========
echo ""
echo "=============================================="
echo "Phase 2: 训练 Position Probe (Layer ${POSITION_PROBE_LAYER})"
echo "=============================================="
cd "${PROJECT_ROOT}/src/decode"

python train_token_probe.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_TRAIN}" \
    --probe_at mlp \
    --layer ${POSITION_PROBE_LAYER} \
    --neg_strategy mixed \
    --hard_window 8 \
    --neg_pos_ratio 5.0 \
    --use_pca 256 \
    --use_scaler \
    --max_samples ${N_TRAIN_SAMPLES} \
    --out_dir "${PROBE_DIR}"

echo "✅ Position Probe 训练完成"

# Position Probe 单独评估
echo ""
echo "=== Eval: Position Probe ==="
cd "${PROJECT_ROOT}/src/eval"
POS_PROBE_PATH="${PROBE_DIR}/qwen3-8b_triviaqa_pure.joblib"

python eval_pos_probe_span.py \
    --dataset "${DATASET_TEST}" \
    --model "${MODEL_PATH}" \
    --pos_probe_path "${POS_PROBE_PATH}" \
    --max_samples 500 \
    --balanced_eval \
    --thr 0.8 \
    --pick first

echo "✅ Position Probe 评估完成"

# ========== Phase 3: 训练 Prefill Hallucination Probe (所有层) ==========
echo ""
echo "=============================================="
echo "Phase 3: 训练 Prefill Hallucination Probe (所有层)"
echo "=============================================="
cd "${PROJECT_ROOT}/src/prefill"

python train_prefill_probes_all_layers.py \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_TRAIN}" \
    --probe_at mlp \
    --token last_q_token \
    --n_per_class ${N_PER_CLASS} \
    --n_validation_samples 200 \
    --layer_start ${PREFILL_LAYER_START} \
    --layer_end -1 \
    --out_dir "${OUTPUT_DIR}/prefill_all_layers" \
    --save_clf_dir "${CHECKPOINT_DIR}"

echo "✅ Prefill Probe 训练完成"

# Prefill Probe 单独评估 (使用 prefill_only 模式)
echo ""
echo "=== Eval: Prefill Probe (prefill_only) ==="
cd "${PROJECT_ROOT}/src/eval"
HALLU_PROBE_BASE="${CHECKPOINT_DIR}/clf_qwen3-8b_${DATASET_TRAIN}"

python eval_end_to_end_realtime.py \
    --dataset "${DATASET_TEST}" \
    --model "${MODEL_PATH}" \
    --pos_probe_path "${POS_PROBE_PATH}" \
    --hallu_probe_base "${HALLU_PROBE_BASE}" \
    --max_samples 500 \
    --balanced_eval \
    --prefill_only \
    --stop_threshold 0.6

echo "✅ Prefill Probe 评估完成"

# ========== Phase 4: 训练 Decode Hallucination Probe (15层~最后一层) ==========
echo ""
echo "=============================================="
echo "Phase 4: 训练 Decode Hallucination Probe (Layer ${DECODE_HALLU_LAYER_START} ~ last)"
echo "=============================================="
cd "${PROJECT_ROOT}/src/decode"

# 训练每一层的 decode hallu probe
for layer in $(seq ${DECODE_HALLU_LAYER_START} 35); do
    echo ""
    echo "--- Training Decode Hallu Probe @ Layer ${layer} ---"
    
    python train_hallu_probes_at_key_positions.py \
        --model "${MODEL_PATH}" \
        --dataset "${DATASET_TRAIN}" \
        --test_dataset "${DATASET_TEST}" \
        --layer ${layer} \
        --probe_at mlp \
        --max_samples ${N_TRAIN_SAMPLES} \
        --max_samples_test ${N_TEST_SAMPLES} \
        --positions "exact_answer_last_token,exact_answer_first_token,full_answer_last_token" \
        --out_dir "${OUTPUT_DIR}/hallu_keypos" \
        --save_dir "${CHECKPOINT_DIR}/hallu_keypos"
done

echo "✅ Decode Hallu Probe 训练完成 (Layer ${DECODE_HALLU_LAYER_START} ~ 35)"

# Decode Probe 单独评估 (使用 decode_only 模式)
echo ""
echo "=== Eval: Decode Hallu Probe (decode_only) ==="
cd "${PROJECT_ROOT}/src/eval"

python eval_end_to_end_realtime.py \
    --dataset "${DATASET_TEST}" \
    --model "${MODEL_PATH}" \
    --pos_probe_path "${POS_PROBE_PATH}" \
    --pos_probe_at mlp \
    --hallu_probe_base "${HALLU_PROBE_BASE}" \
    --max_samples 500 \
    --balanced_eval \
    --decode_only \
    --teacher_force_decode \
    --diagnose_decode \
    --decode_layers $(seq -s ' ' ${DECODE_HALLU_LAYER_START} 35) \
    --stop_threshold 0.6 \
    --pos_threshold 0.8

echo "✅ Decode Hallu Probe 评估完成"

# ========== Phase 5: 端到端评估 ==========
echo ""
echo "=============================================="
echo "Phase 5: 端到端评估 (E2E)"
echo "=============================================="
cd "${PROJECT_ROOT}/src/eval"

python eval_end_to_end_realtime.py \
    --dataset "${DATASET_TEST}" \
    --model "${MODEL_PATH}" \
    --pos_probe_path "${POS_PROBE_PATH}" \
    --pos_probe_at mlp \
    --hallu_probe_base "${HALLU_PROBE_BASE}" \
    --max_samples 500 \
    --balanced_eval \
    --teacher_force_decode \
    --diagnose_decode \
    --pos_span_eval \
    --pos_score_dist \
    --decode_layers $(seq -s ' ' ${DECODE_HALLU_LAYER_START} 35) \
    --prefill_layers $(seq -s ' ' 20 35) \
    --stop_threshold 0.6 \
    --pos_threshold 0.8

echo ""
echo "=============================================="
echo "🎉 完整 Pipeline 执行完成!"
echo "=============================================="
echo ""
echo "Probe 配置总结:"
echo "  - Position Probe: Layer ${POSITION_PROBE_LAYER}"
echo "  - Prefill Hallu Probe: Layer 0 ~ 35 (所有层)"
echo "  - Decode Hallu Probe: Layer ${DECODE_HALLU_LAYER_START} ~ 35"
echo ""
echo "输出文件位置:"
echo "  - Position Probe: ${POS_PROBE_PATH}"
echo "  - Prefill Probes: ${CHECKPOINT_DIR}/clf_qwen3-8b_*_prefill_*.pkl"
echo "  - Decode Probes:  ${CHECKPOINT_DIR}/hallu_keypos/"
echo "  - 训练结果:       ${OUTPUT_DIR}/"
echo ""
