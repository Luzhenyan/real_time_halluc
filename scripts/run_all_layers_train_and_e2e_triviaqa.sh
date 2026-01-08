#!/usr/bin/env bash
set -euo pipefail

# Train + test (E2E) for triviaqa with our "all layers" probes.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME="/mnt/pcllzy_2/.cache/huggingface"

ROOT="/mnt/pcllzy_2/LLMsKnow"
DREAM="/mnt/pcllzy_2/dreamcatcher"
cd "${ROOT}/src"

MODEL_PATH="${MODEL_PATH:-/mnt/pcllzy/llama3-instruction-8b}"
FRIENDLY="${FRIENDLY:-llama-3-8b-instruct-local}"
DATASET_TRAIN="${DATASET_TRAIN:-triviaqa}"
DATASET_TEST="${DATASET_TEST:-triviaqa_test}"

# All-layer probe params
ALL_LAYERS_SEED="${ALL_LAYERS_SEED:-0}"
ALL_LAYERS_N_PER_CLASS="${ALL_LAYERS_N_PER_CLASS:-500}"
ALL_LAYERS_VAL="${ALL_LAYERS_VAL:-200}"
ALL_LAYERS_START="${ALL_LAYERS_START:-0}"
ALL_LAYERS_END="${ALL_LAYERS_END:--1}"

# Position probe params
POS_LAYER="${POS_LAYER:-last}"
NEG_POS_RATIO_POS_PROBE="${NEG_POS_RATIO_POS_PROBE:-5.0}"
PCA_COMPONENTS="${PCA_COMPONENTS:-256}"
MAX_SAMPLES_POS_PROBE="${MAX_SAMPLES_POS_PROBE:-2000}"

# E2E eval params
E2E_MAX_SAMPLES="${E2E_MAX_SAMPLES:-500}"
STOP_THRESHOLD="${STOP_THRESHOLD:-0.8}"
POS_THRESHOLD="${POS_THRESHOLD:-0.5}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using MODEL_PATH=${MODEL_PATH}"
echo "Using DATASET_TRAIN=${DATASET_TRAIN}, DATASET_TEST=${DATASET_TEST}"

echo "=== Step 1: Extract exact answers (train/test) ==="
bash -lc "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
  python extract_exact_answer.py --model \"${MODEL_PATH}\" --extraction_model \"${MODEL_PATH}\" --dataset \"${DATASET_TRAIN}\" && \
  python extract_exact_answer.py --model \"${MODEL_PATH}\" --extraction_model \"${MODEL_PATH}\" --dataset \"${DATASET_TEST}\""

echo "=== Step 2: Train position probe (pure) ==="
mkdir -p ../output/token_probes_pure
bash -lc "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
  python train_token_probe.py \
    --model \"${MODEL_PATH}\" \
    --dataset \"${DATASET_TRAIN}\" \
    --layer \"${POS_LAYER}\" \
    --neg_pos_ratio \"${NEG_POS_RATIO_POS_PROBE}\" \
    --use_pca \"${PCA_COMPONENTS}\" \
    --max_samples \"${MAX_SAMPLES_POS_PROBE}\" \
    --out_dir ../output/token_probes_pure"

echo "=== Step 3: Train ALL-LAYER hallu probes (prefill last_q_token + decode exact_answer_last_token) ==="
mkdir -p ../checkpoints ../output/prefill_all_layers
bash -lc "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
  python train_prefill_probes_all_layers.py \
    --model \"${MODEL_PATH}\" \
    --dataset \"${DATASET_TRAIN}\" \
    --probe_at mlp \
    --token last_q_token \
    --n_samples all \
    --n_per_class \"${ALL_LAYERS_N_PER_CLASS}\" \
    --n_validation_samples \"${ALL_LAYERS_VAL}\" \
    --layer_start \"${ALL_LAYERS_START}\" \
    --layer_end \"${ALL_LAYERS_END}\" \
    --seeds \"${ALL_LAYERS_SEED}\" && \
  python train_prefill_probes_all_layers.py \
    --model \"${MODEL_PATH}\" \
    --dataset \"${DATASET_TRAIN}\" \
    --probe_at mlp \
    --token exact_answer_last_token \
    --n_samples all \
    --n_per_class \"${ALL_LAYERS_N_PER_CLASS}\" \
    --n_validation_samples \"${ALL_LAYERS_VAL}\" \
    --layer_start \"${ALL_LAYERS_START}\" \
    --layer_end \"${ALL_LAYERS_END}\" \
    --seeds \"${ALL_LAYERS_SEED}\""

echo "=== Step 4: Dreamcatcher E2E realtime eval on test ==="
mkdir -p "${DREAM}/output"
POS_PROBE_PATH="${ROOT}/output/token_probes_pure/${FRIENDLY}_${DATASET_TRAIN}_pure.joblib"
HALLU_BASE="${ROOT}/checkpoints/clf_${FRIENDLY}_${DATASET_TRAIN}"
bash -lc "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
  cd \"${DREAM}\" && \
  python -u eval_end_to_end_realtime.py \
    --dataset \"${DATASET_TEST}\" \
    --pos_probe_path \"${POS_PROBE_PATH}\" \
    --hallu_probe_base \"${HALLU_BASE}\" \
    --max_samples \"${E2E_MAX_SAMPLES}\" \
    --stop_threshold \"${STOP_THRESHOLD}\" \
    --pos_threshold \"${POS_THRESHOLD}\" \
  > \"${DREAM}/output/eval_e2e_realtime_${DATASET_TEST}_max${E2E_MAX_SAMPLES}_thr${STOP_THRESHOLD}.log\" 2>&1"

echo "=== DONE (triviaqa) ==="


