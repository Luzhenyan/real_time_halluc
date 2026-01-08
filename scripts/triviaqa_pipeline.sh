#!/usr/bin/env bash
set -euo pipefail

# TriviaQA end-to-end pipeline (similar to math_pipeline.sh):
# - generate answers (+ ids) for train/test
# - extract exact_answer (needed for exact_answer_* tokens and position probe)
# - train targeted position probe ("pure hidden state")
# - train hallucination probes at multiple positions
# - evaluate dynamic pipeline on test

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# Force HF to use the cache directory provided by the user
export HF_HOME="/mnt/pcllzy_2/.cache/huggingface"

ROOT="/mnt/pcllzy_2/LLMsKnow"
cd "${ROOT}/src"

MODEL_PATH="${MODEL_PATH:-/mnt/pcllzy/llama3-instruction-8b}"
DATASET_TRAIN="${DATASET_TRAIN:-triviaqa}"
DATASET_TEST="${DATASET_TEST:-triviaqa_test}"

# sample counts (0 => full csv); TriviaQA is large, using 2000 as default for consistency
MAX_SAMPLES_TRAIN="${MAX_SAMPLES_TRAIN:-2000}"
MAX_SAMPLES_TEST="${MAX_SAMPLES_TEST:-500}"

# position probe params
LAYER="${LAYER:-last}"
NEG_POS_RATIO_POS_PROBE="${NEG_POS_RATIO_POS_PROBE:-5.0}"
PCA_COMPONENTS="${PCA_COMPONENTS:-256}"

# dynamic pipeline position selection
POS_SELECT_STRATEGY="${POS_SELECT_STRATEGY:-argmax}"
POS_THRESHOLD="${POS_THRESHOLD:-0.5}"
POS_FAIL_ACTION="${POS_FAIL_ACTION:-skip}"

echo "Using MODEL_PATH=${MODEL_PATH}"
echo "Using DATASET_TRAIN=${DATASET_TRAIN}, DATASET_TEST=${DATASET_TEST}"
echo "Using MAX_SAMPLES_TRAIN=${MAX_SAMPLES_TRAIN}, MAX_SAMPLES_TEST=${MAX_SAMPLES_TEST}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "=== Step 0: Clean up old score files (optional) ==="
# Safer cleanup: only remove scores for the datasets we are about to regenerate.
rm -f ../output/*scores-"${DATASET_TRAIN}".pt ../output/*scores-"${DATASET_TEST}".pt || true

echo "=== Step 1: Generate model answers (triviaqa train/test) ==="
bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
  python generate_model_answers.py --model \"${MODEL_PATH}\" --dataset \"${DATASET_TRAIN}\" --n_samples \"${MAX_SAMPLES_TRAIN}\" && \
  python generate_model_answers.py --model \"${MODEL_PATH}\" --dataset \"${DATASET_TEST}\" --n_samples \"${MAX_SAMPLES_TEST}\""

echo "=== Step 2: Extract exact answers (triviaqa train/test) ==="
bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
  python extract_exact_answer.py --model \"${MODEL_PATH}\" --extraction_model \"${MODEL_PATH}\" --dataset \"${DATASET_TRAIN}\" && \
  python extract_exact_answer.py --model \"${MODEL_PATH}\" --extraction_model \"${MODEL_PATH}\" --dataset \"${DATASET_TEST}\""

echo "=== Step 3: Train targeted position probe (pure hidden state) ==="
POS_PROBE_OUT_DIR="../output/token_probes_pure"
mkdir -p "${POS_PROBE_OUT_DIR}"
bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
  python train_token_probe.py \
    --model \"${MODEL_PATH}\" \
    --dataset \"${DATASET_TRAIN}\" \
    --layer \"${LAYER}\" \
    --neg_pos_ratio \"${NEG_POS_RATIO_POS_PROBE}\" \
    --use_pca \"${PCA_COMPONENTS}\" \
    --max_samples \"${MAX_SAMPLES_TRAIN}\" \
    --out_dir \"${POS_PROBE_OUT_DIR}\""

echo "=== Step 4: Train hallucination probes (mlp) at multiple positions ==="
mkdir -p ../checkpoints
bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
  python probe.py --model \"${MODEL_PATH}\" --probe_at mlp --seeds 0 5 26 42 63 --n_samples all --save_clf --dataset \"${DATASET_TRAIN}\" --layer 15 --token last_q_token && \
  python probe.py --model \"${MODEL_PATH}\" --probe_at mlp --seeds 0 5 26 42 63 --n_samples all --save_clf --dataset \"${DATASET_TRAIN}\" --layer 15 --token exact_answer_last_token && \
  python probe.py --model \"${MODEL_PATH}\" --probe_at mlp --seeds 0 5 26 42 63 --n_samples all --save_clf --dataset \"${DATASET_TRAIN}\" --layer 15 --token full_answer_last_token"

echo "=== Step 4b: Train hallucination probes for ALL layers (mlp + scaler, strict id alignment) ==="
# Keep overall pipeline behavior aligned with the legacy version:
# - Step 4 remains unchanged (layer-15 probes without scaler) to support eval_dynamic_pipeline.py defaults.
# - Step 4b additionally trains per-layer probes (seed=0) for end-to-end realtime eval / multi-layer usage.
ALL_LAYERS_SEED="${ALL_LAYERS_SEED:-0}"
ALL_LAYERS_N_PER_CLASS="${ALL_LAYERS_N_PER_CLASS:-500}"
ALL_LAYERS_VAL="${ALL_LAYERS_VAL:-200}"
ALL_LAYERS_START="${ALL_LAYERS_START:-0}"
ALL_LAYERS_END="${ALL_LAYERS_END:--1}"

bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
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

echo "=== Step 5: Evaluate dynamic pipeline on test ==="
POS_PROBE_PATH="${POS_PROBE_OUT_DIR}/llama-3-8b-instruct-local_${DATASET_TRAIN}_pure.joblib"
bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamcatcher && \
  python eval_dynamic_pipeline.py \
    --model \"${MODEL_PATH}\" \
    --dataset \"${DATASET_TEST}\" \
    --pos_probe_path \"${POS_PROBE_PATH}\" \
    --max_samples \"${MAX_SAMPLES_TEST}\" \
    --pos_select_strategy \"${POS_SELECT_STRATEGY}\" \
    --pos_threshold \"${POS_THRESHOLD}\" \
    --pos_fail_action \"${POS_FAIL_ACTION}\""

echo "=== ALL DONE ==="

