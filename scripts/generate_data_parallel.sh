#!/bin/bash
#
# 并行数据生成脚本：8 GPU × 4 进程/GPU = 32 并行任务
#
# 生成任务：
# 1. Llama3 + TriviaQA (train + test)
# 2. Qwen3 + Movies (all: train + test)
#
# 用法：
#   bash scripts/generate_data_parallel.sh
#
# 环境变量（可覆盖）：
#   NUM_GPUS=8            GPU 数量
#   PROCS_PER_GPU=4       每 GPU 并行进程数
#   OUTPUT_DIR=../output  输出目录
#   LLAMA3_MODEL=/var/luzhenyan/Meta-Llama-3-8B-Instruct
#   QWEN3_MODEL=/var/wangyicheng/models/Qwen3-8B

set -e

NUM_GPUS="${NUM_GPUS:-8}"
PROCS_PER_GPU="${PROCS_PER_GPU:-4}"
TOTAL_SHARDS=$((NUM_GPUS * PROCS_PER_GPU))  # 32

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_GEN_DIR="$PROJECT_DIR/src/data_gen"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
LLAMA3_MODEL="${LLAMA3_MODEL:-/var/luzhenyan/Meta-Llama-3-8B-Instruct}"
QWEN3_MODEL="${QWEN3_MODEL:-/var/wangyicheng/models/Qwen3-8B}"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "并行数据生成"
echo "GPU 数量: $NUM_GPUS  每 GPU 进程数: $PROCS_PER_GPU  总分片: $TOTAL_SHARDS"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

LOG_DIR="$PROJECT_DIR/logs/data_generation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "日志目录: $LOG_DIR"

export WANDB_MODE=disabled

# ============================================
# 阶段 1: 生成模型回答（32 并行）
# ============================================

run_parallel() {
    local desc="$1"; shift
    local cmd_template="$1"; shift
    local PIDS=()
    echo ""
    echo ">>> $desc ($TOTAL_SHARDS shards)"
    for shard_id in $(seq 0 $((TOTAL_SHARDS - 1))); do
        gpu_id=$((shard_id % NUM_GPUS))
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            eval "${cmd_template/SHARD_ID/$shard_id}" \
                > "$LOG_DIR/${desc// /_}_shard${shard_id}.log" 2>&1
            echo "[$(date +%H:%M:%S)] Shard $shard_id (GPU $gpu_id) done: $desc"
        ) &
        PIDS+=($!)
        if [ $(((shard_id + 1) % PROCS_PER_GPU)) -eq 0 ]; then
            sleep 5
        fi
    done
    for pid in "${PIDS[@]}"; do wait $pid; done
    echo "✅ $desc 完成"
}

echo ""
echo "=== 阶段 1: 生成模型回答 ==="

run_parallel "Llama3 TriviaQA Train" \
    "python $DATA_GEN_DIR/generate_model_answers.py --model $LLAMA3_MODEL --dataset triviaqa --shard_id SHARD_ID --num_shards $TOTAL_SHARDS"

run_parallel "Llama3 TriviaQA Test" \
    "python $DATA_GEN_DIR/generate_model_answers.py --model $LLAMA3_MODEL --dataset triviaqa_test --shard_id SHARD_ID --num_shards $TOTAL_SHARDS"

run_parallel "Qwen3 Movies Train" \
    "python $DATA_GEN_DIR/generate_model_answers.py --model $QWEN3_MODEL --dataset movies --shard_id SHARD_ID --num_shards $TOTAL_SHARDS"

run_parallel "Qwen3 Movies Test" \
    "python $DATA_GEN_DIR/generate_model_answers.py --model $QWEN3_MODEL --dataset movies_test --shard_id SHARD_ID --num_shards $TOTAL_SHARDS"

# ============================================
# 阶段 1.5: 合并 CSV 分片
# ============================================

merge_csv_shards() {
    local pattern="$1"
    local output_file="$2"
    python -c "
import pandas as pd, glob, sys
files = sorted(glob.glob('$OUTPUT_DIR/$pattern'))
print(f'合并 {len(files)} 个分片: $pattern')
dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv('$OUTPUT_DIR/$output_file', index=False)
print(f'✅ {output_file}: {len(merged)} 样本')
"
}

echo ""
echo "=== 阶段 1.5: 合并分片 ==="
LLAMA3_FRIENDLY="llama3-8b-instruct"
QWEN3_FRIENDLY="qwen3-8b"

merge_csv_shards "${LLAMA3_FRIENDLY}-answers-triviaqa_shard*.csv"    "${LLAMA3_FRIENDLY}-answers-triviaqa.csv"
merge_csv_shards "${LLAMA3_FRIENDLY}-answers-triviaqa_test_shard*.csv" "${LLAMA3_FRIENDLY}-answers-triviaqa_test.csv"
merge_csv_shards "${QWEN3_FRIENDLY}-answers-movies_shard*.csv"        "${QWEN3_FRIENDLY}-answers-movies.csv"
merge_csv_shards "${QWEN3_FRIENDLY}-answers-movies_test_shard*.csv"   "${QWEN3_FRIENDLY}-answers-movies_test.csv"

# ============================================
# 阶段 2: 提取 exact_answer（32 并行）
# ============================================

echo ""
echo "=== 阶段 2: 提取 exact_answer ==="

run_parallel "Extract Llama3 TriviaQA Train" \
    "python $DATA_GEN_DIR/extract_exact_answer.py --model $LLAMA3_MODEL --extraction_model $LLAMA3_MODEL --dataset triviaqa --shard_id SHARD_ID --num_shards $TOTAL_SHARDS"

run_parallel "Extract Llama3 TriviaQA Test" \
    "python $DATA_GEN_DIR/extract_exact_answer.py --model $LLAMA3_MODEL --extraction_model $LLAMA3_MODEL --dataset triviaqa_test --shard_id SHARD_ID --num_shards $TOTAL_SHARDS"

run_parallel "Extract Qwen3 Movies Train" \
    "python $DATA_GEN_DIR/extract_exact_answer.py --model $QWEN3_MODEL --extraction_model $QWEN3_MODEL --dataset movies --shard_id SHARD_ID --num_shards $TOTAL_SHARDS"

run_parallel "Extract Qwen3 Movies Test" \
    "python $DATA_GEN_DIR/extract_exact_answer.py --model $QWEN3_MODEL --extraction_model $QWEN3_MODEL --dataset movies_test --shard_id SHARD_ID --num_shards $TOTAL_SHARDS"

# ============================================
# 阶段 2.5: 最终合并（含 exact_answer）
# ============================================

echo ""
echo "=== 阶段 2.5: 最终合并（含 exact_answer）==="
merge_csv_shards "${LLAMA3_FRIENDLY}-answers-triviaqa_shard*.csv"    "${LLAMA3_FRIENDLY}-answers-triviaqa_final.csv"
merge_csv_shards "${LLAMA3_FRIENDLY}-answers-triviaqa_test_shard*.csv" "${LLAMA3_FRIENDLY}-answers-triviaqa_test_final.csv"
merge_csv_shards "${QWEN3_FRIENDLY}-answers-movies_shard*.csv"        "${QWEN3_FRIENDLY}-answers-movies_final.csv"
merge_csv_shards "${QWEN3_FRIENDLY}-answers-movies_test_shard*.csv"   "${QWEN3_FRIENDLY}-answers-movies_test_final.csv"

echo ""
echo "=========================================="
echo "✅ 数据生成完成！输出目录: $OUTPUT_DIR"
echo "日志目录: $LOG_DIR"
echo "=========================================="
