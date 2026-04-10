#!/bin/bash
source /root/miniconda3/bin/activate halluc

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd ${PROJECT_ROOT}/src
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"
export WANDB_MODE=disabled

MODEL_PATH="/var/luzhenyan/Meta-Llama-3-8B-Instruct"
FRIENDLY="llama3-8b-instruct"
DATASET_TRAIN="winobias"
DATASET_TEST="winobias_test"
N_TRAIN=2000
N_TEST=500
NUM_GPUS=8
NUM_SHARDS=8  # 减少分片数，每个GPU 1 个任务

PROBE_DIR="${PROJECT_ROOT}/probe/${FRIENDLY}_${DATASET_TRAIN}"
mkdir -p ${PROBE_DIR}/{prefill,decode}
LOG="${PROJECT_ROOT}/logs/llama3_winobias_$(date +%Y%m%d_%H%M%S).log"

echo "===== Llama3 + winobias 实验 =====" | tee -a $LOG
echo "日志: $LOG" | tee -a $LOG

# Step 1: 生成训练数据
echo "[1/6] 生成训练数据 (${NUM_SHARDS} 并行)..." | tee -a $LOG
cd ${PROJECT_ROOT}/src/data_gen
PIDS=()
for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=$((i % NUM_GPUS))
    CUDA_VISIBLE_DEVICES=$GPU_ID python generate_model_answers.py \
        --model ${MODEL_PATH} --dataset ${DATASET_TRAIN} \
        --n_samples ${N_TRAIN} --shard_id $i --num_shards ${NUM_SHARDS} >> $LOG 2>&1 &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done

# 合并分片
python -c "
import pandas as pd
import torch
import glob
import os

base = '${PROJECT_ROOT}/src/output/${FRIENDLY}-answers-${DATASET_TRAIN}'
csv_shards = sorted(glob.glob(base + '_shard*.csv'), key=lambda x: int(x.split('_shard')[1].split('.')[0]))
if csv_shards:
    dfs = [pd.read_csv(s) for s in csv_shards]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(base + '.csv', index=False)
    print(f'CSV合并: {len(merged)} 条')
    for s in csv_shards: os.remove(s)

pt_base = base.replace('-answers-', '-input_output_ids-')
pt_shards = sorted(glob.glob(pt_base + '_shard*.pt'), key=lambda x: int(x.split('_shard')[1].split('.')[0]))
if pt_shards:
    all_ids = []
    for s in pt_shards:
        all_ids.extend(torch.load(s, map_location='cpu'))
        os.remove(s)
    torch.save(all_ids, pt_base + '.pt')
    print(f'PT合并: {len(all_ids)} 条')
" 2>&1 | tee -a $LOG
echo "✓ 训练数据完成" | tee -a $LOG

# Step 2: 生成测试数据
echo "[2/6] 生成测试数据..." | tee -a $LOG
PIDS=()
for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_ID=$((i % NUM_GPUS))
    CUDA_VISIBLE_DEVICES=$GPU_ID python generate_model_answers.py \
        --model ${MODEL_PATH} --dataset ${DATASET_TEST} \
        --n_samples ${N_TEST} --shard_id $i --num_shards ${NUM_SHARDS} >> $LOG 2>&1 &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done

python -c "
import pandas as pd
import torch
import glob
import os

base = '${PROJECT_ROOT}/src/output/${FRIENDLY}-answers-${DATASET_TEST}'
csv_shards = sorted(glob.glob(base + '_shard*.csv'), key=lambda x: int(x.split('_shard')[1].split('.')[0]))
if csv_shards:
    dfs = [pd.read_csv(s) for s in csv_shards]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(base + '.csv', index=False)
    print(f'CSV合并: {len(merged)} 条')
    for s in csv_shards: os.remove(s)

pt_base = base.replace('-answers-', '-input_output_ids-')
pt_shards = sorted(glob.glob(pt_base + '_shard*.pt'), key=lambda x: int(x.split('_shard')[1].split('.')[0]))
if pt_shards:
    all_ids = []
    for s in pt_shards:
        all_ids.extend(torch.load(s, map_location='cpu'))
        os.remove(s)
    torch.save(all_ids, pt_base + '.pt')
    print(f'PT合并: {len(all_ids)} 条')
" 2>&1 | tee -a $LOG
echo "✓ 测试数据完成" | tee -a $LOG

# Step 3: 提取 exact answer
echo "[3/6] 提取 exact answer..." | tee -a $LOG
for DS in ${DATASET_TRAIN} ${DATASET_TEST}; do
    PIDS=()
    for ((i=0; i<NUM_SHARDS; i++)); do
        GPU_ID=$((i % NUM_GPUS))
        CUDA_VISIBLE_DEVICES=$GPU_ID python extract_exact_answer.py \
            --model ${MODEL_PATH} --extraction_model ${MODEL_PATH} \
            --dataset ${DS} --shard_id $i --num_shards ${NUM_SHARDS} >> $LOG 2>&1 &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait $pid; done
    
    # 合并
    python -c "
import pandas as pd
import glob
import os
base = '${PROJECT_ROOT}/src/output/${FRIENDLY}-answers-${DS}'
shards = sorted(glob.glob(base + '_shard*.csv'), key=lambda x: int(x.split('_shard')[1].split('.')[0]))
if shards:
    dfs = [pd.read_csv(s) for s in shards]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(base + '.csv', index=False)
    print(f'${DS} 合并: {len(merged)} 条')
    for s in shards: os.remove(s)
" 2>&1 | tee -a $LOG
done
echo "✓ exact answer 提取完成" | tee -a $LOG

# Step 4-6: 训练 probes
echo "[4/6] 训练 Prefill Probe..." | tee -a $LOG
cd ${PROJECT_ROOT}/src/prefill
python train_prefill_probes_all_layers.py \
    --model ${MODEL_PATH} --dataset ${DATASET_TRAIN} \
    --layers 10 12 14 16 18 20 22 24 26 28 30 \
    --n_validation_samples 200 \
    --save_clf_dir ${PROBE_DIR}/prefill \
    --out_dir ${PROBE_DIR}/prefill 2>&1 | tee -a $LOG

echo "[5/6] 训练 Key Token Probe..." | tee -a $LOG
cd ${PROJECT_ROOT}/src/decode
python train_token_probe.py \
    --model ${MODEL_PATH} --dataset ${DATASET_TRAIN} \
    --probe_at mlp --layer 12 --neg_strategy mixed --hard_window 8 \
    --use_scaler --use_pca 256 --max_samples ${N_TRAIN} \
    --out_dir ${PROBE_DIR} 2>&1 | tee -a $LOG

echo "[6/6] 训练 Decode Probe..." | tee -a $LOG
python train_hallu_probes_at_key_positions.py \
    --model ${MODEL_PATH} --dataset ${DATASET_TRAIN} \
    --layers 12 16 20 24 28 --positions exact_answer_last_token \
    --max_samples ${N_TRAIN} --out_dir ${PROBE_DIR}/decode 2>&1 | tee -a $LOG

echo "===== 实验完成! =====" | tee -a $LOG
echo "Probes: ${PROBE_DIR}" | tee -a $LOG
