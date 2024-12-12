#!/bin/bash

# Example: ./shells/train_star.sh sft_llama-1b 3 0
# Example: ./shells/train_star.sh sft_llama-8b 3 1
# Example: nohup ./train_star.sh sft_llama-8b 3 0 >> logs/sh.log 2>&1 &

MODEL_NAME=$1  # 첫 번째 argument로 모델 이름
ITERATIONS=$2  # 두 번째 argument로 반복 횟수
CUDA_DEVICE=$3  # 세 번째 argument로 CUDA_VISIBLE_DEVICES 설정

if [ -z "$MODEL_NAME" ] || [ -z "$ITERATIONS" ] || [ -z "$CUDA_DEVICE" ]; then
    echo "Usage: $0 <model_name> <iterations> <cuda_device>"
    exit 1
fi

# CUDA_VISIBLE_DEVICES 설정
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
echo "CUDA_VISIBLE_DEVICES is set to $CUDA_DEVICE"

for ((i = 0; i < ITERATIONS; i++)); do
    echo "Running iteration $i for model $MODEL_NAME..."
    if [ "$i" -ge 2 ]; then
        # iter_num != 0일 때만 collect_dset.py 실행
        python scripts/collect_dset.py \
            --model_name "$MODEL_NAME" \
            --task_name gsm8k \
            --config_path configs/basic.yml \
            --iter_num $i >> "logs/collect_dset-${MODEL_NAME}.log" 2>&1
    else
        echo "Skipping collect_dset.py for iteration $i (already executed)."
    fi

    # train_star.py 실행
    if [ "$i" -ge 1 ]; then
        python scripts/train_star.py \
            --model_name "$MODEL_NAME" \
            --task_name gsm8k \
            --config_path configs/basic.yml \
            --iter_num $i >> "logs/train_star-${MODEL_NAME}.log" 2>&1
    else
        echo "Skipping train_star.py for iteration $i (already executed)."
    fi
done

echo "All iterations completed for model $MODEL_NAME."

# # Iteration 1
# nohup python scripts/collect_dset.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml --iter_num 0 >> logs/collect_dset-1b.log 2>&1 &
# nohup python scripts/collect_dset.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --iter_num 0 >> logs/collect_dset-8b.log 2>&1 &

# nohup python scripts/train_star.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml --iter_num 0 >> logs/train_star-1b.log 2>&1 &
# nohup python scripts/train_star.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --iter_num 0 >> logs/train_star-8b.log 2>&1 &

# # Iteration 2
# nohup python scripts/collect_dset.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml --iter_num 1 >> logs/collect_dset-1b.log 2>&1 &
# nohup python scripts/collect_dset.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --iter_num 1 >> logs/collect_dset-8b.log 2>&1 &

# nohup python scripts/train_star.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml --iter_num 1 >> logs/train_star-1b.log 2>&1 &
# nohup python scripts/train_star.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --iter_num 1 >> logs/train_star-8b.log 2>&1 &

# # Iteration 2
# nohup python scripts/collect_dset.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml --iter_num 2 >> logs/collect_dset-1b.log 2>&1 &
# nohup python scripts/collect_dset.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --iter_num 2 >> logs/collect_dset-8b.log 2>&1 &

# nohup python scripts/train_star.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml --iter_num 2 >> logs/train_star-1b.log 2>&1 &
# nohup python scripts/train_star.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --iter_num 2 >> logs/train_star-8b.log 2>&1 &

