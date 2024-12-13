#!/bin/bash
# nohup python scripts/train_sft.py --pt_name llama-1b --task_name gsm8k --config_path configs/basic.yml >> logs/train_sft-1b.log 2>&1 &
# nohup python scripts/train_sft.py --pt_name llama-3b --task_name gsm8k --config_path configs/basic.yml >> logs/train_sft.log 2>&1 &
# nohup python scripts/train_sft.py --pt_name llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/train_sft.log 2>&1 &

nohup python scripts/collect_dset.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml >> logs/collect_dset-1b.log 2>&1 &
# nohup python scripts/train_star.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --max_iter 3 >> logs/train_star.log 2>&1 &
nohup python scripts/collect_dset.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/collect_dset-8b.log 2>&1 &
# nohup python scripts/collect_dset.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --iter_num 0 >> logs/collect_dset.log 2>&1 &


# nohup python scripts/eval_model.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml >> logs/evaluate.log 2>&1 &
# nohup python scripts/eval_model.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/evaluate.log 2>&1 &

# nohup python scripts/eval_model.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml >> logs/eval-1b.log 2>&1 &
# nohup python scripts/eval_model.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/eval-8b.log 2>&1 &

nohup python scripts/train_verifier.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml >> logs/train_verifier-1b.log 2>&1 &
nohup python scripts/train_verifier.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/train_verifier-8b.log 2>&1 &


nohup python scripts/eval_model.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml --verifier >> logs/test_verifier.log 2>&1 &
nohup python scripts/eval_model.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/test_verifier-8b.log 2>&1 &

nohup python scripts/eval_model.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/eval-star-8b.log 2>&1 &

nohup python scripts/eval_model.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml >> logs/eval-star-1b.log 2>&1 &
nohup python scripts/eval_model.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/eval-star-8b.log 2>&1 &

nohup python scripts/eval_model_with_data.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml --eval_set multi_samples_star-1b >> logs/eval_with_d-1b.log 2>&1 &
nohup python scripts/eval_model_with_data.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --eval_set multi_samples_star-8b >> logs/eval_with_d-8b.log 2>&1 &
nohup python scripts/eval_model_with_data.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml --eval_set multi_samples-1b >> logs/eval_with_d-1b.log 2>&1 &
nohup python scripts/eval_model_with_data.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml --eval_set multi_samples-8b >> logs/eval_with_d-8b.log 2>&1 &


nohup python scripts/train_verifier_ipo.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/train_verifier-ipo-8b.log 2>&1 &
# nohup python scripts/gen_verifier_data.py --model_name sft_llama-1b --task_name gsm8k --config_path configs/basic.yml
# nohup python scripts/gen_verifier_data.py --model_name sft_llama-8b --task_name gsm8k --config_path configs/basic.yml