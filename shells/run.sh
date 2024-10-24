#!/bin/bash
nohup python scripts/train_sft.py --pt_name llama-1b --task_name gsm8k --config_path configs/basic.yml >> logs/train_sft.log 2>&1 &
nohup python scripts/train_sft.py --pt_name llama-3b --task_name gsm8k --config_path configs/basic.yml >> logs/train_sft.log 2>&1 &
nohup python scripts/train_sft.py --pt_name llama-8b --task_name gsm8k --config_path configs/basic.yml >> logs/train_sft.log 2>&1 &


nohup python scripts/eval_model.py --model_name llama-1b --task_name gsm8k --config_path configs/basic.yml >> logs/evaluate.log 2>&1 &