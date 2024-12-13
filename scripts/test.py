# nohup python scripts/test.py >> logs/test.log 2>&1 &
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk

from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
    
)
from tqdm import tqdm
import json
from utils import HF_NAME_MAP
from utils import set_seed, init_tokenizer, validate_args, _extract_answer

config_path = "configs/basic.yml"
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def is_answer(answer, pred):
    pn, gn = _extract_answer(pred, answer)
    return pn == gn

def get_likelihood(verifier, tokenizer, question, predictions):
    instruction = ""
    questions = [instruction + question + pred for pred in predictions]

    # Step 2: Tokenize 입력 배치
    batch_inputs = tokenizer(questions, return_tensors='pt', padding=True, padding_side='right', max_length=896, truncation=True)
    attention_mask = batch_inputs["attention_mask"]


    # Step 3: Tokenize 정답 레이블 (패딩 포함)
    label_tokens = tokenizer(predictions, return_tensors="pt", padding=True, padding_side='right', max_length=896, truncation=True)

    # Step 4: Verifier 모델 호출 (배치 처리)
    with torch.no_grad():
        logits = verifier(batch_inputs.input_ids, attention_mask=attention_mask).logits

    # Step 5: Mask 생성 및 로그 확률 계산
    label_mask = (label_tokens.input_ids != tokenizer.pad_token_id)  # 패딩이 아닌 부분은 True
    shifted_labels = label_tokens.input_ids[:, 1:]  # 첫 번째 토큰 제외 (Decoder 방식)

    log_probs = torch.gather(
        logits[:, -shifted_labels.shape[1]-1:, :].log_softmax(-1),  # Label 길이에 맞춘 logits
        dim=2,
        index=shifted_labels.unsqueeze(2)  # 레이블 차원 확장
    ).squeeze(2)  # [Batch, Sequence]

    # Mask 적용 후, 각 문장의 총 로그 확률 계산
    masked_log_probs = log_probs * label_mask[:, 1:].float()  # 첫 번째 패딩 제외
    total_log_probs = masked_log_probs.sum(dim=1)  # 배치별 로그 확률 합계

    return total_log_probs.detach().cpu(), masked_log_probs

model_name = "sft_llama-8b"
task_name = "gsm8k"
model_type, pt_name = model_name.split("_")
hf_name = HF_NAME_MAP[pt_name]

tokenizer = AutoTokenizer.from_pretrained(hf_name)
init_tokenizer(tokenizer)

dset = load_from_disk("data/ver_sft_llama-1b_gsm8k/test")
train_set = load_from_disk("data/ver_sft_llama-8b_gsm8k/train")

gen_model = AutoModelForCausalLM.from_pretrained(
    "/home/chanwoo/chanwoo/repo/verifier/models/sft_llama-8b_gsm8k",
    # "meta-llama/Llama-3.1-8B",
    quantization_config=BitsAndBytesConfig(**config['qt']),
    **config['model'][pt_name]
)

train_res = []
for idx in range(1000):
    question = train_set[idx]['prompt']
    predictions = [train_set[idx]['chosen'], train_set[idx]['rejected']]
    lik, masked_log_probs = get_likelihood(gen_model, tokenizer, question, predictions)
    train_res.append({
        "question": question,
        "predictions": predictions,
        "likelihood": lik.numpy(),
        "masked_log_probs": masked_log_probs.numpy()
    })

    if idx % 100 == 0:
        print(f"idx: {idx}")
        with open("train_res.pkl", 'wb') as f:
            pickle.dump(train_res, f)
with open("train_res.pkl", 'wb') as f:
    pickle.dump(train_res, f)