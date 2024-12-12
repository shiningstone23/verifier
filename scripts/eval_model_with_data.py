import os ; os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import argparse
from tqdm import tqdm
import torch
import yaml
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
)

from utils import HF_NAME_MAP
from utils import Logger
from utils import set_seed, init_tokenizer, validate_args, _extract_answer

def is_answer(answer, pred):
            pn, gn = _extract_answer(pred, answer)
            return pn == gn

def get_likelihood(model, tokenizer, question, predictions, answer):
    instruction = ""
    questions = [instruction + question + pred for pred in predictions]

    # Step 2: Tokenize 입력 배치
    batch_inputs = tokenizer(questions, return_tensors='pt', padding=True, padding_side='right')
    attention_mask = batch_inputs["attention_mask"]


    # Step 3: Tokenize 정답 레이블 (패딩 포함)
    label_tokens = tokenizer(predictions, return_tensors="pt", padding=True, padding_side='right')

    # Step 4: Verifier 모델 호출 (배치 처리)
    logits = verifier(batch_inputs.input_ids, attention_mask=attention_mask).logits.detach().cpu()

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

    return total_log_probs



if __name__ == '__main__':
    logger = Logger(__name__)
    logger.info('Starting Evaluate script')
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--model_name', type=str, help='Model to evaluate', default=f"sft_llama-1b")
    parser.add_argument('--task_name', type=str, help='Task to evaluate on', default='gsm8k')
    parser.add_argument('--config_path', type=str, help='Path to config file', default='configs/basic.yml')
    parser.add_argument('--eval_set', type=str, help='Path to evaluation set')
    args = parser.parse_args()
    validate_args(args)

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    set_seed(config['seed'])

    logger.info(f"Arguments: {args}")
    logger.info(f"Config: {config}")

    # Load model
    model_path = f"models/{args.model_name}_{args.task_name}"
    model_type, pt_name = args.model_name.split("_")
    hf_name = HF_NAME_MAP[pt_name]
    
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    init_tokenizer(tokenizer)
    
    verifier_path = f"models/veri_{args.model_name}_{args.task_name}/target"
    verifier = AutoModelForCausalLM.from_pretrained(
        verifier_path,
        quantization_config=BitsAndBytesConfig(**config['qt']),
        **config['model'][pt_name]
    )

    with open(f"data/{args.eval_set}.json", "r") as f:
        samples = json.load(f)

    _iter = 0
    n_correct = 0
    for ditem in tqdm(samples):
        question = ditem['Question'][0]
        predictions = ditem['Prediction']
        answer = ditem['Answer'][0]

        likelihood = get_likelihood(verifier, tokenizer, question, predictions, answer)
        ans_idx = likelihood.argmax()
        pred = predictions[ans_idx]

        is_answer_list = [is_answer(answer, pred) for pred in predictions]
        total_num_answer = sum(is_answer_list)

        logger.info("\n")
        logger.info(f"Question: {question}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Correct answers: {is_answer_list}")
        logger.info(f"Number of correct answers: {total_num_answer}")
        logger.info(f"Best answer index: {ans_idx}")
        logger.info(f"Is Verifier Correct: {is_answer_list[ans_idx]}")
        logger.info(f"Best Pred: {predictions[ans_idx]}")
        logger.info(f"Likelihood: {likelihood}")

        _iter += 1
    
        
