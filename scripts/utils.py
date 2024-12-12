import re
import random
import numpy as np
import torch
import logging
from transformers import StoppingCriteria, LogitsProcessor

# ENUMS
HF_NAME_MAP = {
    "llama-1b": "meta-llama/Llama-3.2-1B",
    "llama-3b": "meta-llama/Llama-3.2-3B",
    "llama-8b": "meta-llama/Llama-3.1-8B"
}


# CLASSES
class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

class Evaluator:
    def __init__(self, name):
        self.name = name
        self.data = {
            "predictions": [],
            "ground_truths": []
        }

    def add(self, prediction, ground_truth=None):
        if isinstance(prediction, list):
            self.data["predictions"].extend(prediction)
        else:
            raise ValueError("Predictions should be a list of lists")
        
        if ground_truth is not None:
            if isinstance(ground_truth, list):
                self.data["ground_truths"].extend(ground_truth)
            else:
                raise ValueError("Ground truths should be a list of lists")
    
    def clear(self):
        self.data = {
            "predictions": [],
            "ground_truths": []
        }

    def extract_answer(self, prediction, ground_truth):
        if len(prediction) != len(ground_truth):
            raise ValueError("Prediction and Ground Truth should have the same length")
        
        predicted_answer = []
        gt_number = []
        for pred, gt in zip(prediction, ground_truth):
            pn, gn = _extract_answer(pred, gt)

            predicted_answer.append(pn)
            gt_number.append(gn)

        return predicted_answer, gt_number

    def evaluate(self):
        if len(self.data["predictions"]) == 0:
            return 0, []
        
        predictions = self.data["predictions"]
        ground_truths = self.data["ground_truths"]

        total = len(predictions)
        correct = 0
        details = []

        predicted_answer, gt_number = self.extract_answer(predictions, ground_truths)
        for pred, gt in zip(predicted_answer, gt_number):
            is_correct = pred == gt

            # Increment the correct count if the answer matches
            if is_correct:
                correct += 1

            # Store the comparison details for each prediction
            details.append({
                'prediction': pred,
                'predicted_answer': predicted_answer,
                'ground_truth': gt_number,
                'is_correct': is_correct
            })

        # Calculate the accuracy
        accuracy = correct / total * 100 if total > 0 else 0

        return accuracy, details
    
class RegexStopAndExtractCriteria(StoppingCriteria):
    def __init__(self, pattern, tokenizer):
        super().__init__()
        self.pattern = pattern  # 정규표현식 패턴
        self.tokenizer = tokenizer
        self.extracted_value = None  # 추출한 값을 저장할 변수

    def __call__(self, input_ids, scores, **kwargs):
        # 생성된 텍스트를 디코딩
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # 정규표현식 패턴을 이용해 매칭 확인
        match = re.search(self.pattern, decoded_text)
        if match:
            self.extracted_value = match.group(1)  # 숫자 부분 추출
            return True  # 패턴이 매칭되면 멈춤
        return False  # 패턴이 매칭되지 않으면 계속 생성
    

class RegexLogitsProcessor(LogitsProcessor):
    def __init__(self, pattern, tokenizer):
        """
        Args:
            pattern (str): 탐지할 정규식 패턴.
            tokenizer (Tokenizer): 토크나이저 객체.
        """
        self.pattern = pattern
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        """
        Args:
            input_ids (torch.Tensor): (n_samples, seq_len) 형태의 입력 토큰 ID.
            scores (torch.Tensor): (n_samples, n_vocab) 형태의 로그 확률 점수.

        Returns:
            torch.Tensor: 수정된 로그 확률 점수.
        """
        pad_token = self.tokenizer.pad_token_id

        for i in range(input_ids.size(0)):
            # 전체 시퀀스를 디코딩
            decoded = self.tokenizer.decode(input_ids[i])
            # 패턴이 디코딩된 텍스트에서 발견되면 처리
            if re.search(self.pattern, decoded):
                scores[i, :] = -1e9  # 모든 토큰 비활성화
                scores[i, pad_token] = 0  # pad_token만 활성화

        return scores

def _extract_answer(prediction, ground_truth):
    pn = None
    matched = re.search(r'####\s*\d+', prediction)
    if matched:
        try:
            pn = matched.group(0).split('####')[-1].strip()
            _ = int(pn)
        except ValueError:
            pass

    # Compare the extracted answer with the ground truth
    gn = re.search(r'####\s*-?\d+', ground_truth).group(0).split('####')[-1].strip()

    return pn, gn

# FUNCTIONS
def set_seed(seed):
    # set all seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_tokenizer(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def validate_args(args):
    # TODO
    pass

def formatting_func(item):
    instruction = "Please calculate the solution step-by-step and conclude the answer with '\n#### ' followed by the result.\n"
    question, answer = item['question'], item['answer']
    return f"{instruction}Question: {question} \nAnswer: {answer}"