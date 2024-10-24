import random
import numpy as np
import torch
import logging

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

    def evaluate(self):
        if len(self.data["predictions"]) == 0:
            return 0, []
        
        predictions = self.data["predictions"]
        ground_truths = self.data["ground_truths"]

        total = len(predictions)
        correct = 0
        details = []

        for pred, gt in zip(predictions, ground_truths):
            # Check if '###' is present in the prediction
            if '###' in pred:
                # Extract the answer part after '###'
                predicted_answer = pred.split('###')[-1].strip()
            else:
                # If '###' is not present, it's an incorrect prediction
                predicted_answer = None

            # Compare the extracted answer with the ground truth
            is_correct = predicted_answer == gt.strip()

            # Increment the correct count if the answer matches
            if is_correct:
                correct += 1

            # Store the comparison details for each prediction
            details.append({
                'prediction': pred,
                'predicted_answer': predicted_answer,
                'ground_truth': gt,
                'is_correct': is_correct
            })

        # Calculate the accuracy
        accuracy = correct / total * 100 if total > 0 else 0

        return accuracy, details


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