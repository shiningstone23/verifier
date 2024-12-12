import argparse
import random
from collections import defaultdict
from dotenv import load_dotenv; load_dotenv(override=True)
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import yaml
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset

from utils import HF_NAME_MAP
from utils import Logger, Evaluator, RegexLogitsProcessor
from utils import set_seed, init_tokenizer, validate_args


if __name__ == "__main__":
    logger = Logger(__name__)
    logger.info('Starting Evaluate script')
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--model_name', type=str, help='Model to evaluate', default=f"sft_llama-1b")
    parser.add_argument('--task_name', type=str, help='Task to evaluate on', default='gsm8k')
    parser.add_argument('--config_path', type=str, help='Path to config file', default='configs/basic.yml')
    parser.add_argument('--iter_num', type=int, help='Number of iterations', default=2)
    parser.add_argument('--gen_samples', type=int, help='Number of samples to generate', default=50)
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
    max_samples = config['sampling']['max_samples'][args.model_name]


    if args.task_name == "gsm8k":
        original = load_dataset("openai/gsm8k", "main")

    correct_set = load_from_disk(f"data/{args.model_name}_{args.task_name}_correct_{max_samples}_0")
    correct_set = concatenate_datasets([original['train'], correct_set])
    false_set = load_from_disk(f"data/{args.model_name}_{args.task_name}_false_{max_samples}_0")

    for i in range(1, args.iter_num+1):
        new_correct_set = load_from_disk(f"data/{args.model_name}_{args.task_name}_correct_{max_samples}_{i}")
        correct_set = concatenate_datasets([correct_set, new_correct_set])
        false_set = load_from_disk(f"data/{args.model_name}_{args.task_name}_false_{max_samples}_{i}")


    correct_answers = defaultdict(set)
    for key, value in zip(correct_set["question"], correct_set["answer"]):
        correct_answers[key].add(value)
    correct_answers = dict(correct_answers)

    false_answers = defaultdict(set)
    for key, value in zip(false_set["question"], false_set["answer"]):
        false_answers[key].add(value)
    false_answers = dict(false_answers)

    logger.info(f"Correct questions: {len(correct_answers)}, False questions: {len(false_answers)}")

    # Generate result
    result = set()  # Use a set to avoid duplicates
    instruction = "Please calculate the solution step-by-step and conclude the answer with \n#### followed by the result.\nQuestion: "
    for question, _correct_set in correct_answers.items():
        if question in false_answers:
            _false_set = false_answers[question]
            
            # Generate multiple unique pairs for each question
            for _ in range(args.gen_samples):
                chosen = random.choice(list(_correct_set))  # Always pick from correct answers
                rejected = random.choice(list(_false_set))  # Always pick from false answers
                question = instruction + question
                pair = (question, chosen, rejected)  # Tuple ensures uniqueness in set
                result.add(pair)

    # Convert set back to list of dictionaries
    final_result = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for entry in result:
        final_result["prompt"].append(entry[0])
        final_result["chosen"].append(entry[1])
        final_result["rejected"].append(entry[2])

    logger.info(f"Data size after filtering: {len(final_result['prompt'])}")

    Dataset.from_dict(final_result).train_test_split(test_size=0.1).save_to_disk(f"data/ver_{args.model_name}_{args.task_name}")

    logger.info("Data saved to disk")