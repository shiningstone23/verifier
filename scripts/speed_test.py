# python scripts/speed_test.py
import argparse
import time
from dotenv import load_dotenv; load_dotenv(override=True)
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
    GenerationConfig, 
    StoppingCriteriaList,
)
from datasets import load_dataset
from utils import HF_NAME_MAP
from utils import Logger, RegexStopAndExtractCriteria
from utils import set_seed, init_tokenizer, validate_args
import torch

if __name__ == '__main__':
    logger = Logger(__name__)
    logger.info('Starting Evaluate script')

    parser = argparse.ArgumentParser(description='Evaluate a model with and without torch.compile')
    parser.add_argument('--model_name', type=str, help='Model to evaluate', default=f"sft_llama-8b")
    parser.add_argument('--task_name', type=str, help='Task to evaluate on', default='gsm8k')
    parser.add_argument('--config_path', type=str, help='Path to config file', default='configs/basic.yml')
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(**config['qt']),
        **config['model'][pt_name]
    )

    # Load data
    if args.task_name == "gsm8k":
        _dataset = load_dataset("openai/gsm8k", "main")
    loader = DataLoader(_dataset['test'].select(range(4)), batch_size=1, shuffle=False)

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        **config['generator']
    )
    instruction = "Please calculate the solution step-by-step and conclude the answer with \n#### followed by the result.\n"
    stopping_criteria = StoppingCriteriaList([
        RegexStopAndExtractCriteria(pattern=r"####\s*\d+(\D)", tokenizer=tokenizer)
    ])

    # Measure inference time without torch.compile
    logger.info("Running without torch.compile...")
    start_time = time.time()
    for ditem in tqdm(loader):
        question, answer = ditem['question'], ditem['answer']
        question = [f"{instruction}Question: {q}" for q in question]
        question = tokenizer(question, padding=True, truncation=True, return_tensors="pt", max_length=512)
        prediction = model.generate(
            inputs=question['input_ids'].to(model.device),
            attention_mask=question['attention_mask'].to(model.device),
            generation_config=gen_config,
            stopping_criteria=stopping_criteria
        )
    no_compile_time = time.time() - start_time
    logger.info(f"Inference time without torch.compile: {no_compile_time:.2f} seconds")

    # Apply torch.compile and measure inference time
    logger.info("Running with torch.compile...")
    compiled_model = torch.compile(model, mode="reduce-overhead")
    start_time = time.time()
    for ditem in tqdm(loader):
        question, answer = ditem['question'], ditem['answer']
        question = [f"{instruction}Question: {q}" for q in question]
        question = tokenizer(question, padding=True, truncation=True, return_tensors="pt", max_length=512)
        prediction = compiled_model.generate(
            inputs=question['input_ids'].to(compiled_model.device),
            attention_mask=question['attention_mask'].to(compiled_model.device),
            generation_config=gen_config,
            stopping_criteria=stopping_criteria
        )
    compile_time = time.time() - start_time
    logger.info(f"Inference time with torch.compile: {compile_time:.2f} seconds")

    # Compare results
    logger.info(f"Performance improvement: {(no_compile_time / compile_time):.2f}x faster with torch.compile")
