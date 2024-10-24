import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
    GenerationConfig
)
from datasets import load_dataset
from peft import PeftModel, PeftConfig

from utils import HF_NAME_MAP
from utils import Logger, Evaluator
from utils import set_seed, init_tokenizer, validate_args


if __name__ == '__main__':
    logger = Logger(__name__)
    logger.info('Starting Evaluate script')
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--model_name', type=str, help='Model to evaluate', default=f"sft_llama-1b")
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
    hf_name = HF_NAME_MAP[args.pt_name]
    
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
        

    loader = DataLoader(_dataset['test'].select(range(100)), batch_size=1, shuffle=False)
    # loader = DataLoader(_dataset['test'], batch_size=8, shuffle=False)

    # Set evaluator
    evaluator = Evaluator(args.task_name)

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        **config['generator']
    )

    for ditem in tqdm(loader):
        question, answer = ditem['question'], ditem['answer']
        question = tokenizer(question, padding=True, truncation=True, return_tensors="pt", max_length=512)
        prediction = model.generate(
            inputs=question['input_ids'].to(model.device),
            attention_mask=question['attention_mask'].to(model.device),
            generation_config=gen_config,
        )
        generated = prediction[:, question['input_ids'].shape[1]:]
        prediction = tokenizer.batch_decode(generated, skip_special_tokens=True)

        evaluator.add(prediction, answer)
        print(f"Question: {ditem['question']}")
        print(f"Prediction: {prediction}")
        print(f"Answer: {ditem['answer']}")


    metric = evaluator.evaluate()
    print(f"Accuracy: {metric}")
        
