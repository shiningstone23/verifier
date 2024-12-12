import os
import argparse
# from dotenv import load_dotenv; load_dotenv(override=True)
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
    GenerationConfig, 
    StoppingCriteriaList,
)
from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset
from datasets import load_from_disk, load_dataset, concatenate_datasets
from peft import get_peft_model, LoraConfig

from utils import HF_NAME_MAP
from utils import Logger, Evaluator, RegexStopAndExtractCriteria
from utils import set_seed, init_tokenizer, validate_args, formatting_func, _extract_answer


if __name__ == '__main__':
    logger = Logger(__name__)
    logger.info('Starting STaR Trainig')
    parser = argparse.ArgumentParser(description='STaR Training')
    parser.add_argument('--model_name', type=str, help='Model to train', default=f"sft_llama-8b")
    parser.add_argument('--task_name', type=str, help='Task to train on', default='gsm8k')
    parser.add_argument('--config_path', type=str, help='Path to config file', default='configs/basic.yml')
    parser.add_argument('--iter_num', type=int, help='Iteration number', default=0)
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

    peft_model = get_peft_model(
        model, 
        peft_config=LoraConfig(**config['lora'])
    )


    # Load data
    if args.task_name == "gsm8k":
        _dataset = load_dataset("openai/gsm8k", "main")

    eval_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=_dataset['test'],
        formatting_func=formatting_func,
        **config['dataset']
    )
        
    evaluator = Evaluator(args.task_name)

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        **config['generator']
    )

    instruction = "Please calculate the solution step-by-step and conclude the answer with \n#### followed by the result.\n"
    stopping_criteria = StoppingCriteriaList([
        RegexStopAndExtractCriteria(pattern=r"####\s*\d+(\D)", tokenizer=tokenizer)
    ])

    _iter = args.iter_num
    cnt = config['sampling']['max_samples'][args.model_name]
    correct_dset = load_from_disk(f"data/{args.model_name}_{args.task_name}_correct_{cnt}_{_iter}")
    for i in range(1,_iter+1):
        new_dataset = load_from_disk(f"data/{args.model_name}_{args.task_name}_correct_{cnt}_{i}")
        correct_dset = concatenate_datasets([correct_dset, new_dataset])

    train_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=concatenate_datasets([_dataset['train'], correct_dset]),
        formatting_func=formatting_func,
        **config['dataset']
    )

    config['trainer']['learning_rate'] = float(config['trainer']['learning_rate'])
    trainer = SFTTrainer(
        model=peft_model,
        args=SFTConfig(**config['star_trainer']),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save
    trainer.save_model(
        f"models/{args.model_name}_{args.task_name}_star_{_iter}"
    )

    logger.info(f"STAR Finetuning complete for iteration {_iter}")

        # print(f"Question: {ditem['question']}", flush=True)
        # print(f"Prediction: {prediction}", flush=True)
        # print(f"Answer: {ditem['answer']}", flush=True)


        
