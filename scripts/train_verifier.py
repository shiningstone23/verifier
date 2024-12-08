import os
import argparse
from dotenv import load_dotenv; load_dotenv(override=True)
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
)
from trl import DPOConfig, DPOTrainer
from trl.trainer import ConstantLengthDataset
from datasets import load_dataset, concatenate_datasets

from utils import HF_NAME_MAP
from utils import Logger
from utils import set_seed, init_tokenizer, validate_args, formatting_func

if __name__ == '__main__':
    logger = Logger(__name__)
    logger.info('Starting Verifier Trainig')
    parser = argparse.ArgumentParser(description='Verifier Training')
    parser.add_argument('--model_name', type=str, help='Model to train', default=f"sft_llama-1b")
    parser.add_argument('--task_name', type=str, help='Task to train on', default='gsm8k')
    parser.add_argument('--config_path', type=str, help='Path to config file', default='configs/basic.yml')
    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations', default=3)
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
    # model.forward = th_compile(model.forward, mode="reduce-overhead", fullgraph=True)


    # Load data
    max_iter = 0
    if args.task_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
        def func(x):
            x['label'] = 1

        dataset['train'].map(label=lambda x : 1)
        for dir_name in os.listdir("data"):
            logger.info(f"Find data for training verifier : {dir_name}")
            if args.model_name in dir_name:
                if "correct" in dir_name:
                    _dataset = load_dataset(f"data/{dir_name}")
                    _dataset.map(label=lambda x : 1)
                else:
                    _dataset = load_dataset(f"data/{dir_name}")
                    _dataset.map(label=lambda x : 0)

                _iter = int(dir_name.split("_")[-1])
                if _iter >= max_iter:
                    max_iter = _iter

                logger.info(f"Dataset size : {_dataset.num_rows}")
                dataset['train'] = concatenate_datasets([dataset['train'], _dataset])

        
        train_dataset = ConstantLengthDataset(
            tokenizer=tokenizer,
            # dataset=_dataset['train'].select(range(400)),
            dataset=dataset['train'],
            formatting_func=formatting_func,
            **config['dataset']
        )
        eval_dataset = ConstantLengthDataset(
            tokenizer=tokenizer,
            # dataset=_dataset['test'].select(range(100)),
            dataset=dataset['test'],
            formatting_func=formatting_func,
            **config['dataset']
        )

    config['trainer']['learning_rate'] = float(config['trainer']['learning_rate'])
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(**config['dpo_trainer']), # TODO : Fix why the logging is not working
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save
    trainer.save_model(
        f"models/veri_{args.pt_name}_{args.task_name}_{max_iter}"
    )

    logger.info('Complete Training Verifier')
        


        
