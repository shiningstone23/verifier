import argparse
import yaml
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer
)
from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

from utils import HF_NAME_MAP
from utils import Logger
from utils import set_seed, validate_args

if __name__ == '__main__':
    logger = Logger(__name__)
    logger.info('Starting Supervised FineTuning(SFT) script')
    parser = argparse.ArgumentParser(description='Finetune a model')
    parser.add_argument('--pt_name', type=str, help='Name of pre-trained model', default='llama-1b')
    parser.add_argument('--task_name', type=str, help='Task to finetune on', default='gsm8k')
    parser.add_argument('--config_path', type=str, help='Path to config file', default='configs/basic.yml')
    args = parser.parse_args()
    validate_args(args)

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    set_seed(config['seed'])

    logger.info(f"Arguments: {args}")
    logger.info(f"Config: {config}")

    # Load model
    hf_name = HF_NAME_MAP[args.pt_name]
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        quantization_config=BitsAndBytesConfig(**config['qt']),
        **config['model'][args.pt_name],
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    # Load data
    if args.task_name == "gsm8k":
        # TODO : GO to util.py and add the following function
        def formatting_func(item):
            question, answer = item['question'], item['answer']
            return f"Question: {question} \nAnswer: {answer}"
        
        _dataset = load_dataset("openai/gsm8k", "main")
        train_dataset = ConstantLengthDataset(
            tokenizer=tokenizer,
            # dataset=_dataset['train'].select(range(400)),
            dataset=_dataset['train'],
            formatting_func=formatting_func,
            **config['dataset']
        )
        eval_dataset = ConstantLengthDataset(
            tokenizer=tokenizer,
            # dataset=_dataset['test'].select(range(100)),
            dataset=_dataset['test'],
            formatting_func=formatting_func,
            **config['dataset']
        )
        

    # Apply PEFT to the model
    peft_model = get_peft_model(
        model, 
        peft_config=LoraConfig(**config['lora'])
    )

    # Trainer setup
    config['trainer']['learning_rate'] = float(config['trainer']['learning_rate'])
    trainer = SFTTrainer(
        model=peft_model,
        args=SFTConfig(**config['trainer']), # TODO : Fix why the logging is not working
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save
    trainer.save_model(
        f"models/sft_{args.pt_name}_{args.task_name}"
    )

    logger.info('Finetuning complete')

        




