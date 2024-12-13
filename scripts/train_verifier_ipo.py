import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import yaml
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
)
from trl import DPOConfig, DPOTrainer
from datasets import load_from_disk
from peft import PeftModel

from utils import HF_NAME_MAP
from utils import Logger
from utils import set_seed, init_tokenizer, validate_args

if __name__ == '__main__':
    logger = Logger(__name__)
    logger.info('Starting Verifier Trainig')
    parser = argparse.ArgumentParser(description='Verifier Training')
    parser.add_argument('--model_name', type=str, help='Model to train', default=f"sft_llama-1b")
    parser.add_argument('--task_name', type=str, help='Task to train on', default='gsm8k')
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
        hf_name,
        quantization_config=BitsAndBytesConfig(**config['qt']),
        **config['model'][pt_name]
    )

    model = PeftModel.from_pretrained(
        model, 
        model_id=model_path,
        is_trainable=True,
        adapter_name="target",
    )
    model.load_adapter(model_path, adapter_name="reference")

    # Load data
    if args.task_name == "gsm8k":
        train_dataset = load_from_disk(f"data/ver_{args.model_name}_{args.task_name}/train")
        eval_dataset = load_from_disk(f"data/ver_{args.model_name}_{args.task_name}/test")

    config['dpo_trainer']['learning_rate'] = float(config['dpo_trainer']['learning_rate'])
    config['dpo_trainer']['output_dir'] = config['dpo_trainer']['output_dir'] + f"/veri_{args.model_name}_{args.task_name}"

    config['dpo_trainer']['loss_type'] = "ipo"
    config['dpo_trainer']['output_dir'] = "./models/verifier/ipo/checkpoints"
    config['dpo_trainer']['save_strategy'] = "epoch"
    config['dpo_trainer']['save_steps'] = None
    
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=DPOConfig(**config['dpo_trainer']), # TODO : Fix why the logging is not working
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    # Save
    trainer.save_model(
        f"models/veri_{args.model_name}_{args.task_name}_ipo",
    )

    logger.info('Complete Training Verifier With IPO')
        


        
