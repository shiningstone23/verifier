import os
import argparse
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
from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset
from datasets import Dataset, load_dataset, concatenate_datasets
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

    peft_model = get_peft_model(
        model, 
        peft_config=LoraConfig(**config['lora'])
    )
    # model.forward = th_compile(model.forward, mode="reduce-overhead", fullgraph=True)


    # Load data
    if args.task_name == "gsm8k":
        _dataset = load_dataset("openai/gsm8k", "main")

    eval_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=_dataset['test'].select(range(4)),
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

    for _iter in range(args.max_iter):
        logger.info(f"Starting iteration {_iter}")
        
        augmented_correct = {
            "question": [],
            "answer": []
        }
        augmented_false = {
            "question": [],
            "answer": []
        }

        prev_path = f"data/{args.model_name}_{args.task_name}_correct_{_iter-1}"
        if os.path.exists(prev_path):
            new_dataset = load_dataset(prev_path)
            _dataset = concatenate_datasets([_dataset, new_dataset])

        loader = DataLoader(_dataset['train'].select(range(10)), batch_size=1, shuffle=False)
        # loader = DataLoader(_dataset['train'].select(range(4)), batch_size=1, shuffle=False)

        for ditem in tqdm(loader):
            question, answer = ditem['question'], ditem['answer']
            question = [f"{instruction}Question: {q}" for q in question]
            question = tokenizer(
                question, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            prediction = model.generate(
                inputs=question['input_ids'].to(model.device),
                attention_mask=question['attention_mask'].to(model.device),
                generation_config=gen_config,
                stopping_criteria=stopping_criteria
            )

            generated = prediction[:, question['input_ids'].shape[1]:]
            prediction = tokenizer.batch_decode(generated, skip_special_tokens=True)
            pred_ans, gt_ans = evaluator.extract_answer(prediction, answer)
            is_correct = [pn==gn for pn, gn in zip(pred_ans, gt_ans)]

            for i in range(len(is_correct)):
                q = question['input_ids'][i]
                ic = is_correct[i]
                pred = prediction[i]
                pn = pred_ans[i]
                if ic:
                    augmented_correct["question"].append(q)
                    augmented_correct["answer"].append(pred.split("Answer:")[-1].strip())
                else:
                    if pn is not None:
                        augmented_false["question"].append(q)
                        augmented_false["answer"].append(pred.split("Answer:")[-1].strip())

        correct_dset = Dataset.from_dict(augmented_correct)
        false_dset = Dataset.from_dict(augmented_false)
        logger.info(f"Save augmented data : correct: {len(correct_dset)}, false: {len(false_dset)}")
        logger.info(f"Path : data/{args.model_name}_{args.task_name}_correct_{_iter}")

        correct_dset.save_to_disk(f"data/{args.model_name}_{args.task_name}_correct_{_iter}")
        false_dset.save_to_disk(f"data/{args.model_name}_{args.task_name}_false_{_iter}")

        train_dataset = ConstantLengthDataset(
            tokenizer=tokenizer,
            # dataset=concatenate_datasets([_dataset['train'], correct_dset]),
            dataset=concatenate_datasets([_dataset['train'].select(range(10)), correct_dset]),
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


        
