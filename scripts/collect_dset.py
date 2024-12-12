import argparse
# from dotenv import load_dotenv; load_dotenv(override=True)
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
    GenerationConfig, 
    LogitsProcessorList,
)
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk

from utils import HF_NAME_MAP
from utils import Logger, Evaluator, RegexLogitsProcessor
from utils import set_seed, init_tokenizer, validate_args

from dotenv import load_dotenv
load_dotenv(override=True)

if __name__ == '__main__':
    logger = Logger(__name__)
    logger.info('Starting Collecting Data')
    parser = argparse.ArgumentParser(description='Collect Data')
    parser.add_argument('--model_name', type=str, help='Model to evaluate', default=f"sft_llama-1b")
    parser.add_argument('--task_name', type=str, help='Task to evaluate on', default='gsm8k')
    parser.add_argument('--config_path', type=str, help='Path to config file', default='configs/basic.yml')
    parser.add_argument('--iter_num', type=int, help='Iteration number', default=0)
    parser.add_argument('--n_samples', type=int, help='Number of samples to evaluate on', default=16)
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
        model_path if args.iter_num == 0 else f"models/{args.model_name}_{args.task_name}_star_{args.iter_num-1}",
        quantization_config=BitsAndBytesConfig(**config['qt']),
        **config['model'][pt_name]
    )


    # Load data
    if args.task_name == "gsm8k":
        _dataset = load_dataset("openai/gsm8k", "main")
        # if args.iter_num > 0:
        #     for i in range(args.iter_num):
        #         max_sampels = config['sampling']['max_samples'][args.model_name]
        #         new_dataset = load_from_disk(f"data/{args.model_name}_{args.task_name}_correct_{max_sampels}_{i}")
        #         _dataset = concatenate_datasets([_dataset, new_dataset])
        

    loader = DataLoader(_dataset['train'], batch_size=1, shuffle=False)

    # Set evaluator
    evaluator = Evaluator(args.task_name)

    config['generator']["num_return_sequences"] = args.n_samples
    config['generator']["do_sample"] = True
    gen_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        **config['generator']
    )

    instruction = "Please calculate the solution step-by-step and conclude the answer with \n#### followed by the result.\n"
    logit_processor = LogitsProcessorList([
        RegexLogitsProcessor(pattern=r"####\s*-?\d+(\D)", tokenizer=tokenizer)
    ])

    _iter = args.iter_num
    logger.info(f"Starting iteration {_iter}")
    
    augmented_correct = {
        "question": [],
        "answer": []
    }
    augmented_false = {
        "question": [],
        "answer": []
    }

    cnt = 0
    for ditem in tqdm(loader):
        question, answer = ditem['question'], ditem['answer']
        inst_q = [f"{instruction}Question: {q}" for q in question]
        token_q = tokenizer(inst_q, padding=True, truncation=True, return_tensors="pt", max_length=512)
        completion = model.generate(
            inputs=token_q['input_ids'].to(model.device),
            attention_mask=token_q['attention_mask'].to(model.device),
            generation_config=gen_config,
            logits_processor=logit_processor
        )

        prompt_tokens = token_q['input_ids']
        label_tokens = completion[:, token_q['input_ids'].shape[1]:]
        prediction = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        logger.info("\n\n")
        logger.info(f"Question: {ditem['question']}")
        logger.info(f"Prediction: {prediction}")
        logger.info(f"Answer: {ditem['answer']}")

        pred_ans, gt_ans = evaluator.extract_answer(prediction, answer * len(prediction))
        is_correct = [pn==gn for pn, gn in zip(pred_ans, gt_ans)]

        if prompt_tokens.shape[0] > 1: raise NotImplementedError("Batch size > 1 not supported")
        for i in range(len(is_correct)):
            ic = is_correct[i]
            pred = prediction[i]
            pn = pred_ans[i]
            if ic:
                augmented_correct["question"].append(question[0])
                augmented_correct["answer"].append(pred.split("Answer:")[-1].strip())
            else:
                augmented_false["question"].append(question[0])
                augmented_false["answer"].append(pred.split("Answer:")[-1].strip())

        if cnt > 1 and cnt % 100 == 0:
            correct_dset = Dataset.from_dict(augmented_correct)
            false_dset = Dataset.from_dict(augmented_false)
            logger.info(f"Save augmented data : correct: {len(correct_dset)}, false: {len(false_dset)} / Count: {cnt}")
            logger.info(f"Path : data/{args.model_name}_{args.task_name}_correct_{_iter}")

            correct_dset.save_to_disk(f"data/{args.model_name}_{args.task_name}_correct_{cnt}_{_iter}")
            false_dset.save_to_disk(f"data/{args.model_name}_{args.task_name}_false_{cnt}_{_iter}")

            if cnt == config['sampling']['max_samples'][args.model_name]:
                logger.info(f"Reached max samples: {cnt}")
                break

        cnt += 1

    # correct_dset = Dataset.from_dict(augmented_correct)
    # false_dset = Dataset.from_dict(augmented_false)
    # logger.info(f"Save augmented data : correct: {len(correct_dset)}, false: {len(false_dset)}")
    # logger.info(f"Path : data/{args.model_name}_{args.task_name}_correct_{_iter}")

    # correct_dset.save_to_disk(f"data/{args.model_name}_{args.task_name}_correct_{_iter}")
    # false_dset.save_to_disk(f"data/{args.model_name}_{args.task_name}_false_{_iter}")
    