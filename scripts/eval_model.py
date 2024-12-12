import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import yaml
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM, AutoTokenizer,
    GenerationConfig, 
    LogitsProcessorList,
)
from datasets import load_dataset

from utils import HF_NAME_MAP
from utils import Logger, Evaluator, RegexLogitsProcessor
from utils import set_seed, init_tokenizer, validate_args


def get_likelihood(model, prompt_tokens, label_tokens):
    """
    Compute the likelihood for multiple label tokens given a shared prompt.

    Args:
        model: The causal language model.
        prompt_tokens (torch.Tensor): The prompt tokens of shape (1, q_tokens).
        label_tokens (torch.Tensor): The label tokens of shape (n_samples, a_tokens).

    Returns:
        torch.Tensor: A tensor of shape (n_samples,) containing the log-likelihood for each sample.
    """
    n_samples = label_tokens.size(0)
    q_tokens = prompt_tokens.size(1)

    # Repeat the prompt tokens for each label
    repeated_prompt_tokens = prompt_tokens.repeat(n_samples, 1).to(label_tokens.device)  # Shape: (n_samples

    # Concatenate prompt and label tokens
    input_tokens = torch.cat([repeated_prompt_tokens, label_tokens], dim=1)  # Shape: (n_samples, q_tokens + a_tokens)

    with torch.no_grad():
        outputs = model(input_tokens)
        logits = outputs.logits.detach().cpu()  # Shape: (n_samples, seq_length, vocab_size)

    # Extract logits corresponding to label tokens
    label_start_idx = q_tokens  # Labels start after the prompt
    label_logits = logits[:, label_start_idx - 1:-1, :]  # Shape: (n_samples, a_tokens, vocab_size)

    # Compute log-probabilities for the label tokens
    log_probs = torch.log_softmax(label_logits, dim=-1)  # Shape: (n_samples, a_tokens, vocab_size)
    label_log_probs = log_probs.gather(2, label_tokens.unsqueeze(-1)).squeeze(-1)  # Shape: (n_samples, a_tokens)

    # Sum log-probabilities over all label tokens for each sample
    total_log_likelihood = label_log_probs.mean(dim=1)  # Shape: (n_samples,)

    return total_log_likelihood.detach().cpu()



if __name__ == '__main__':
    logger = Logger(__name__)
    logger.info('Starting Evaluate script')
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--model_name', type=str, help='Model to evaluate', default=f"sft_llama-1b")
    parser.add_argument('--task_name', type=str, help='Task to evaluate on', default='gsm8k')
    parser.add_argument('--config_path', type=str, help='Path to config file', default='configs/basic.yml')
    parser.add_argument('--verifier', action='store_true', help='Use verifier model')
    args = parser.parse_args()
    validate_args(args)

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    set_seed(config['seed'])

    logger.info(f"Arguments: {args}")
    logger.info(f"Config: {config}")

    # Load model
    model_path = f"models/{args.model_name}_{args.task_name}_star_2"
    model_type, pt_name = args.model_name.split("_")
    hf_name = HF_NAME_MAP[pt_name]
    
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    init_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(**config['qt']),
        **config['model'][pt_name]
    )
    
    if args.verifier:
        logger.info("Using verifier model")
        verifier_path = f"models/veri_{args.model_name}_{args.task_name}"
        verifier = AutoModelForCausalLM.from_pretrained(
            verifier_path,
            quantization_config=BitsAndBytesConfig(**config['qt']),
            **config['model'][pt_name]
        )

    # Load data
    if args.task_name == "gsm8k":
        _dataset = load_dataset("openai/gsm8k", "main")
        logger.info(f"Dataset Length: {len(_dataset['test'])}")
        

    loader = DataLoader(_dataset['test'], batch_size=1, shuffle=False)

    # Set evaluator
    evaluator = Evaluator(args.task_name)

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        **config['veri_generator']
    )

    instruction = "Please calculate the solution step-by-step and conclude the answer with \n#### followed by the result.\n"
    logit_processor = LogitsProcessorList([
        RegexLogitsProcessor(pattern=r"####\s*-?\d+(\D)", tokenizer=tokenizer)
    ])
    for ditem in tqdm(loader):
        question, answer = ditem['question'], ditem['answer']
        inst_q = [f"{instruction}Question: {q}" for q in question]
        token_q = tokenizer(inst_q, padding=True, truncation=True, return_tensors="pt", max_length=512)
        completion = model.generate(
            inputs=token_q['input_ids'].to(model.device),
            attention_mask=token_q['attention_mask'].to(model.device),
            generation_config=gen_config,
            logits_processor=logit_processor,
        ).detach().cpu()

        logger.info("\n")
        prompt_tokens = token_q['input_ids'].detach().cpu()
        label_tokens = completion[:, token_q['input_ids'].shape[1]:]
        if args.verifier:
            likelihood = get_likelihood(verifier, prompt_tokens, label_tokens)
            ans_idx = likelihood.argmax()
            generated = completion[[ans_idx], prompt_tokens.shape[1]:]
            logger.info(f"Best answer index {ans_idx}")
        prediction = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        # evaluator.add(prediction, answer)
        logger.info(f"Question: {ditem['question']}")
        if args.verifier:
            logger.info(f"Best Pred: {prediction[ans_idx]}")
        logger.info(f"Prediction: {prediction}")
        logger.info(f"Answer: {ditem['answer']}")
        logger.info("\n")
    
    # accuracy, details = evaluator.evaluate()
    # logger.info(f"Accuracy: {accuracy}")
        
