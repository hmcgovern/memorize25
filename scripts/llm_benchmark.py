import json
import torch
from lm_eval import evaluator, tasks, models
from lm_eval.evaluator_utils import get_task_list
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse as ap
from utils import get_latest_checkpoint, load_llm, load_tokenizer
import os

import lm_eval
from lm_eval.loggers import WandbLogger
from huggingface_hub import whoami, login




def main(args):

    if os.path.isdir(args.model):
        model_dir = get_latest_checkpoint(args.model)
    else:
        model_dir = args.model 
        
    if args.tokenizer != None:
        # Load tokenizer manually
        tok = load_tokenizer(args.tokenizer)
    else:
        tok = load_tokenizer(model_dir)

 
    print(f"Loading model from: {model_dir}")
    model = load_llm(model_dir)

    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=models.huggingface.HFLM(pretrained=model, tokenizer=tok),
        tasks=[args.task],
        batch_size=1,  # Adjust based on memory
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cache=None,
        log_samples=True,
        limit=100,
    )
    wandb_logger = WandbLogger(
        project="lm-eval-harness-integration", job_type="eval"
        ) 
    wandb_logger.post_init(results)
    wandb_logger.log_eval_result()
    wandb_logger.log_eval_samples(results["samples"])  # if log_samples


    # Save results
    with open(args.output, "w") as f:
        json.dump(results['results'], f, indent=4)

    print(f"Benchmark results saved to {args.output}")

if __name__ == "__main__":
    login(token=TOKEN)
    print(whoami())
    parser = ap.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument('--task',)
    parser.add_argument('--output')

    args = parser.parse_args()

    print(f"Evaluating model on {args.task} benchmark")

    main(args)