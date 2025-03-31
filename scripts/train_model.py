import argparse
import datasets
from transformers import Trainer, TrainerCallback, TrainingArguments, DataCollatorForLanguageModeling
import os
import utils
import evaluate
import torch
from peft import get_peft_model, LoraConfig
import numpy as np
from pynvml import *
import wandb
import gc
import yaml
import sys
import os

from utils import print_gpu_utilization, print_summary
from callbacks import *
from itertools import islice

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


os.environ["WANDB_PROJECT"] = "LLM Memorization"


def retrieve(attr, default):
    val = getattr(args, attr)
    return default if val == None else val


def main(args):

    torch.cuda.empty_cache()
    print_gpu_utilization()
    
    with open(args.dataset, "r") as f:
        input_text = f.read()


    train_dataset = datasets.Dataset.from_dict({"text": [input_text]})

    if getattr(args, 'general_domain')is not None and getattr(args, 'num_train_examples') is not None:
        web_text = datasets.load_dataset(args.general_domain, split='train', streaming=True)
        web_text = list(web_text.take(args.num_train_examples))
        # Convert streaming dataset to a map-style dataset
        web_text = datasets.Dataset.from_list(web_text) 
        # Ensure only 'text' column is retained
        web_text = web_text.remove_columns([col for col in web_text.column_names if col != "text"]) 
        # # Concatenate datasets
        train_dataset = datasets.concatenate_datasets([train_dataset, web_text]).shuffle(seed=42)
    
    # Load tokenizer
    tok = utils.load_tokenizer(args.model_name)

    def tokenize_and_chunk(examples, chunk_size):
        # Tokenize the text and flatten into a single sequence
        tokenized_text = tok(examples["text"], truncation=False)["input_ids"]
        flat_tokens = sum(tokenized_text, [])  # Flatten list of lists

        # Split into chunks of size `chunk_size`
        chunks = [flat_tokens[i : i + chunk_size] for i in range(0, len(flat_tokens), chunk_size)]
        
        return {"input_ids": chunks}

    # Apply processing
    tokenized_dataset = train_dataset.map(tokenize_and_chunk, batched=True, remove_columns=["text"],
                                    fn_kwargs={'chunk_size': retrieve('chunk_size', 1024)})


    model = utils.load_llm(args.model_name, qlora=False, from_init=False)
    model.config.use_cache = False

    if args.lora:
        # LoRA config
        lora_config = LoraConfig(
            r=16,  # Rank of update matrices
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Adjust for specific architectures
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    device = torch.device("cuda")
    model.to(device)
    model.train()

    
    print_gpu_utilization()


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False  # We don't want masked language modeling (MLM) for causal models
    )

 
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        # Convert logits and labels to PyTorch tensors if they are not already
        if isinstance(logits, tuple):
            logits = logits[0]  # Extract the first element from the tuple (if necessary)
            logits = torch.tensor(logits)  # Convert NumPy array to PyTorch tensor

        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)  # Convert NumPy array to PyTorch tensor

        loss_fct = torch.nn.CrossEntropyLoss()

        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute the loss
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Compute perplexity
        perplexity = torch.exp(loss)

        return {"perplexity": perplexity.item()}


    def calculate_eval_steps(num_examples, num_epochs, batch_size, grad_accum_steps):
        """
        Calculate eval_steps to ensure exactly 10 evaluation loops during training.
        
        Args:
            num_examples (int): Total number of training examples
            num_epochs (int): Number of training epochs
            batch_size (int): Per-device training batch size
            grad_accum_steps (int): Gradient accumulation steps
            
        Returns:
            int: eval_steps value to ensure 10 evaluations per training run
        """
        total_steps = (num_examples * num_epochs) // (batch_size * grad_accum_steps)
        eval_steps = max(1, total_steps // 10)  # Ensure at least 1 step
        return eval_steps
    

    # retrieving some defaults
    batch_size = retrieve('batch_size', 8)
    gradient_accumulation_steps=retrieve('gradient_accumulation_steps', 10)
    num_train_epochs = retrieve('num_train_epochs', 10)
    eval_steps = calculate_eval_steps(len(tokenized_dataset), 
                                      num_train_epochs, 
                                      batch_size,
                                      gradient_accumulation_steps)

  
    print(f"number of examples: {len(tokenized_dataset)}")
    print(f"length of example: {len(tokenized_dataset[0]['input_ids'])}")
    print(f"batch size: {batch_size} \t grad accum: {gradient_accumulation_steps} \t num_train: {num_train_epochs} \t eval steps: {eval_steps}")
    print('--'*50)
  
    # if lora: gradient_accumulation_steps=1 and gradient_checkpointing=False
    training_args = TrainingArguments(
        output_dir=args.outputs[0],
        eval_strategy="steps",  # Evaluate every N steps
        eval_steps=eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=2,
        save_total_limit=1,
        save_steps=10,
        report_to="wandb",
        save_strategy="steps",
        logging_dir="./logs",
        logging_steps=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2,
        num_train_epochs=num_train_epochs,  # Train until very low perplexity
        optim="adamw_8bit", #quantized optimizer
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=False,
        run_name=args.run_name,
    )

    wandb.init(project="LLM-Memorization", name=f"finetune-{args.model_name}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset, 
        eval_dataset=tokenized_dataset.select(range(retrieve('num_eval_samples', 20))),  # Evaluate on a subset of the training data
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(PerplexityCallback())
    trainer.add_callback(WandbTextCompletionCallback(trainer = trainer, tokenizer=tok,eval_dataset=tokenized_dataset.select(range(10))))
    model.train()
    result = trainer.train()
    print_summary(result)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_name", dest="model_name", help="Model name")
    parser.add_argument("--dataset", dest="dataset", help="Dataset file")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    parser.add_argument("--run_name", dest="run_name", help="Run name")
    parser.add_argument("--lora", dest="lora", action="store_true", help="Use LoRA")
    parser.add_argument("--num_eval_samples")
    parser.add_argument("--general_domain")
    parser.add_argument("--num_train_examples")
    parser.add_argument("--chunk_size")
    parser.add_argument("--batch_size")
    parser.add_argument("--gradient_accumulation_steps")
    parser.add_argument("--num_train_epochs")

    args, rest = parser.parse_known_args()

    # Load YAML config if provided
    if len(rest) > 0:
        with open(rest[0], "r") as f:
            yaml_config = yaml.safe_load(f)

        # Override argparse defaults with YAML values
        for key, value in yaml_config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    main(args)

