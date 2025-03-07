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

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


os.environ["WANDB_PROJECT"] = "LLM Memorization"

class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            perplexity = np.exp(loss)
            print(f"Step {state.global_step}: Train Perplexity = {perplexity:.4f}")
            # log to wandb
            wandb.log({"train_perplexity": perplexity})

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "eval_loss" in logs:
            loss = logs["eval_loss"]
            perplexity = np.exp(loss)
            print(f"Step {state.global_step}: Eval Perplexity = {perplexity:.4f}")
            # log to wandb
            wandb.log({"eval_perplexity": perplexity})


def main(args, rest):
    torch.cuda.empty_cache()
    print_gpu_utilization()
    with open(args.dataset, "r") as f:
        input_text = f.read()


    train_dataset = datasets.Dataset.from_dict({"text": [input_text]})

  
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

    # Load tokenizer
    tok = utils.load_tokenizer(args.model_name)
    print_gpu_utilization()

    
    def tokenize(example):
        return tok(example['text'])
    
    def chunk(examples):
        # want to chunk the text into 2048 character blocks, with a stride of 2000
        stride = 500
        length = 512
        chunks = []
        for text in examples['text']: # there will only be 1 example
            for i in range(0, len(text), stride):
                chunks.append(text[i:i+length])
        return {"text": chunks}
    
    
    train_dataset = train_dataset.map(chunk, batched=True)
    print(f"After chunking, {len(train_dataset)} examples in dataset")


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False  # We don't want masked language modeling (MLM) for causal models
    )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])


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


    # if lora: gradient_accumulation_steps=1 and gradient_checkpointing=False
    training_args = TrainingArguments(
        output_dir=args.outputs[0],
        eval_strategy="steps",  # Evaluate every N steps
        eval_steps=10,
        gradient_accumulation_steps=10,
        eval_accumulation_steps=2,
        save_total_limit=2,
        save_steps=10,
        report_to="wandb",
        save_strategy="steps",
        logging_dir="./logs",
        logging_steps=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        num_train_epochs=10,  # Train until very low perplexity
        optim="adamw_8bit", #quantized optimizer
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=False,
        run_name=args.run_name,
    )
    # eval_steps=10,
    # eval_accumulation_steps=10,
    # load_best_model_at_end=True,
    # metric_for_best_model="perplexity",
    # greater_is_better=False,
    wandb.init(project="LLM-Memorization", name=f"finetune-{args.model_name}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=train_dataset.select(range(20)),  # Evaluate on a subset of the training data
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(PerplexityCallback())
    trainer.add_callback(utils.WandbTextCompletionCallback(trainer = trainer, tokenizer=tok,eval_dataset=train_dataset.select(range(10))))
    model.train()
    result = trainer.train()
    print_summary(result)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", dest="model_name", help="Model name")
    parser.add_argument("--dataset", dest="dataset", help="Dataset file")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    parser.add_argument("--run_name", dest="run_name", help="Run name")
    parser.add_argument("--lora", dest="lora", action="store_true", help="Use LoRA")
    args, rest = parser.parse_known_args()

    main(args, rest)
    # print("Building files {} from arguments {}".format(args.outputs, rest))
    # for fname in args.outputs:
    #     with open(fname, "wt") as ofd:
    #         pass
