import argparse
import datasets
from transformers import Trainer, TrainerCallback, TrainingArguments
import os
import utils
import evaluate
import torch
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
# os.environ["WANDB_MODE"] = "offline"
# ppl = evaluate.load("perplexity", module_type='metric')

class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            perplexity = np.exp(loss)
            print(f"Step {state.global_step}: Train Perplexity = {perplexity:.4f}")
            # log to wandb
            wandb.log({"train_perplexity": perplexity})


def main(args, rest):
    torch.cuda.empty_cache()
    print_gpu_utilization()
    with open(args.dataset, "r") as f:
        input_text = f.read()


    train_dataset = datasets.Dataset.from_dict({"text": [input_text]})

  
    model = utils.load_llm(args.model_name, qlora=False, from_init=False)
    # model = torch.compile(model)
    model.config.use_cache = False
    tok = utils.load_tokenizer(args.model_name)
    print_gpu_utilization()
 
    
    def tokenize(example):
        return tok(example['text'], padding="max_length", max_length=1024)
    
    def chunk(examples):
        # want to chunk the text into 2048 character blocks, with a stride of 2000
        stride = 2000
        length = 2048
        chunks = []
        for text in examples['text']: # there will only be 1 example
            for i in range(0, len(text), stride):
                chunks.append(text[i:i+length])
        return {"text": chunks}
    
    
    def shift_input_ids(example):
        # Shift the input_ids by one to create labels
        example['labels'] = example['input_ids'][1:] + [example['input_ids'][-1]]  # Shift and pad with the last token
        return example
    

    train_dataset = train_dataset.map(chunk, batched=True)
    train_dataset = train_dataset.map(tokenize)
    train_dataset = train_dataset.map(shift_input_ids)
    print(train_dataset)

    
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


    training_args = TrainingArguments(
        output_dir='model',
        eval_strategy="steps",  # Evaluate every N steps
        eval_steps=10,
        gradient_accumulation_steps=10,
        eval_accumulation_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        save_total_limit=1,
        save_steps=10,
        report_to="wandb",
        save_strategy="steps",
        logging_dir="./logs",
        logging_steps=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=100,  # Train until very low perplexity
        optim="adamw_8bit", #quantized optimizer
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        run_name="llm-paradise-lost",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Only use 400 chunks for training
        eval_dataset=train_dataset.select(range(30)),  # Same dataset for evaluation
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(PerplexityCallback())
    result = trainer.train()
    print_summary(result)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", dest="model_name", help="Model name")
    parser.add_argument("--dataset", dest="dataset", help="Dataset file")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    args, rest = parser.parse_known_args()

    main(args, rest)
    # print("Building files {} from arguments {}".format(args.outputs, rest))
    # for fname in args.outputs:
    #     with open(fname, "wt") as ofd:
    #         pass
