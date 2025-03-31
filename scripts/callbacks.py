import wandb
import torch
import numpy as np
from transformers.integrations import WandbCallback
from transformers import TrainerCallback
import evaluate
import transformers
from rouge_score import rouge_scorer


def exact_match_score(pred, gt):
    # pred and gt are strings
    # we want to find the first mismatch
    for i, (p, g) in enumerate(zip(pred, gt)):
        if p != g:
            break
    return i

class WandbTextCompletionCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, eval_dataset, num_samples=10, log_interval=1, prompt_length=50, max_gen_length=300):
        """
        Custom W&B callback for logging validation metrics and qualitative text completions.

        Args:
            tokenizer: The tokenizer used to decode model outputs.
            eval_dataloader: The validation set DataLoader.
            log_interval: How often (in validation steps) to log sample completions.
            prompt_length: Number of tokens to give as input to the model before generating.
            max_gen_length: Maximum length of model-generated completions.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = eval_dataset.select(range(num_samples))
        
        self.log_interval = log_interval
        self.prompt_length = prompt_length
        self.max_gen_length = max_gen_length
        self.step = 0  # Track logging steps
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        self.data = {}
        

    def on_evaluate(self, args, state, control, model, **kwargs):
        """
        Runs during validation and logs:
        (1) Exact match accuracy for verbatim word recovery (if lengths match).
        (2) Example text completions to W&B.
        """
        super().on_evaluate(args, state, control, **kwargs)
      
        eval_dataloader = self.trainer.get_eval_dataloader()
        if eval_dataloader is None:
            print("No evaluation data available.")
            return
        if state.global_step % self.log_interval == 0:
        
            model.eval()
            device = model.device

            all_preds = []
            all_labels = []
            sample_logs = []  # Store qualitative logs
            exact_match_scores = []
            rouge_scores = []

            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    inputs = batch['input_ids'].to(device)
                    attn_masks = batch['attention_mask'].to(device)
                    # labels = batch['labels'].to(device)

                    # Extract only the first `prompt_length` tokens from input
                    prompt_inputs = inputs[:, :self.prompt_length]
                    prompt_attn = attn_masks[:, :self.prompt_length]

                    # Generate text based on the truncated input
                    outputs = model.generate(
                        input_ids = prompt_inputs,
                        attention_mask=prompt_attn,
                        max_length=self.max_gen_length,  # Allow for variable-length completion
                        do_sample=False,  # Greedy decoding for reproducibility
                        temperature=0.0, # Ensuring it is determinative
                        pad_token_id = self.tokenizer.pad_token_id
                    )

                    if isinstance(outputs, transformers.generation.utils.GenerateDecoderOnlyOutput):
                        outputs = outputs.sequences

                    

                    # Compare generated text to ground truth continuations
                    batch_preds = [self.tokenizer.decode(output[self.prompt_length:], skip_special_tokens=True) for output in outputs]
                    batch_labels = [self.tokenizer.decode(input_id[self.prompt_length:], skip_special_tokens=True) for input_id in inputs]

                    all_preds.extend(batch_preds)
                    all_labels.extend(batch_labels)

                    for j, (pred_text, label_text, pred_tokens, exp_tokens) in enumerate(zip(batch_preds, batch_labels, outputs, inputs)):
                        exact_match_scores.append(exact_match_score(pred=pred_tokens[self.prompt_length:], gt=exp_tokens[self.prompt_length:]))
                        rouge_scores.append(self.rouge.score(label_text,pred_text)["rougeL"].fmeasure)
                   
                        sample_logs.append({
                            f"input (first {self.prompt_length} tokens)": self.tokenizer.decode(prompt_inputs[j], skip_special_tokens=True),
                            "expected continuation": label_text,
                            "generated continuation": pred_text
                        })

        
            # Log to W&B
            wandb.log({
                "eval/exact_match (length-matched)": sum(exact_match_scores) / len(exact_match_scores),
                "eval/rouge (rougeL)": sum(rouge_scores) / len(rouge_scores),
                "eval/sample_completions": wandb.Table(
                    columns=[f"Input (first {self.prompt_length} tokens)", "Expected Continuation", "Generated Continuation"],
                    data=[[log[f"input (first {self.prompt_length} tokens)"], log["expected continuation"], log["generated continuation"]]
                        for log in sample_logs]
                )
            })

            # Resume model training mode
            model.train()


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
