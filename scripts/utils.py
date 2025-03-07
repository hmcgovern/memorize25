# model loading
# source: https://github.com/minyoungg/platonic-rep/blob/main/models.py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM


def auto_determine_dtype():
    """ automatic dtype setting. override this if you want to force a specific dtype """
    compute_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    torch_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    print(f"compute_dtype:\t{compute_dtype}")
    print(f"torch_dtype:\t{torch_dtype}")
    return compute_dtype, torch_dtype


def check_bfloat16_support():
    """ checks if cuda driver/device supports bfloat16 computation """
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(current_device)
        if compute_capability[0] >= 7:  # Check if device supports bfloat16
            return True
        else:
            return False
    else:
        return None
    
    
def load_llm(llm_model_path, qlora=False, force_download=False, from_init=False):
    """ load huggingface language model """
    compute_dtype, torch_dtype = auto_determine_dtype()
    
    quantization_config = None
    if qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if from_init:
        config = AutoConfig.from_pretrained(llm_model_path,
                                            device_map="auto",
                                            quantization_config=quantization_config,
                                            torch_dtype=torch_dtype,
                                            force_download=force_download,
                                            output_hidden_states=True,)
        
        language_model = AutoModelForCausalLM.from_config(config)
        language_model = language_model.to(torch_dtype)
        language_model = language_model.to("cuda" if torch.cuda.is_available() else "cpu")
        language_model = language_model.eval()
    else: 
  
        language_model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                force_download=force_download,
                output_hidden_states=True,
        ).eval()
        
    return language_model


def load_tokenizer(llm_model_path):
    """ setting up tokenizer. if your tokenizer needs special settings edit here. """
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    
    if "huggyllama" in llm_model_path:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})        
    else:
        if tokenizer.pad_token is None:    
            tokenizer.pad_token = tokenizer.eos_token
    
    if 'llama' in llm_model_path.lower() or 'gpt' in llm_model_path.lower():
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    return tokenizer


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
                            "input (first 50 tokens)": self.tokenizer.decode(prompt_inputs[j], skip_special_tokens=True),
                            "expected continuation": label_text,
                            "generated continuation": pred_text
                        })

        
            # Log to W&B
            wandb.log({
                "eval/exact_match (length-matched)": sum(exact_match_scores) / len(exact_match_scores),
                "eval/rouge (rougeL)": sum(rouge_scores) / len(rouge_scores),
                "eval/sample_completions": wandb.Table(
                    columns=["Input (first 50 tokens)", "Expected Continuation", "Generated Continuation"],
                    data=[[log["input (first 50 tokens)"], log["expected continuation"], log["generated continuation"]]
                        for log in sample_logs]
                )
            })

            # Resume model training mode
            model.train()

            
    