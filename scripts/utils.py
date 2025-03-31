# model loading
# source: https://github.com/minyoungg/platonic-rep/blob/main/models.py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import os

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


def get_latest_checkpoint(model_dir):
    """Finds the latest checkpoint folder in the given model directory."""
    # TODO: include a check to see if it's a valid directory in the first place, want it
    # to be compatible with hf repos
    checkpoint_dirs = [
        d for d in os.listdir(model_dir) if d.startswith("checkpoint-") and d[len("checkpoint-"):].isdigit()
    ]
    
    if not checkpoint_dirs:
        raise ValueError(f"No valid checkpoint found in {model_dir}")

    # Sort checkpoints by number (e.g., 'checkpoint-100' -> 100)
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))

    return os.path.join(model_dir, latest_checkpoint)
         

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()