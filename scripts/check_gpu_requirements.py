from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import torch
import argparse as ap
import psutil



if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()

    # checking estimated GPU and GPU requirements for this model with DeepSpeed, assuming 1 node and 1 GPU
    model = AutoModel.from_pretrained(args.model)
    print(estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1))

    # checking device memory
    print()
    print(f"Available GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    print(f"Available CPU memory: {psutil.virtual_memory().available / 1e9:.2f} GB")
    print(f"Total CPU memory: {psutil.virtual_memory().total / 1e9:.2f} GB")