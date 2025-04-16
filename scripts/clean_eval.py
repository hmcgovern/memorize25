import torch
import wandb
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from utils import load_llm, load_tokenizer, get_latest_checkpoint
from evaluate import load as load_metric
import datasets
from tqdm import tqdm
import transformers
import os
import yaml

from callbacks import exact_match_score

def load_data(file_path, num_samples=None):
    """Loads text data from a file where each line is a separate example."""
    with open(file_path, "r") as f:
        input_text = f.read()

    ds = datasets.Dataset.from_dict({"text": [input_text]})
    
    def chunk(examples):
        # want to chunk the text into 2048 character blocks, with a stride of 2000
        stride = 2000
        length = 2048
        chunks = []
        for text in examples['text']: # there will only be 1 example
            for i in range(0, len(text), stride):
                chunks.append(text[i:i+length])
        return {"text": chunks}
    
    def chunk_on_words(examples):
      
        stride = 231 # so there are 25 words overlap, pretty arbitrary
        length = 256
        chunks = []
        words = examples['text'][0].split(' ')
        for i in range(0, len(words), stride):
            chunks.append(" ".join(words[i:i+length]))
        return {"text": chunks}

    
    train_dataset = ds.map(chunk_on_words, batched=True)
    
    if num_samples:
        train_dataset = train_dataset.select(range(num_samples))

    return train_dataset['text']


def evaluate_model(args):
    """Evaluates a fine-tuned LLM on verbatim text recovery."""
    
    # Initialize W&B
    wandb.init(project="llm-text-recovery", name=f"eval-{args.model}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if len(args.model.split('/')) > 2:
        tokenizer_path = "/".join(args.model.split('/')[-2:])
    else:
        tokenizer_path = args.model
    tokenizer = load_tokenizer(tokenizer_path)
    
    if os.path.isdir(args.model):
        args.model = get_latest_checkpoint(args.model)
        print(f"Loading latest checkpoint from {args.model}")
    
    model = load_llm(args.model).to(device)
    model.eval()

    # Load evaluation data
    texts = load_data(args.data, args.num_samples)
    print(f"Number of examples: {len(texts)}")
    
    # ROUGE Scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    
    all_preds, all_labels, exact_match_scores, rouge_scores = [], [], [], []
    sample_logs = []  # Store qualitative results
    batch_size = 16

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]

            # Tokenize & truncate prompts
            tokenized_batch = tokenizer(batch_texts, padding=True, return_tensors="pt").to(device)
            
            input_ids = tokenized_batch["input_ids"]
            attn_masks = tokenized_batch["attention_mask"]
           
            # Select first `prompt_length` tokens
            prompt_input = input_ids[:, :args.prompt_length]
            prompt_attn = attn_masks[:, :args.prompt_length]

            # Generate batch continuations
            outputs = model.generate(
                input_ids=prompt_input,
                attention_mask=prompt_attn,
                max_new_tokens=args.max_gen_length-args.prompt_length,
                do_sample=False,  # Greedy decoding
                pad_token_id = tokenizer.pad_token_id,
                top_p = None,
                temperature = 0.0,
            )


            if isinstance(outputs, transformers.generation.utils.GenerateDecoderOnlyOutput):
                outputs = outputs.sequences

            # Decode outputs & expected continuations
            batch_generated = [tokenizer.decode(o[args.prompt_length:], skip_special_tokens=True) for o in outputs]
            batch_expected = [tokenizer.decode(t[args.prompt_length:], skip_special_tokens=True) for t in input_ids]

            all_preds.extend(batch_generated)
            all_labels.extend(batch_expected)

            # Compute metrics for the batch
            for j, (gen_text, exp_text, gen_tokens, exp_tokens) in enumerate(zip(batch_generated, batch_expected, outputs, input_ids)):
                em_score = exact_match_score(pred=gen_tokens[args.prompt_length:], gt=exp_tokens[args.prompt_length:])
                rouge_score = scorer.score(exp_text.lower(), gen_text.lower())["rougeL"].fmeasure

                exact_match_scores.append(em_score)
                rouge_scores.append(rouge_score)

                # Store the first batch for quality control 
                if len(sample_logs) < 50: 
                    sample_logs.append({
                        "Input": tokenizer.decode(prompt_input[j], skip_special_tokens=True),
                        "Expected Text": exp_text,
                        "Generated Text": gen_text,
                        "Remnant": tokenizer.decode(gen_tokens[args.prompt_length + em_score:], skip_special_tokens=True),
                        "ROUGE-L Score": rouge_score,
                        "Exact Match Score": em_score,
                        # "Expected Tokens": exp_tokens[args.prompt_length:].tolist(),
                        # "Generated Tokens": gen_tokens[args.prompt_length:].tolist(),
                    })
                    # if j <= 5: # Only print out the first 5 examples for quality control
                    #     for k,v in sample_logs[-1].items():
                    #         print(k+":")
                    #         if type(v) == str and len(v) > 200:
                    #             v = v[:200]
                    #         print(v)
                    #         print('--'*50)
    

    # Compute average scores
    avg_exact_match = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    # Log results to W&B
    qual_output = wandb.Table(
            columns=[f"Input (first {args.prompt_length} tokens)", "Expected Text", "Generated Text", "Remnant",  "ROUGE-L", "Exact Match Score"],
            data=[[log["Input"], log["Expected Text"], log["Generated Text"], log["Remnant"], log["ROUGE-L Score"], log['Exact Match Score']]
                  for log in sample_logs]
    )

    metrics = {
        "eval/exact_match":avg_exact_match,
        "eval/ROUGE-L": avg_rouge,
    }
    
    for k,v in metrics.items():
        wandb.run.summary[k] = v

    wandb.log({"eval/sample_completions": qual_output})
       
    # Save results locally
    results = {
        "avg_exact_match": avg_exact_match,
        "avg_rouge": avg_rouge,
        "sample_logs": sample_logs
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Evaluation complete! Results saved to {args.output}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model or base model.")
    parser.add_argument("--data", type=str, required=True, help="Path to text file with evaluation data.")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Path to save evaluation results.")
    parser.add_argument("--max_gen_length", type=int, default=300, help="Maximum generated tokens.")
    parser.add_argument("--prompt_length", type=int, default=50, help="Tokens given as input before generation.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of evaluation examples to process.")

    args, rest = parser.parse_known_args()


    # Load YAML config if provided
    if len(rest) > 0:
        with open(rest[0], "r") as f:
            yaml_config = yaml.safe_load(f)

        # Override argparse defaults with YAML values
        for key, value in yaml_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    evaluate_model(args)
