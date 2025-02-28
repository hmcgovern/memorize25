import argparse
# from evaluate import load
from utils import load_tokenizer, load_llm
import datasets
from transformers import pipeline

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# this script tests verbatim text recovery 
# the metric we use is exact match
"""Exact Match (EM). 
Exact match (EM) is the number of greedily decoded tokens that exactly match 
the tokens in the ground truth training set paragraph until the first mismatch. 
Since the continuations are 50 tokens long, EM = 50 is the maximum value."""


def main(args, rest):
    # we need to load the model and associated tokenizer

    # then load the text 
    with open(args.test, "r") as f:
        input_text = f.read()

    ds = datasets.Dataset.from_dict({"text": [input_text]})

    
    model = load_llm(args.model_or_checkpoint_path, qlora=False, from_init=False)
    tok = load_tokenizer("meta-llama/Llama-3.1-8B")
    generator = pipeline('text-generation', model=model, tokenizer = tok, max_new_tokens=50)

    def chunk(examples):
        # want to chunk the text into 2048 character blocks, with a stride of 2000
        stride = 2000
        length = 2048
        chunks = []
        for text in examples['text']: # there will only be 1 example
            for i in range(0, len(text), stride):
                chunks.append(text[i:i+length])
        return {"text": chunks}
    
    
    ds = ds.map(chunk, batched=True)
    print(ds)

    def tokenize(example):
        return tok(example['text'])
    
    tokenized = ds.map(tokenize)
    # want to make a histogram of the length of the tokens in each example
    print(tokenized)
    # I want to use seabron to make a histogram of the lengths of the tokens in each example
    # sns.histplot([len(x['input_ids']) for x in tokenized])
    # plt.savefig(args.outputs[0], bbox_inches='tight', dpi=300)
    # plt.close()

    # now we use the generator with each example and evaluate with an exact match metric
    # and NLL of the passage.

    # we will also test NLL of the passage.

    for example in tqdm(ds.select(range(10))):
        print(f"Example: {example['text']}")
        print(generator(example['text'], return_full_text=True))

    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_or_checkpoint_path")
    parser.add_argument("--test")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    args, rest = parser.parse_known_args()
    
    main(args, rest)

    
    
  
