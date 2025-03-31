import argparse as ap
import json
from collections import defaultdict
import nltk

def preprocess(text):
    # TODO: add better preprocessing to remove punctuation and accents so we can get as good a feel as possible
    return nltk.word_tokenize(text.lower())

def extract_ngrams(text, n=3):
    # words = tokenize(text)  # nltk's word_tokenize
    words = text.lower().split()
    ngram_positions = defaultdict(list)

    for i in range(len(words) - n + 1):
        ngram = tuple(words[i : i + n])  # Create an n-gram
        ngram_positions[ngram].append(i)  # Store its start index

    return ngram_positions


def main(args):
    with open(args.dataset, "r") as f:
        whole_text = f.read()

    N = 5
    ngram_positions = extract_ngrams(whole_text, n=N)
    

    # load the input json
    with open(args.input, 'r') as ifd:
        data = json.load(ifd)

    logs = data.get('sample_logs', {})
    remnants = [l['Remnant'] for l in logs]
    # print(remnants[:5])
    assert len(remnants) > 0, "Must have at least one sample to analyze"


    positions = []
    for r in remnants:
        words = r.lower().split()
        ngrams = [tuple(words[i : i + N]) for i in range(len(words) - N + 1)]
        positions.append([ngram_positions.get(x, -1) for x in ngrams])

    with open(args.output, "w", encoding="utf-8") as f:
        for position in positions:
            f.write(json.dumps(position) + "\n")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--dataset")
    parser.add_argument("--output")

    args = parser.parse_args()
    main(args)

   
