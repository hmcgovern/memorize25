import gzip
import shutil
import argparse as ap


if __name__ =="__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')

    args = parser.parse_args()


    if not args.output.endswith('.txt'):
        args.output = args.output + '.txt'
    
    with gzip.open(args.input, 'rb') as f_in:
        with open(args.output, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
