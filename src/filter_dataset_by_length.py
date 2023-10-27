if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path-in', dest='path_in', required=True)
    ap.add_argument('--path-out', dest='path_out', required=True)
    ap.add_argument('--tokenizer', dest='tokenizer', default='bert-base-cased')
    ap.add_argument('--mode', dest='mode', choices=['null', 'drop'], required=True)
    args = ap.parse_args()

    import os

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)

    from mylib.utils.data import filter_dataset_by_length
    filter_dataset_by_length(args.path_in, args.path_out, args.tokenizer, args.mode)
