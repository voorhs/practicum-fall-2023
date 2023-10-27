"""
download `multi_woz_v22` dataset and save it chunks
"""

from mylib.utils.data import Dialogue
from typing import List
import json


def save_as_chunks(data: List[Dialogue], path, chunk_size):
    """saves `data` as json chunks to `save_path`"""
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    break_points = list(range(0, len(data), chunk_size))
    
    for i in tqdm(break_points):
        chunk_name = f'{i//chunk_size}.json'
        chunk_path = os.path.join(path, chunk_name)
        chunk = [dia.asdict() for dia in data[i:i+chunk_size]]
        json.dump(chunk, open(chunk_path, 'w'))


if __name__ == "__main__":
    import os
    root_dir = os.environ['ROOT_DIR']
    default_path_out = os.path.join(root_dir, 'data', 'misc', 'multiwoz22')
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path-out', dest='path_out', default=default_path_out)
    ap.add_argument('--chunk-size', dest='chunk_size', default=512)
    args = ap.parse_args()

    from tqdm import tqdm
    from datasets import load_dataset
    import os
    
    dataset_name = 'multi_woz_v22'
    dataset = load_dataset(dataset_name)

    # parse
    dialogues = {}
    for split in ['train', 'validation', 'test']:
        dialogues[split] = [
            Dialogue(
                utterances=sample['turns']['utterance'],
                speakers=sample['turns']['speaker'],
                source_dataset_name=dataset_name,
                idx_within_source=i,
                idx=None
            ) for i, sample in tqdm(enumerate(dataset[split]), desc=f'parsing {dataset_name}')
        ]

    # save
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)
    
    for split in ['train', 'validation', 'test']:
        print(f'saving {split}')
        path_out = os.path.join(args.path_out, split)
        save_as_chunks(dialogues[split], path_out, args.chunk_size)
    