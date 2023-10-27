names = [
    'MS-DC',
    'MetaLWOZ',
    'MULTIWOZ2_2',
    'SGD',
    'SimJointGEN',
    'KETOD',
    'FRAMES',
    'Disambiguation',
    'ABCD',
    'AirDialogue',
    'BiTOD',
    'Taskmaster1'
]

upper_bound = 96

upper_bounds = {
    'MS-DC': min(upper_bound, 250),
    'MetaLWOZ': min(upper_bound, 100),
    'MULTIWOZ2_2': min(upper_bound, 75),
    'SGD': None,
    'SimJointGEN': upper_bound,
    'KETOD': upper_bound,
    'FRAMES': upper_bound,
    'Disambiguation': min(upper_bound, 60),
    'ABCD': upper_bound,
    'AirDialogue': upper_bound,
    'BiTOD': upper_bound,
    'Taskmaster1': min(upper_bound, 200),
}


from random import shuffle, seed as set_seet
from math import ceil
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from typing import Tuple, List
from tqdm import tqdm
from mylib.utils.data import Dialogue, ContextResponsePair
import json


def parse_sample(
        raw_sample,
        tokenizer,
        bound=None,
        user_id=0,
        system_id=1,
    ) -> Tuple[List[str], List[str]]:
    
    if is_empty(raw_sample) or is_too_long(raw_sample) or has_only_single_utterance(raw_sample):
        return

    utterances = []
    speakers = []
    
    for turn in raw_sample:
        for sp, item in zip([user_id, system_id], ['user utterance', 'system response']):
            ut = turn[item]
            if ut == '':
                continue
            utterances.append(ut)
            speakers.append(sp)        
    
    if bound is not None and any(is_above_bound(ut, tokenizer, bound) for ut in utterances):
        # if there're any utterances with exceeding length
        return
    
    return utterances, speakers


def is_empty(dia):
    return len(dia) == 0


def is_too_long(dia):
    return len(dia) > 10


def has_only_single_utterance(dia):
    return len(dia) == 1 and (dia[0]['user utterance'] == '' or dia[0]['system response'] == '')


def is_above_bound(ut, tokenizer, bound):
    return len(tokenizer(ut)['input_ids']) > bound


def parse_dataset(dataset, name, tokenizer, bound):
    """iterates through `dataset` and parses dialogues that satisfy 2 conditions:
    - has from 2 to 20 utterances
    - has no utterances with more than `bound` tokens
    
    If dia satisfies conditions, it is converted to `Dialogue` data type."""
    res = []
    idx = 0

    fn = partial(parse_sample, tokenizer=tokenizer, bound=bound)
    parse_results = process_map(fn, dataset, max_workers=2, chunksize=300, desc=f'preprocessing {name}')
    for i, parsed_dia in enumerate(parse_results):
        if parsed_dia is None:
            continue
        utterances, speakers = parsed_dia
        dia = Dialogue(
            utterances=utterances,
            speakers=speakers,
            source_dataset_name=name,
            idx_within_source=i,
            idx=idx
        )
        idx += 1
        res.append(dia)

    # for i, raw_dia in tqdm(enumerate(dataset), desc=f'preprocessing {name}'):
    #     parse_results = parse_sample(raw_dia, tokenizer, bound)
    #     if parse_results is None:
    #         continue
    #     utterances, speakers = parse_results
    #     dia = Dialogue(
    #         utterances=utterances,
    #         speakers=speakers,
    #         source_dataset_name=name,
    #         idx_within_source=i,
    #         idx=idx
    #     )
    #     idx += 1
    #     res.append(dia)
    return res


def train_test_split(data, frac=0.9, seed=0):
    """resulting sizes:
    - train: `frac`
    - test: `(1 - frac) // 2`
    - val: `(1 - frac) // 2`"""
    
    set_seet(seed)
    shuffle(data)

    n_total = len(data)
    train_size = ceil(frac * n_total)
    test_size = (n_total - train_size) // 2
    val_size = n_total - train_size - test_size

    res = {
        'train': data[:train_size],
        'test': data[train_size:train_size+test_size],
        'val': data[train_size+test_size:]
    }

    print('dataset splits sizes:')
    print(f'{n_total=}, {train_size=}, {test_size=}, {val_size=}')

    return res


def save_as_chunks(data: List[Dialogue], path, chunk_size, del_last_chunk=False):
    """saves `data` as json chunks to `save_path`"""
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    break_points = list(range(0, len(data) - chunk_size, chunk_size))
    
    if del_last_chunk:
        del break_points[-1]
    
    for i in tqdm(break_points):
        chunk_name = f'{i//chunk_size}.json'
        chunk_path = os.path.join(path, chunk_name)
        chunk = [dia.asdict() for dia in data[i:i+chunk_size]]
        json.dump(chunk, open(chunk_path, 'w'))


def make_pairs(dialogues):
    res = []
    for dia in tqdm(dialogues, desc='making pairs'):
        pairs = []
        for i in range(len(dia)-1):
            pairs.append((dia[:i+1], dia[i+1]))
        res.extend(pairs)
    shuffle(res)
    res = [ContextResponsePair(context=c, response=r, idx=i) for i, (c, r) in enumerate(res)]
    return res


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from collections import defaultdict
    from mylib.utils.training import seed_everything

    seed_everything(0)

    # supress warnings about long sequences
    import logging
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    #! not the same as roberta, replace in future
    tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base')

    # load datasets from hugging face, parse, filter and merge into single list
    merged_dataset = []
    for name in names:
        dataset = load_dataset('Salesforce/dialogstudio', name)['train']['log']
        parsed_dataset = parse_dataset(dataset, name, tokenizer, upper_bounds[name])
        merged_dataset.extend(parsed_dataset)


    # shuffle and define splits
    dialogues = train_test_split(merged_dataset)


    # save splits to file system as json chunks ('mylib/data/train/source')
    import os
    root_dir = os.environ['ROOT_DIR']
    save_path = os.path.join(root_dir, 'mylib', 'data', 'source')

    for split, data in dialogues.items():
        print(f'saving chunks for {split} dialogues')
        path = os.path.join(save_path, split)
        save_as_chunks(data, path, chunk_size=512)

    # === context-response pairs dataset ===

    # make pairs
    save_path = os.path.join(root_dir, 'mylib', 'data', 'train', 'context-response-pairs')
    nsp_dataset = defaultdict(list)
    for split in ['train', 'test', 'val']:
        nsp_dataset[split] = make_pairs(dialogues[split])
        print(split, len(nsp_dataset[split]))

    # save as chunks
    save_path = os.path.join(root_dir, 'mylib', 'data', 'train', 'context-response-pairs')
    for split, data in nsp_dataset.items():
        print(f'saving chunks for {split} context-response pairs')
        path = os.path.join(save_path, split)
        del_last = (split == 'test')
        save_as_chunks(data, path, chunk_size=2048, del_last_chunk=del_last)
