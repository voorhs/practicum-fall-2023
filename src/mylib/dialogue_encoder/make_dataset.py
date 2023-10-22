original_path = '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/original'

negative_paths = [
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/replace',
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/replace-cut',
]

positive_paths = [
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/back-translate',
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/back-translate-cut',
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/back-translate-shuffle',
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/cut-insert',
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/insert',
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/pairwise-shuffler',
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/pairwise-cutter',
    '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/augmented/shuffle-insert',
]

res_path = '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/dataset/train'

N_CHUNKS = 80
CHUNK_SIZE = 512
ALLOW_ABSENT_NEGATIVES = True

import json
import os
from tqdm import tqdm

# === data validation ===

def validate(paths):
    all_chunk_names = []
    for pos_path in tqdm(paths, desc='validating provided paths'):
        chunk_names = [filename for filename in os.listdir(pos_path) if filename.endswith('.json') and not filename.startswith('ru')]
        if len(chunk_names) != N_CHUNKS:
            print('wrong number of chunks:', pos_path)
        for chunk_name in chunk_names:
            chunk_path = os.path.join(pos_path, chunk_name)
            chunk = json.load(open(chunk_path, 'r'))
            if len(chunk) != 512:
                print('wrong chunk size:', chunk_name, pos_path)
        all_chunk_names.append(chunk_names)

    for chunk_names in all_chunk_names[1:]:
        if any(name1 != name2 for name1, name2 in zip(all_chunk_names[0], chunk_names)):
            print('chunk names must match')

validate(positive_paths + negative_paths + [original_path])

# === generating dataset ===

if not os.path.exists(res_path):
    os.makedirs(res_path)

chunk_names = [filename for filename in os.listdir(positive_paths[0]) if filename.endswith('.json') and not filename.startswith('ru')]
chunk_names = sorted(chunk_names)[:N_CHUNKS]

def read_chunk(paths, chunk_name):
    """read `chunk_name` from all `paths` and return as list"""
    res = []
    for path in paths:
        chunk_path = os.path.join(path, chunk_name)
        chunk = json.load(open(chunk_path, 'r'))
        res.append(chunk)
    return [[dia for dia in dias if dia is not None] for dias in zip(*res)]

def join(dia):
    uts = [item['utterance'] for item in dia]
    res = '###'.join(uts)
    return res

def my_filter(dias, orig_joined):
    """filter out `dias` that are identical to `orig_joined`"""
    return [dia for dia in dias if join(dia) != orig_joined]

for chunk_name in tqdm(chunk_names, desc='generating dataset'):
    orig_chunk = read_chunk([original_path], chunk_name)
    pos_chunk = read_chunk(positive_paths, chunk_name)
    neg_chunk = read_chunk(negative_paths, chunk_name)
    
    res_chunk = []
    for orig_dia, pos_dias, neg_dias in zip(orig_chunk, pos_chunk, neg_chunk):
        if not orig_dia:
            # this case corresponds to dia that is too long (see `is_short_enough()` in filter_dataset_by_length.py)
            print('dia is too long')
            continue
        
        orig_dia = orig_dia[0]
        orig_joined = join(orig_dia)
        
        pos = my_filter(pos_dias, orig_joined)
        neg = my_filter(neg_dias, orig_joined)
        if not pos:
            # dia that has no positives that differ from it
            print('dia has no positives')
            continue

        if not neg:
            # dia that has no negatives that differ from it
            print('dia has no negatives')
            if not ALLOW_ABSENT_NEGATIVES:
                continue

        cur_res = {
            'orig': orig_dia,
            'pos': pos,
            'neg': neg,
        }
        res_chunk.append(cur_res)
    
    res_chunk_path = os.path.join(res_path, chunk_name)
    json.dump(res_chunk, open(res_chunk_path, 'w'))

# === final validation ===

for chunk_name in tqdm(chunk_names, desc='post validation'):
    chunk_path = os.path.join(res_path, chunk_name)
    chunk = json.load(open(chunk_path, 'r'))
    for sample in chunk:
        # check if any of `orig`, `pos`, (optionally `neg`) is absent
        for key, val in sample.items():
            if not val:
                print(f'{key} is null')
