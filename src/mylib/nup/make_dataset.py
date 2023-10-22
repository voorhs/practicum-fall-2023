import json
from random import shuffle, seed
from math import ceil
import os
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


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
    # 'SGD': upper_bound,
    'SimJointGEN': upper_bound,
    'KETOD': upper_bound,
    'FRAMES': upper_bound,
    'Disambiguation': min(upper_bound, 60),
    'ABCD': upper_bound,
    'AirDialogue': upper_bound,
    'BiTOD': upper_bound,
    'Taskmaster1': min(upper_bound, 200),
}

tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base', max_length=10000)

merged_dataset = []
for name in names:
    dataset = load_dataset('Salesforce/dialogstudio', name)['train']['log']
    res = []
    if name not in upper_bounds.keys():
        for dia in tqdm(dataset, desc=name):
            cur_dia = []
            if len(dia) == 0 or len(dia) > 10 or len(dia) == 1 and (dia[0]['user utterance'] == '' or dia[0]['system response'] == ''):
                continue
            for turn in dia:
                ut = turn['user utterance']
                if ut != '':
                    cur_dia.append({'utterance': ut, 'speaker': 0})
                ut = turn['system response']
                if ut != '':
                    cur_dia.append({'utterance': ut, 'speaker': 1})
            res.append(cur_dia)

    else:
        bound = upper_bounds[name]
        for dia in tqdm(dataset, desc=name):
            cur_dia = []
            if len(dia) == 0 or len(dia) > 10 or len(dia) == 1 and (dia[0]['user utterance'] == '' or dia[0]['system response'] == ''):
                continue
            is_below_bound = []
            for turn in dia:
                for sp, item in zip([0, 1], ['user utterance', 'system response']):
                    ut = turn[item]
                    if ut == '':
                        continue
                    cur_dia.append({'utterance': ut, 'speaker': sp})
                    is_below_bound.append(len(tokenizer(ut)['input_ids']) <= bound)
            if all(is_below_bound):
                res.append(cur_dia)
    merged_dataset.extend(res)

seed(0)
shuffle(merged_dataset)

n_total = len(merged_dataset)
train_size = ceil(0.9 * n_total)
test_size = (n_total - train_size) // 2
val_size = n_total - train_size - test_size

dialogues = {
    'train': merged_dataset[:train_size],
    'test': merged_dataset[train_size:train_size+test_size],
    'val': merged_dataset[train_size+test_size:]
}

chunk_size = 512
print('dialogues number:')
print(f'{n_total=}, {train_size=}, {test_size=}, {val_size=}')

for split, data in dialogues.items():
    print(f'saving chunks for {split} dialogues')
    path = f'dialogues/{split}'
    if not os.path.exists(path):
        os.makedirs(path)
    break_points = list(range(0, len(data) - chunk_size, chunk_size))
    for i in tqdm(break_points):
        json.dump(data[i:i+chunk_size], open(f'{path}/{i//chunk_size}.json', 'w'))

nsp_dataset = defaultdict(list)
for split in ['train', 'test', 'val']:
    for dia in tqdm(dialogues[split], desc=f'making pairs for {split}'):
        pairs = []
        for i in range(len(dia)-1):
            pairs.append({
                'context': dia[:i+1],
                'target': dia[i+1]
            })
        nsp_dataset[split].extend(pairs)
    shuffle(nsp_dataset[split])
    print(split, len(nsp_dataset[split]))

chunk_size = 2048
for split, data in nsp_dataset.items():
    print(f'saving chunks for {split} pairs')
    path = f'pairs/{split}'
    if not os.path.exists(path):
        os.makedirs(path)
    break_points = list(range(0, len(data) - chunk_size, chunk_size))
    if split == 'test':
        del break_points[-1]
    for i in tqdm(break_points):
        json.dump(data[i:i+chunk_size], open(f'{path}/{i//chunk_size}.json', 'w'))
