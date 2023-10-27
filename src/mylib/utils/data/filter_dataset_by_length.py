import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from ...modeling.dialogue import BaselineDialogueEncoder
from typing import Literal
from ...utils.data import Dialogue


def is_short_enough(dia: dict, tokenizer, upper_bound=512):
    """`dia` should be shorter than 512 minus number of SEP and CLS tokens"""
    dia = Dialogue.get_train_sample(dia)
    input_ids = BaselineDialogueEncoder._tokenize(tokenizer, [dia])['input_ids'][0]
    return len(input_ids) <= upper_bound


def filter_dataset_by_length(
        path_in,
        path_out,
        tokenizer,
        mode: Literal['null', 'drop'] = 'null',
        name_validator=None,
        number_extractor=None
    ):
    """Copies all json chunks of a dataset from `path_in` to `path_out`.
    Each dia which is `None` or exceeding the length limit is
    replaced with `None` or dropped according to `mode`.
    
    About length limit: each dia should be shorter than 512
    minus number of SEP and CLS tokens (see `BaselineDialogueEncoder`)."""

    if name_validator is None:
        name_validator = lambda x: x.endswith('.json') and not x.startswith('ru')
    
    if number_extractor is None:
        number_extractor = lambda x: int(x.split('.')[0])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
 
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    chunk_names = [filename for filename in os.listdir(path_in) if name_validator(filename)]
    chunk_names = sorted(chunk_names, key=number_extractor)

    for chunk_name in tqdm(chunk_names):
        chunk_path_in = os.path.join(path_in, chunk_name)
        chunk = json.load(open(chunk_path_in, 'r'))
        
        filtered_chunk = []
        for i, dia in enumerate(chunk):
            res = dia
            if not (dia is None or not is_short_enough(dia, tokenizer)):
                filtered_chunk.append(res)
            else: 
                print(f'rejected dia #{i}')
                if mode == 'null':
                    filtered_chunk.append(None)
                elif mode == 'drop':
                    pass
                else:
                    raise ValueError(f'unknown mode {mode}')

        chunk_path_out = os.path.join(path_out, chunk_name)
        json.dump(filtered_chunk, open(chunk_path_out, 'w'))
