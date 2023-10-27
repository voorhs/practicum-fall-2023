from torch.utils.data import Dataset
import os
import json
from bisect import bisect_right
import numpy as np


class ContrastiveDataset(Dataset):
    def __init__(self, path):
        self.path = path
        
        chunk_names = [filename for filename in os.listdir(path) if filename.endswith('.json') and not filename.startswith('ru')]
        self.chunk_names = sorted(chunk_names, key=lambda x: int(x.split('.')[0]))
        chunk_sizes = [len(chunk) for chunk in (json.load(open(os.path.join(path, chunk_name))) for chunk_name in self.chunk_names)]
        self.chunk_beginnings = np.cumsum(chunk_sizes).tolist()
        
        self.n_chunks = len(self.chunk_names)
        self.len = self.chunk_beginnings[-1]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one training sample as
        {
            'orig': dia,
            'pos': list of dias,
            'neg': list of dias
        }
        where each dia is represented with an object of the following schema:
        ```
        {
            "type": "array",
            "items":
            {
                "type": "object",
                "properties":
                {
                    "utterance": {"type": "string"},
                    "speaker": {"type": "number"}
                }
            }
        }
        ```"""
        i_chunk = bisect_right(self.chunk_beginnings, x=i)
        tmp = [0] + self.chunk_beginnings
        idx_within_chunk = i -  tmp[i_chunk]
        item = json.load(open(os.path.join(self.path, self.chunk_names[i_chunk]), 'r'))[idx_within_chunk]
        return item
