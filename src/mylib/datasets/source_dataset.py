import json
import os
import os
from bisect import bisect_right
import numpy as np
from torch.utils.data import Dataset
import math
import json
from ..utils.data import Dialogue


class DialogueDataset(Dataset):
    def __init__(self, path, fraction=1.):
        self.path = path
        
        chunk_names = [filename for filename in os.listdir(path) if filename.endswith('.json') and not filename.startswith('ru')]
        self.chunk_names = sorted(chunk_names, key=lambda x: int(x.split('.')[0]))
        
        size = math.ceil(len(self.chunk_names) * fraction)
        self.chunk_names = self.chunk_names[:size]
        
        chunk_sizes = [len(chunk) for chunk in (json.load(open(os.path.join(path, chunk_name))) for chunk_name in self.chunk_names)]
        self.chunk_beginnings = np.cumsum(chunk_sizes).tolist()

        self.n_chunks = len(self.chunk_names)
        self.len = self.chunk_beginnings[-1]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one training sample as
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
        idx_within_chunk = i - tmp[i_chunk]
        item = json.load(open(os.path.join(self.path, self.chunk_names[i_chunk]), 'r'))[idx_within_chunk]
        return Dialogue.get_train_sample(item)
