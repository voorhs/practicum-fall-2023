import math
from torch.utils.data import Dataset
import os
import json
from bisect import bisect_right
import numpy as np
import torch


class MultiWOZServiceClfDataset(Dataset):
    def __init__(self, path, fraction=1.):
        self.path = path

        dia_name_validator = lambda x: x.startswith('dia') and x.endswith('.json')
        services_name_validator = lambda x: x.startswith('services') and x.endswith('.json')
        number_extractor = lambda x: int(x.split('-')[1].split('.')[0])

        def get_names(name_validator):
            chunk_names = [filename for filename in os.listdir(path) if name_validator(filename)]
            return sorted(chunk_names, key=number_extractor)

        self.dia_chunk_names = get_names(dia_name_validator)
        self.services_chunk_names = get_names(services_name_validator)

        size = math.ceil(len(self.dia_chunk_names) * fraction)
        self.dia_chunk_names = self.dia_chunk_names[:size]
        self.services_chunk_names = self.services_chunk_names[:size]

        chunk_sizes = [len(chunk) for chunk in (json.load(open(os.path.join(path, chunk_name))) for chunk_name in self.dia_chunk_names)]
        self.chunk_beginnings = np.cumsum(chunk_sizes).tolist()

        self.services = [
            'attraction', 'bus', 'hospital',
            'hotel', 'restaurant', 'taxi', 'train'
        ]
        self.len = self.chunk_beginnings[-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        i_chunk = bisect_right(self.chunk_beginnings, x=i)
        tmp = [0] + self.chunk_beginnings
        idx_within_chunk = i - tmp[i_chunk]

        dia = self._get_item(self.dia_chunk_names, i_chunk, idx_within_chunk)
        services = self._get_item(self.services_chunk_names, i_chunk, idx_within_chunk)
        
        target = torch.tensor([float(serv in services) for serv in self.services])    # multi one hot
        
        return dia, target
    
    def _get_item(self, names, i_chunk, idx_within_chunk):
        path_to_chunk = os.path.join(self.path, names[i_chunk])
        chunk = json.load(open(path_to_chunk, 'r'))
        return chunk[idx_within_chunk]
