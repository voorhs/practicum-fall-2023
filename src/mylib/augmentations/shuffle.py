import numpy as np
import torch
from tqdm import tqdm
import random
from .prune import _load_pairwise_cat, _cluster


class Shuffler:
    def __init__(
            self,
            ckpt_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/pairwise-cat-speaker-issue/checkpoints/last.ckpt',
            device='cpu',
            thresh=-np.inf
        ):
        self.thresh = thresh
        self.model = _load_pairwise_cat(ckpt_path, device)
    
    def from_argument(self, dialogues):
        res = []
        for dia in tqdm(dialogues, desc='shuffling dialogues'):
            aug, score = self._shuffle(self.model, dia)
            res.append(aug if score >= self.thresh else None)
        return res

    @staticmethod
    @torch.no_grad()
    def _shuffle(model, dia):
        if len(dia) < 12:
            return None, -np.inf
        end = len(dia) // 3
        start = 4
        variations = []
        for n_clusters in range(start, end+1):
            clusterwise_uts = _cluster(model, dia, n_clusters)
            for i_try in range(n_clusters):
                random.shuffle(clusterwise_uts)
                aug = []
                for ut_ids in clusterwise_uts:
                    aug.extend([dia[i] for i in ut_ids])
                score = model.score(aug)
                variations.append((aug, score))
        res, score = max(variations, key=lambda x: x[1])
        return res, score
