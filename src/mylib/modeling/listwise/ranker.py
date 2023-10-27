from ...utils.training import LightningCkptLoadable, HParamsPuller
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from bisect import bisect_left


class RankerHead(nn.Module, HParamsPuller):
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()

        self.lin = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.ranker = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, x: torch.Tensor):
        x = F.gelu(self.lin(x)) + x
        x = self.dropout(x)
        x = self.ranker(x)
        return x.squeeze(-1)


class SortingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, ranks_logits, dia_lens):
        ranks_logprobs = F.log_softmax(ranks_logits, dim=1)
        _, T = ranks_logits.shape
        device = ranks_logits.device
        ranks_true = self._make_true_ranks(T, dia_lens, device)
        return self.loss_fn(ranks_logprobs, ranks_true)

    @staticmethod
    def _make_true_ranks(T, dia_lens, device):
        res = []
        
        for length in dia_lens:
            ranks = torch.linspace(1, 0, length, device=device)
            ranks = F.pad(ranks, pad=(0, T-length), value=0)
            ranks = ranks / ranks.sum()
            res.append(ranks)
        
        return torch.stack(res)


class SortingMetric:
    def __call__(self, ranks_logits, mask, dia_lens):
        unbinded_ranks_logits = self._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self._to_permutations(unbinded_ranks_logits)
        return 1-np.mean([self._normalized_inversions_count(perm) for perm in permutations])
       
    @staticmethod
    def _unbind_logits(logits, mask, dia_lens):
        """get list of tensors with logits corresponding to tokens(utterances) that are not padding ones only"""
        return logits[~mask].detach().cpu().split(dia_lens)

    @staticmethod
    def _to_permutations(unbinded_ranks_logits):
        """permutations with respect to descending order"""
        return [logits.argsort(descending=True) for logits in unbinded_ranks_logits]

    @staticmethod
    def _normalized_inversions_count(arr):
        """Function to count number of inversions in a permutation of 0, 1, ..., n-1."""
        n = len(arr)
        v = list(range(n))
        ans = 0
        for i in range(n):
            itr = bisect_left(v, arr[i])
            ans += itr
            del v[itr]
        max_inversions = n * (n - 1) / 2
        return ans / max_inversions


class RankerListwise(nn.Module, LightningCkptLoadable, HParamsPuller):
    def __init__(self, dialogue_model, dropout_prob):
        super().__init__()

        self.dialogue_model = dialogue_model
        self.dropout_prob = dropout_prob
        
        self.ranker_head = RankerHead(dialogue_model.get_hidden_size(), dropout_prob)
        self.sorting_loss = SortingLoss()
        self.metric_fn = SortingMetric()

    @property
    def device(self):
        return self.dialogue_model.device

    def get_logits(self, batch):
        hidden_states = self.dialogue_model(batch)
        return hidden_states, self.ranker_head(hidden_states)

    def forward(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        hidden_states, ranks_logits = self.get_logits(batch)

        # zero attention to padding token-utterances
        mask = _make_mask(dia_lens, device)
        ranks_logits.masked_fill_(mask, -1e4)

        loss = self.sorting_loss(ranks_logits, dia_lens)
        metric = self.metric_fn(ranks_logits, mask, dia_lens)

        return loss, metric

    @torch.no_grad()
    def augment(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        ranks_logits = self.get_logits(batch)
        mask = _make_mask(dia_lens, device)
        unbinded_ranks_logits = self.metric_fn._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self.metric_fn._to_permutations(unbinded_ranks_logits)

        return [[dia[i] for i in perm] for dia, perm in zip(batch, permutations)]


def _make_mask(dia_lens, device):
    """this mask indicates padding tokens(utterances). used for ranking (not for transformer)"""
    T = max(dia_lens)
    dia_lens_expanded = torch.tensor(dia_lens, device=device)[:, None]
    max_dia_len_expanded = torch.arange(T, device=device)[None, :]
    return dia_lens_expanded <= max_dia_len_expanded
