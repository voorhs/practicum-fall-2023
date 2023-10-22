import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from bisect import bisect_left
from .train_utils import LightningCkptLoadable, HParamsPuller
from typing import Literal
import warnings


# ======= ranker =======

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


class UtteranceSorter(nn.Module, LightningCkptLoadable, HParamsPuller):
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
        mask = self._make_mask(dia_lens, device)
        ranks_logits.masked_fill_(mask, -1e4)

        loss = self.sorting_loss(ranks_logits, dia_lens)
        if self.with_contrastive:
            hidden_states = self.contraster_head(hidden_states)
            pairing_loss = self.pairing_loss(hidden_states, mask, dia_lens)
            loss = (loss * 5 + pairing_loss) / 6
        metric = self.metric_fn(ranks_logits, mask, dia_lens)

        return loss, metric

    @torch.no_grad()
    def augment(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        ranks_logits = self.get_logits(batch)
        mask = self._make_mask(dia_lens, device)
        unbinded_ranks_logits = self.metric_fn._unbind_logits(ranks_logits, mask, dia_lens)
        permutations = self.metric_fn._to_permutations(unbinded_ranks_logits)

        return [[dia[i] for i in perm] for dia, perm in zip(batch, permutations)]

    @staticmethod
    def _make_mask(dia_lens, device):
        """this mask indicates padding tokens(utterances). used for ranking (not for transformer)"""
        T = max(dia_lens)
        dia_lens_expanded = torch.tensor(dia_lens, device=device)[:, None]
        max_dia_len_expanded = torch.arange(T, device=device)[None, :]
        return dia_lens_expanded <= max_dia_len_expanded
    

# ======= classifier =======

class ClfRankerHead(nn.Module, HParamsPuller):
    def __init__(self, hidden_size, n_classes, dropout_prob):
        super().__init__()

        self.lin = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.clf = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x: torch.Tensor):
        # x = F.gelu(self.lin(x)) + x
        # x = self.dropout(x)
        x = self.clf(x)
        return x


class ClfSortingLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, mask, dia_lens):
        """
        logits: (B, T, C)
        mask: (B, T), mask indicating padding utterances
        """
        device = logits.device

        for i, length in enumerate(dia_lens):
            logits[i, :, length:] = -1e4

        B, T, C = logits.shape
        logits = logits.reshape(-1, C)
        labels = self._make_true_labels(B, T, device)
        dirty_loss = self.loss_fn(logits, labels)
        mask = mask.reshape(-1)
        loss = dirty_loss[~mask].mean()
        return loss
    
    @staticmethod
    def _make_true_labels(B, T, device):
        return torch.arange(T, device=device)[None, :].expand(B, T).reshape(-1)


class ClfSortingMetric(SortingMetric):
    def __call__(self, probs_logits, mask, dia_lens):
        labs = torch.argmax(probs_logits, dim=-1)
        ranks = -labs

        return super().__call__(ranks, mask, dia_lens)


class ClfUtteranceSorter(nn.Module, LightningCkptLoadable, HParamsPuller):
    def __init__(self, dialogue_model, dropout_prob, max_n_uts):
        super().__init__()

        self.dialogue_model = dialogue_model
        self.dropout_prob = dropout_prob
        self.max_n_uts = max_n_uts
        
        self.clf_head = ClfRankerHead(dialogue_model.get_hidden_size(), max_n_uts, dropout_prob)
        self.sorting_loss = ClfSortingLoss()
        self.metric_fn = ClfSortingMetric()

    @property
    def device(self):
        return self.dialogue_model.device

    def get_logits(self, batch):
        hidden_states = self.dialogue_model(batch)
        return self.clf_head(hidden_states)

    def forward(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        # (B, T, C), where C is the # of max uts in dia
        probs_logits = self.get_logits(batch)
        mask = UtteranceSorter._make_mask(dia_lens, device)

        loss = self.sorting_loss(probs_logits, mask, dia_lens)
        metric = self.metric_fn(probs_logits, mask, dia_lens)

        return loss, metric

    @staticmethod
    def make_batch_from_dia(dia):
        return [dia]

    @torch.no_grad()
    def score(self, dia=None, batch=None, return_logits=False):
        if bool(dia) == bool(batch):
            raise ValueError('either dia or batch should be provided')
        if dia is not None:
            batch = self.make_batch_from_dia(dia)

        # (B, T, C)
        probs_logits = self.get_logits(batch)
        
        dia_lens = [len(dia) for dia in batch]
        unbinded_logits = DecoderSortingMetric._unbind_logits(probs_logits, dia_lens)
        
        res_non_reducted = []
        res_geo_mean = []
        res_arith_mean = []
        for logits in unbinded_logits:
            probs = F.softmax(logits, dim=1)
            logits = probs.diag()
            res_non_reducted.append(logits.cpu().numpy())
            res_geo_mean.append(logits.log().mean().exp().cpu().item())
            res_arith_mean.append(logits.mean().cpu().item())

        if return_logits:
            return unbinded_logits, res_non_reducted, res_geo_mean, res_arith_mean
        return res_non_reducted, res_geo_mean, res_arith_mean
    
    @torch.no_grad()
    def augment(self, batch, decoder):
        unbinded_logits, _, _, scores = self.score(batch=batch, return_logits=True)
        permutations = decoder(unbinded_logits)
        return [([dia[i] for i in perm], score) for perm, dia, score in zip(permutations, batch, scores)]


# ======= decoder =======

class DecoderSortingLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, logits, mask, dia_lens):
        """
        logits: (B, T, C)
        mask: (B, T), mask indicating padding utterances
        """
        device = logits.device

        for i, length in enumerate(dia_lens):
            logits[i, length:, :] = -1e4

        B, T, _ = logits.shape
        mask = mask.reshape(-1)
        logits = logits.transpose(1, 2).reshape(-1, T)[~mask]
        labels = self._make_true_labels(dia_lens, device)
        loss = self.loss_fn(logits, labels) #! add temperature

        return loss
    
    @staticmethod
    def _make_true_labels(dia_lens, device):
        res = [torch.arange(length, device=device) for length in dia_lens]
        return torch.concat(res)


class DecoderSortingMetric:
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self, probs_logits, dia_lens):
        """
        probs_logits: (B, T, C)
        """
        unbinded_logits = self._unbind_logits(probs_logits, dia_lens)
        permutations = self.decoder(unbinded_logits)
        return 1-np.mean([SortingMetric._normalized_inversions_count(perm) for perm in permutations])

    @staticmethod
    def _unbind_logits(logits, dia_lens):
        """get list of tensors with logits corresponding to tokens(utterances) that are not padding ones only"""
        res = []
        for lgits, length in zip(logits, dia_lens):
            res.append(lgits[:length, :length])
        return res
    

from scipy.special import log_softmax


class Decoder:
    def __init__(self, top_k=0, top_p=0., beams=0):
        self.top_k = top_k
        self.top_p = top_p
        self.beams = beams
    
    def __call__(self, unbinded_logits):
        device = unbinded_logits[0].device
        if self.beams != 0:
            permutations = Decoder._beam_decode(unbinded_logits, self.beams)
        else:
            permutations = Decoder._sampling_decode(unbinded_logits, device, self.top_k, self.top_p)
        
        return permutations
            
    @staticmethod
    def _sampling_decode(unbinded_logits, device, top_k=0, top_p=0.):
        res = []
        for logits in unbinded_logits:
            T, _ = logits.shape
            not_selected = list(range(T))
            perm = []
            for cur_pos_logits in logits.T:
                # (T,)
                ranks = cur_pos_logits[torch.tensor(not_selected, device=device, dtype=torch.int)]
                if top_k != 0 or top_p != 0.:
                    ranks = Decoder._top_filtering(ranks, top_k, top_p)
                    i_selected = not_selected[torch.multinomial(F.softmax(ranks, dim=0), 1)]
                else:
                    i_selected = not_selected[torch.argmax(ranks)]
                perm.append(i_selected)
                not_selected.remove(i_selected)
            res.append(perm)
        return res
    
    @staticmethod
    def _beam_decode(unbinded_logits, beams):
        # unbinded logits: list of (T, C)
        return [Decoder._beam_search(logits.T.cpu().numpy(), perm=[], score=0, beams=beams)[0][0] for logits in unbinded_logits]
        
    @staticmethod
    def _beam_search(logits, perm, score, beams):
        # logits: tensor of (C, T)
        
        C, T = logits.shape
        cur_pos = len(perm)
        if cur_pos == min(C, T):
            return [(perm, score)]
        
        cur_logits = logits[cur_pos]
        cur_logits[perm] = -1e4
        all_scores = score + log_softmax(cur_logits)
        top_scorers = np.argpartition(all_scores, kth=-beams)[-beams:]
        all_res = []
        for i in top_scorers:
            cur_perm = perm + [i]
            cur_score = all_scores[i]
            all_res.extend(Decoder._beam_search(logits, cur_perm, cur_score, beams))
        
        all_res = sorted(all_res, key=lambda x: x[1], reverse=True)
        
        return all_res[:beams]
    
    @staticmethod
    def _top_filtering(logits, top_k, top_p, filter_value=-torch.inf):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k >0: keep only top k tokens with highest probability (top-k filtering).
                top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            
                Taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits


class DecoderUtteranceSorter(ClfUtteranceSorter):
    def __init__(self, dialogue_model, dropout_prob, max_n_uts, decoder, temperature=0.1):
        super().__init__(dialogue_model, dropout_prob, max_n_uts)

        self.dialogue_model = dialogue_model
        self.dropout_prob = dropout_prob
        self.max_n_uts = max_n_uts
        self.temperature = temperature
        
        self.clf_head = ClfRankerHead(dialogue_model.get_hidden_size(), max_n_uts, dropout_prob)
        self.sorting_loss_clf = ClfSortingLoss()
        self.sorting_loss_dec = DecoderSortingLoss()
        self.metric_fn = DecoderSortingMetric(decoder)

    @property
    def device(self):
        return self.dialogue_model.device

    def get_logits(self, batch):
        hidden_states = self.dialogue_model(batch)
        return self.clf_head(hidden_states)

    def forward(self, batch):
        device = self.device
        dia_lens = [len(dia) for dia in batch]

        if max(dia_lens) > self.max_n_uts:
            raise ValueError('theres a dialogue with exceeding utterances count')
        # (B, T, C), where C is the # of max uts in dia
        probs_logits = self.get_logits(batch) / self.temperature
        mask = UtteranceSorter._make_mask(dia_lens, device)
        loss_1 = self.sorting_loss_clf(probs_logits, mask, dia_lens)
        
        T = max(dia_lens)
        mask = F.pad(mask, pad=(0, self.max_n_uts-T, 0,0), value=True)

        loss_2 = self.sorting_loss_dec(probs_logits, mask, dia_lens)
        metric = self.metric_fn(probs_logits, dia_lens)

        loss = (loss_1 + loss_2) / 2

        return loss, metric
