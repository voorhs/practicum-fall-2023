from torch import nn
import torch
import torch.nn.functional as F
from .ranker import SortingMetric
from .classifier import ClassifierListwise, ClassifierHead, ClassifierSortingLoss, _make_mask, _unbind_logits
import numpy as np
from scipy.special import log_softmax


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
        unbinded_logits = _unbind_logits(probs_logits, dia_lens)
        permutations = self.decoder(unbinded_logits)
        return 1-np.mean([SortingMetric._normalized_inversions_count(perm) for perm in permutations])


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


class DecoderListwise(ClassifierListwise):
    def __init__(self, dialogue_model, dropout_prob, max_n_uts, decoder, temperature=0.1):
        super().__init__(dialogue_model, dropout_prob, max_n_uts)

        self.dialogue_model = dialogue_model
        self.dropout_prob = dropout_prob
        self.max_n_uts = max_n_uts
        self.temperature = temperature
        
        self.clf_head = ClassifierHead(dialogue_model.get_hidden_size(), max_n_uts, dropout_prob)
        self.sorting_loss_clf = ClassifierSortingLoss()
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
        mask = _make_mask(dia_lens, device)
        loss_1 = self.sorting_loss_clf(probs_logits, mask, dia_lens)
        
        T = max(dia_lens)
        mask = F.pad(mask, pad=(0, self.max_n_uts-T, 0,0), value=True)

        loss_2 = self.sorting_loss_dec(probs_logits, mask, dia_lens)
        metric = self.metric_fn(probs_logits, dia_lens)

        loss = (loss_1 + loss_2) / 2

        return loss, metric
