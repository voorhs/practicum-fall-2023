from ...utils.training import LightningCkptLoadable, HParamsPuller
from torch import nn
import torch
import torch.nn.functional as F
from .ranker import SortingMetric, _make_mask


class ClassifierHead(nn.Module, HParamsPuller):
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


class ClassifierSortingLoss(nn.Module):
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


class ClassifierSortingMetric(SortingMetric):
    def __call__(self, probs_logits, mask, dia_lens):
        labs = torch.argmax(probs_logits, dim=-1)
        ranks = -labs

        return super().__call__(ranks, mask, dia_lens)


class ClassifierListwise(nn.Module, LightningCkptLoadable, HParamsPuller):
    def __init__(self, dialogue_model, dropout_prob, max_n_uts):
        super().__init__()

        self.dialogue_model = dialogue_model
        self.dropout_prob = dropout_prob
        self.max_n_uts = max_n_uts
        
        self.clf_head = ClassifierHead(dialogue_model.get_hidden_size(), max_n_uts, dropout_prob)
        self.sorting_loss = ClassifierSortingLoss()
        self.metric_fn = ClassifierSortingMetric()

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
        mask = _make_mask(dia_lens, device)

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
        unbinded_logits = _unbind_logits(probs_logits, dia_lens)
        
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

def _unbind_logits(logits, dia_lens):
    """get list of tensors with logits corresponding to tokens(utterances) that are not padding ones only"""
    res = []
    for lgits, length in zip(logits, dia_lens):
        res.append(lgits[:length, :length])
    return res