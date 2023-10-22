from torch.utils.data import Dataset
import os
import json
import numpy as np
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Tuple
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch
from typing import Literal
import torch.nn.functional as F
from torchmetrics.functional.classification import multilabel_f1_score
from torch.utils.data import DataLoader
import math
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


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


@dataclass
class LearnerConfig:
    kwargs: field(default_factory=dict) = None,
    lr: float = None
    batch_size: int = None
    warmup_period: int = None
    do_periodic_warmup: bool = False
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)
    k: int = 5
    t: float = 0.05
    loss: Literal['contrastive', 'ict', 'multiwoz_service_clf'] = 'contrastive'


class Learner(pl.LightningModule):
    def __init__(self, model, config: LearnerConfig):
        super().__init__()
        self.model = model
        self.config = config

        # list of (embedding, target) pairs for multiwoz service clf (as validation)
        self.multiwoz_train = []
        self.multiwoz_validation = []

        if self.config.loss == 'multiwoz_service_clf':
            self.clf_head = nn.Linear(self.model.get_hidden_size(), 7)

    def forward(self, batch):
        if self.config.loss == 'contrastive':
            return self._contrastive_step(batch)
        if self.config.loss == 'ict':
            raise NotImplementedError()
        if self.config.loss == 'multiwoz_service_clf':
            raise NotImplementedError()

    def _contrastive_step(self, batch):
        """`batch` is a list of samples from ContrastiveDataset"""
        origs = [sample['orig'] for sample in batch]
        
        # select positives
        points = np.random.uniform(low=0, high=1, size=len(batch))
        counts = np.array([len(sample['pos']) for sample in batch])
        pos_indices = np.floor(points * counts).astype(np.int_)
        
        positives = [sample['pos'][i] for i, sample in zip(pos_indices, batch)]

        # select hard_negatives
        hard_negatives = []
        hard_negatives_counts = []
        for sample in batch:
            negs = sample['neg']
            hard_negatives.extend(negs)
            hard_negatives_counts.append(len(negs))

        # encode all dialogues
        origs_enc = F.normalize(self.model(origs), dim=1)                   # (B, H)
        positives_enc = F.normalize(self.model(positives), dim=1)           # (B, H)
        hard_negatives_enc = F.normalize(self.model(hard_negatives), dim=1) # (B+, H)

        # pos and neg scores
        pairwise_scores = (origs_enc @ positives_enc.T / self.config.t).exp()
        pos_scores = pairwise_scores.diag()
        neg_scores1 = pairwise_scores.sum(dim=0)
        neg_scores2 = pairwise_scores.sum(dim=1)
        
        # hard neg scores
        repeats = torch.tensor(hard_negatives_counts, device=self.model.device)
        origs_enc_repeated = torch.repeat_interleave(origs_enc, repeats=repeats, dim=0)
        _hard_neg_scores = (origs_enc_repeated * hard_negatives_enc / self.config.t).sum(dim=1).exp()
        hard_neg_scores = []
        for i, count in enumerate(hard_negatives_counts):
            start = sum(hard_negatives_counts[:i])
            end = start + count
            score = _hard_neg_scores[start:end].sum()
            hard_neg_scores.append(score)
        hard_neg_scores = torch.tensor(hard_neg_scores, device=self.model.device)

        # compute contrastive loss with hard negatives
        loss = (pos_scores / (neg_scores1 + neg_scores2 + hard_neg_scores)).log().neg().sum()
        
        # compute metric: retrieval accuracy
        topk_indicators = [i in top for i, top in enumerate(torch.topk(pairwise_scores, k=self.config.k, dim=1).indices)]
        topk_accuracy = np.mean(topk_indicators)

        return loss, topk_accuracy

    def _multiwoz_service_clf_step(self, batch):
        """`batch` is a list of samples from MultiWOZServiceClfDataset"""
        dialogues = [dia for dia, _ in batch]
        targets = torch.stack([tar for _, tar in batch], dim=0)
        
        embeddings = self.model(dialogues)  # (B, H)
        logits = self.clf_head(embeddings)  # (B, 7)
        
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        metric = multilabel_f1_score(logits, targets, average='macro')

        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self.forward(batch)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        self.log(
            name='train_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        dialogues = [dia for dia, _ in batch]
        targets = torch.stack([tar for _, tar in batch], dim=0).detach().cpu().numpy()
        embeddings = self.model(dialogues).detach().cpu().numpy()
        res = list(zip(embeddings, targets))

        if dataloader_idx == 0:
            self.multiwoz_train.extend(res)
        elif dataloader_idx == 1:
            self.multiwoz_validation.extend(res)
    
    def on_train_start(self):        
        optim_hparams = self.optimizers().defaults
        model_hparams = self.model.get_hparams()
        model_hparams.update(optim_hparams)
        model_hparams['batch size'] = self.config.batch_size
        model_hparams['warmup period'] = self.config.warmup_period
        model_hparams['do periodic warmup'] = self.config.do_periodic_warmup
        model_hparams['k'] = self.config.k
        model_hparams['t'] = self.config.t
        model_hparams.update(self.config.kwargs)
        self.logger.log_hyperparams(model_hparams)

    def on_validation_epoch_end(self) -> None:
        metric = get_multiwoz_service_clf_score_sklearn(
            self.multiwoz_train,
            self.multiwoz_validation
        )

        self.log(
            name='val_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            # batch_size=self.config.batch_size
        )

        self.multiwoz_train.clear()
        self.multiwoz_validation.clear()

    def configure_optimizers(self):
        """Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, )
        # blacklist_weight_modules = (NoneType,)   #(torch.nn.LayerNorm, torch.nn.Embedding)
        for pn, p in self.named_parameters():

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(pn)
            else:
                decay.add(pn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optim_groups, lr=self.config.lr, betas=self.config.betas)
        def lr_foo(step):
            warmup_steps = self.config.warmup_period
            periodic = self.config.do_periodic_warmup
            
            if warmup_steps is None:
                return 1
            if periodic:
                return (step % warmup_steps + 1) / warmup_steps
            else:
                return (step + 1) / warmup_steps if step < warmup_steps else 1

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}


def get_multiwoz_service_clf_score_sklearn(train_dataset, val_dataset, n_epochs=2):
    # configure model
    clf = MLPClassifier(
        batch_size=32,
        learning_rate_init=5e-4,
        max_iter=n_epochs
    )

    # configure data
    X_train = np.stack([emb for emb, _ in train_dataset], axis=0)
    y_train = np.stack([tar for _, tar in train_dataset], axis=0)
    X_val = np.stack([emb for emb, _ in val_dataset], axis=0)
    y_val = np.stack([tar for _, tar in val_dataset], axis=0)
    
    # train model
    clf.fit(X_train, y_train)

    # score model
    y_pred = clf.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro', zero_division=0)
    
    return score


def get_multiwoz_service_clf_score_torch(train_dataset, val_dataset, device, n_epochs=2):
    # === configure model ===
    emb, tar = train_dataset[0]
    input_size = len(emb)
    n_classes = len(tar)
    
    learner = LinearProbeClf(input_size, n_classes).to(device)
    optimizer = learner.configure_optimizers()

    # === configure data ===
    def collate_fn(batch):
        embeddings = np.stack([emb for emb, _ in batch], axis=0)
        targets = np.stack([tar for _, tar in batch], axis=0)

        embeddings = torch.from_numpy(embeddings)
        targets = torch.from_numpy(targets)

        embeddings.requires_grad_(False)
        targets.requires_grad_(False)
        
        return embeddings, targets
        
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    # === train ===
    for i_epoch in range(n_epochs):
        for i_batch, batch in enumerate(train_loader):
            embeddings = batch[0].to(device)
            targets = batch[1].to(device)

            optimizer.zero_grad()
            loss = learner(embeddings, targets)

            loss.backward()
            optimizer.step()

    # === validate ===
    for i_batch, batch in enumerate(val_loader):
        embeddings = batch[0].to(device)
        targets = batch[1].to(device)

        learner.validation_step((embeddings, targets), i_batch)
    
    score = learner.on_validation_epoch_end()
    return score


class LinearProbeClf(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()

        self.clf = nn.Linear(input_size, n_classes)

        self.validation_steps_logits = []
        self.validation_steps_targets = []
    
    def forward(self, embeddings, targets):
        logits = self.clf(embeddings)
        print(logits.requires_grad, targets.requires_grad)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        return loss
    
    # def training_step(self, batch, batch_idx):
    #     embeddings, targets = batch
    #     return self.forward(embeddings, targets)
    
    def validation_step(self, batch, batch_idx):
        embeddings, targets = batch
        logits = self.clf(embeddings)  # (B, 7)
        
        self.validation_steps_logits.append(logits)
        self.validation_steps_targets.append(targets)

    def on_validation_epoch_end(self):
        logits = torch.concat(self.validation_steps_logits, dim=0)
        targets = torch.concat(self.validation_steps_targets, dim=0)
        num_labels = targets.shape[1]
        metric = multilabel_f1_score(logits, targets, num_labels=num_labels, average='macro')
        # self.log(name='result', value=metric)
        return metric.cpu().item()
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-4)


class LightningCkptLoadable:
    @staticmethod
    def from_checkpoint(path_to_ckpt, model, learner_config=LearnerConfig(), map_location=None):
        return Learner.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=model,
            config=learner_config,
        ).model


def freeze_hf_model(hf_model, finetune_layers):
    """Freeze all encoder layers except last `finetune_encoder_layers`"""
    hf_model.requires_grad_(False)
    n_layers = hf_model.config.num_hidden_layers
    for i in range(n_layers):
        hf_model.encoder.layer[i].requires_grad_(i>=n_layers-finetune_layers)
