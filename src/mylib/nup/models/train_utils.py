from torch.utils.data import Dataset
from typing import Literal, Tuple
import math
import json
from dataclasses import dataclass, field
import lightning.pytorch as pl
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


class NUPDataset(Dataset):
    chunk_size = 2048
    def __init__(self, path, split: Literal['train', 'test', 'val'], fraction=1.):
        self.split = split
        self.path = path

        if split == 'train':
            max_n_chunks = 2556
        elif split == 'test' or split == 'val':
            max_n_chunks = 141

        if isinstance(fraction, float):
            self.fraction = min(1., max(0., fraction))
            self.n_chunks = math.ceil(self.fraction * max_n_chunks)
        elif isinstance(fraction, int):
            self.fraction = min(max_n_chunks, max(1, fraction))
            self.n_chunks = fraction
        else:
            raise ValueError('fraction must be int or float')

        self.len = self.n_chunks * self.chunk_size
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one dialogue, represented with an object of the following schema:
        ```
        {
            "type": "object",
            "properties":
            {
                "context":
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
                },
                "target":
                {
                    "type": "object",
                    "properties":
                    {
                        "utterance": {"type": "string"},
                        "speaker": {"type": "number"}
                    }
                }
            }
        }
        ```"""
        i_chunk = math.floor(i / self.chunk_size)
        idx_within_chunk = i % self.chunk_size
        item = json.load(open(f'{self.path}/pairs/{self.split}/{i_chunk}.json', 'r'))[idx_within_chunk]
        return item


class DialogueDataset(Dataset):
    chunk_size = 512
    def __init__(self, path, split: Literal['train', 'test', 'val'], fraction=1.):
        self.split = split
        self.path = path

        if split == 'train':
            max_n_chunks = 880
        elif split == 'test' or split == 'val':
            max_n_chunks = 48

        if isinstance(fraction, float):
            self.fraction = min(1., max(0., fraction))
            self.n_chunks = math.ceil(self.fraction * max_n_chunks)
        elif isinstance(fraction, int):
            self.fraction = min(max_n_chunks, max(1, fraction))
            self.n_chunks = fraction
        else:
            raise ValueError('fraction must be int or float')

        self.len = self.n_chunks * self.chunk_size
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        """
        Loads one chunk and returns one dialogue, represented with an object of the following schema:
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
        i_chunk = math.floor(i / self.chunk_size)
        idx_within_chunk = i % self.chunk_size
        item = json.load(open(f'{self.path}/dialogues/{self.split}/{i_chunk}.json', 'r'))[idx_within_chunk]
        return item


@dataclass
class LearnerConfig:
    kwargs: field(default_factory=dict) = None,
    lr: float = None
    batch_size: int = None
    warmup_period: int = None
    do_periodic_warmup: bool = False
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)


class Learner(pl.LightningModule):
    def __init__(self, model, config: LearnerConfig):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, batch):
        return self.model(batch)

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
    
    def validation_step(self, batch, batch_idx):
        loss, metric = self.forward(batch)
        self.log(
            name='val_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        self.log(
            name='val_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
    
    def on_train_start(self):
        optim_hparams = self.optimizers().defaults
        model_hparams = self.model.get_hparams()
        model_hparams.update(optim_hparams)
        model_hparams['batch size'] = self.config.batch_size
        model_hparams['warmup period'] = self.config.warmup_period
        model_hparams['do periodic warmup'] = self.config.do_periodic_warmup
        model_hparams.update(self.config.kwargs)
        self.logger.log_hyperparams(model_hparams)

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


class LightningCkptLoadable:
    @staticmethod
    def from_checkpoint(path_to_ckpt, model, learner_config=LearnerConfig(), map_location=None):
        return Learner.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=model,
            config=learner_config,
        ).model


class HParamsPuller:
    def get_hparams(self):
        res = {}
        for attr, val in vars(self).items():
            if hasattr(val, 'get_hparams'):
                tmp = val.get_hparams()
                tmp = self.add_prefix(tmp, attr)
                res.update(tmp)
            elif isinstance(val, (int, float, str, bool)):
                res[attr] = val
        return res
    
    @staticmethod
    def add_prefix(dct, prefix):
        res = {}
        for key, val in dct.items():
            res[f'{prefix}.{key}'] = val
        return res
