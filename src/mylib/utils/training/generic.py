import lightning.pytorch as pl
from dataclasses import dataclass, asdict
from typing import Tuple
from torch import nn


@dataclass
class BaseLearnerConfig:
    lr: float = None
    batch_size: int = None
    warmup_period: int = None
    do_periodic_warmup: bool = False
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)


#! fix `get_parameter_group()`
class BaseLearner(pl.LightningModule):
    def get_default_learner_config(self):
        raise NotImplementedError()
    
    def on_train_start(self):
        model_hparams = self.model.get_hparams()
        model_hparams.update(asdict(self.config))
        self.logger.log_hyperparams(model_hparams)

    def get_parameter_groups(self):
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
        
        return optim_groups


class LightningCkptLoadable:
    """Mixin for `nn.Module`"""
    def load_checkpoint(self, path_to_ckpt, learner_class: BaseLearner, map_location=None):
        model = learner_class.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=self,
            config=learner_class.get_default_learner_config(),
        ).model
        
        self.load_state_dict(model.state_dict())


def freeze_hf_model(hf_model, finetune_layers):
    """Freeze all encoder layers except last `finetune_encoder_layers`"""
    hf_model.requires_grad_(False)
    n_layers = hf_model.config.num_hidden_layers
    for i in range(n_layers):
        hf_model.encoder.layer[i].requires_grad_(i>=n_layers-finetune_layers)


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


def train(learner, train_loader, val_loader, args):
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    import os

    if args.logger != 'none':
        checkpoint_callback = ModelCheckpoint(
            monitor='val_metric',
            save_last=True,
            save_top_k=1,
            mode='max',
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback, lr_monitor]
    else:
        callbacks = None

    import lightning.pytorch as pl
    if args.logger == 'tb':
        Logger = pl.loggers.TensorBoardLogger
        suffix = 'tensorboad'
    elif args.logger == 'wb':
        Logger = pl.loggers.WandbLogger
        suffix = 'wandb'
    elif args.logger == 'none':
        Logger = lambda **kwargs: False
        suffix = ''
    
    logger = Logger(
        save_dir=os.path.join(os.environ['ROOT_DIR'], 'logs', suffix),
        name=args.name
    )

    trainer = pl.Trainer(
        # max_epochs=1,
        max_time={'hours': 24},
        
        # max_time={'minutes': 10},
        # max_steps=0,

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        precision="16-mixed",

        # logging and checkpointing
        val_check_interval=args.interval,
        # check_val_every_n_epoch=1,
        logger=logger,
        enable_progress_bar=False,
        profiler=None,
        callbacks=callbacks,
        # log_every_n_steps=5,

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=3,
        num_sanity_val_steps=3
    )

    if args.resume_from is None:
        trainer.validate(learner, val_loader)

    from datetime import datetime
    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.fit(
        learner, train_loader, val_loader,
        ckpt_path=args.resume_from
    )

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.validate(learner, val_loader)


def get_argparser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', dest='name', default=None)
    ap.add_argument('--cuda', dest='cuda', default='0')
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    ap.add_argument('--interval', dest='interval', required=True, type=int)
    ap.add_argument('--logger', dest='logger', choices=['none', 'tb', 'wb'], default='none')
    ap.add_argument('--resume-training-from', dest='resume_from', default=None)
    ap.add_argument('--load-weights-from', dest='weights_from', default=None)
    return ap


def init_environment(args):
    import torch
    torch.set_float32_matmul_precision('medium')

    seed_everything(args.seed)

    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


def seed_everything(seed: int):
    """https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964"""
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
