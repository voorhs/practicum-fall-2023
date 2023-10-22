if __name__ == "__main__":
    import torch
    torch.set_float32_matmul_precision('medium')    

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', dest='name', default=None)
    ap.add_argument('--cuda', dest='cuda', default='0')
    ap.add_argument('--interval', dest='interval', required=True, type=int)
    args = ap.parse_args()

    # from dataclasses import dataclass
    # @dataclass
    # class Args:
    #     name = None
    #     cuda = '2'
    #     interval = 100
    # args = Args()

    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    import sys
    sys.path.insert(0, '/home/alekseev_ilya/dialogue-augmentation/nup')
    from models import SimpleDialogueEncoder
    from train_utils import freeze_hf_model, Learner, LearnerConfig, ContrastiveDataset, MultiWOZServiceClfDataset

    hf_model = 'roberta-base'
    finetune_layers = 1

    learner_config = LearnerConfig(
        k=5,
        t=0.1,
        batch_size=32,
        # warmup_period=200,
        do_periodic_warmup=False,
        lr=3e-6,
        kwargs={
            'finetune_layers': finetune_layers
        }
    )
    
    model = SimpleDialogueEncoder(hf_model)
    freeze_hf_model(model.model, finetune_layers)

    learner = Learner(model, learner_config)

    # ======= DEFINE DATA =======

    path = '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/dataset/train'
    dataset = ContrastiveDataset(path)
    
    dir = '/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/multiwoz_truncated'
    multiwoz_train = MultiWOZServiceClfDataset(
        path=f'{dir}/train',
        fraction=1.
    )
    multiwoz_val = MultiWOZServiceClfDataset(
        path=f'{dir}/validation',
        fraction=1.
    )

    from torch.utils.data import DataLoader
    def collate_fn(batch):
        return batch
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )
    multiwoz_train_loader = DataLoader(
        dataset=multiwoz_train,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )
    multiwoz_val_loader = DataLoader(
        dataset=multiwoz_val,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    # val_loader = DataLoader(
    #     dataset=dataset('.', 'val', fraction=.2),
    #     batch_size=learner_config.batch_size,
    #     shuffle=False,
    #     num_workers=3,
    #     collate_fn=collate_fn
    # )

    # ======= DEFINE TRAINER =======

    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric',
        save_last=True,
        save_top_k=1,
        mode='max',
        # every_n_train_steps=2000
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    import lightning.pytorch as pl
    logger = pl.loggers.TensorBoardLogger(
        save_dir='.',
        version=args.name,
        name='/home/alekseev_ilya/dialogue-augmentation/dialogue_encoder/logs/'
    )

    trainer = pl.Trainer(
        # max_epochs=1,
        max_time={'hours': 24},
        
        # max_time={'minutes': 5},
        # max_steps=0,

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        precision="16-mixed",

        # logging and checkpointing
        # val_check_interval=args.interval,
        # check_val_every_n_epoch=1,
        logger=logger,
        enable_progress_bar=False,
        profiler=None,
        callbacks=[checkpoint_callback, lr_monitor],
        # log_every_n_steps=args.interval,

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    # ======= START TRAINING =======

    from datetime import datetime
    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # trainer.validate(learner, [multiwoz_train_loader, multiwoz_val_loader],)

    # do magic!
    trainer.fit(
        learner, train_loader, [multiwoz_train_loader, multiwoz_val_loader],
        # ckpt_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/listwise-clf/checkpoints/last.ckpt'
    )

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))