if __name__ == "__main__":
    
    # ======= DEFINE TASK =======

    from mylib.utils.training import get_argparser, init_environment
    ap = get_argparser()
    ap.add_argument('--model', dest='model', required=True, choices=[
        'pairwise-cat',
        'pairwise-ema',
        'pairwise-sparse-transformer',
        'pairwise-symmetric-ema'
    ])
    args = ap.parse_args()

    init_environment(args)

    # ======= DEFINE MODEL =======

    mpnet_name = 'sentence-transformers/all-mpnet-base-v2'
    amazon_name = 'aws-ai/dse-bert-large'

    from mylib.utils.training import freeze_hf_model
    from mylib.modeling.aux import mySentenceTransformer
    from mylib.modeling.pairwise import (
        TargetEncoder,
        ContextEncoderConcat,
        ContextEncoderEMA,
        ContextEncoderDM,
        Pairwise,
        SymmetricPairwise
    )
    from mylib.utils.training.pairwise import PairwiseLearner, PairwiseLearnerConfig
    from mylib.modeling.dialogue import SparseTransformerDM

    if args.model == 'pairwise-cat':
        learner_config = PairwiseLearnerConfig(
            batch_size=128,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=7e-6,
            finetune_layers=3,
            k=5,
            temperature=0.05
        )

        encoder = mySentenceTransformer(amazon_name)
        freeze_hf_model(encoder.model, learner_config.finetune_layers)
        target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderConcat(encoder, context_size=3)
        model = Pairwise(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=256,
            k=learner_config.k,
            temperature=learner_config.temperature,
            hard_negative=False
        )
    elif args.model == 'pairwise-symmetric-ema':
        learner_config = PairwiseLearnerConfig(
            batch_size=128,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=7e-6,
            finetune_layers=3,
            k=5,
            temperature=0.05
        )

        encoder = mySentenceTransformer(amazon_name)
        freeze_hf_model(encoder.model, learner_config.finetune_layers)
        target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderEMA(encoder, context_size=3, tau=0.5)
        model = SymmetricPairwise(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=256,
            k=learner_config.k,
            temperature=learner_config.temperature,
            hard_negative=False
        )
    elif args.model == 'pairwise-ema':
        learner_config = PairwiseLearnerConfig(
            batch_size=128,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=7e-6,
            finetune_layers=3,
            k=5,
            temperature=0.05
        )

        encoder = mySentenceTransformer(amazon_name)
        freeze_hf_model(encoder.model, learner_config.finetune_layers)
        target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderEMA(encoder, context_size=3, tau=0.5)
        model = Pairwise(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=256,
            k=learner_config.k,
            temperature=learner_config.temperature,
            hard_negative=False
        )
    elif args.model == 'pairwise-sparse-transformer':
        learner_config = PairwiseLearnerConfig(
            batch_size=128,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=7e-6,
            finetune_layers=3,
            k=5,
            temperature=0.05
        )

        dialogue_model = SparseTransformerDM(amazon_name)
        freeze_hf_model(dialogue_model.model, learner_config.finetune_layers)

        context_encoder = ContextEncoderDM(dialogue_model, tau=0.5)
        encoder = mySentenceTransformer(amazon_name, model=dialogue_model.model)
        target_encoder = TargetEncoder(encoder)
        model = Pairwise(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=256,
            context_size=5
        )
    
    # ======= DEFINE LEARNER =======

    if args.weights_from is not None:
        learner = PairwiseLearner.load_from_checkpoint(
            checkpoint_path=args.weights_from,
            model=model,
            config=learner_config
        )
    else:
        learner = PairwiseLearner(model, learner_config)

    # ======= DEFINE DATA =======

    def collate_fn(batch):
        return batch
    
    import os
    root_dir = os.environ['ROOT_DIR']
    path = os.path.join(root_dir, 'data', 'train', 'context-response-pairs')

    from mylib.utils.data import ContextResponseDataset
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset=ContextResponseDataset(path, 'train', fraction=learner_config.train_fraction),
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=ContextResponseDataset(path, 'val', fraction=learner_config.val_fraction),
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    # ======= TRAIN =======

    from mylib.utils.training import train

    train(learner, train_loader, val_loader, args)
