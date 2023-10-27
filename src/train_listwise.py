if __name__ == "__main__":
    
    # ======= DEFINE TASK =======  

    from mylib.utils.training import get_argparser, init_environment
    ap = get_argparser()
    ap.add_argument('--model', dest='model', required=True, choices=[
        'listwise-utterance-transformer',
        'listwise-sparse-transformer',
        'listwise-hssa',
    ])
    args = ap.parse_args()

    init_environment(args)

    # ======= DEFINE MODEL =======

    mpnet_name = 'sentence-transformers/all-mpnet-base-v2'
    amazon_name = 'aws-ai/dse-bert-large'

    from mylib.utils.training import freeze_hf_model
    from mylib.modeling.dialogue import (
        UtteranceTransformerDM,
        UtteranceTransformerDMConfig,
        SparseTransformerDM,
        HSSADM,
        HSSAConfig
    )
    from mylib.modeling.listwise import RankerListwise  #, ClassifierListwise, DecoderListwise
    from mylib.utils.training.listwise import ListwiseLearner, ListwiseLearnerConfig

    if args.model == 'listwise-utterance-transformer':
        learner_config = ListwiseLearnerConfig(
            batch_size=128,
            warmup_period=200,
            do_periodic_warmup=False,
            lr=3e-5,
            finetune_layers=3,
            train_fraction=1.,
            val_fraction=0.2
        )

        config = UtteranceTransformerDMConfig(
            num_attention_heads=4,
            attention_probs_dropout_prob=0.02,
            n_layers=4,
            encoder_name=amazon_name,
            embed_turn_ids=False,
            is_casual=False
        )
        dialogue_model = UtteranceTransformerDM(config)
        freeze_hf_model(dialogue_model.encoder.model, learner_config.finetune_layers)
        
        model = RankerListwise(
            dialogue_model=dialogue_model,
            dropout_prob=0.02
        ) 
    elif args.model == 'listwise-sparse-transformer':
        learner_config = ListwiseLearnerConfig(
            batch_size=192,
            warmup_period=200,
            do_periodic_warmup=False,
            lr=3e-5,
            finetune_layers=3,
            train_fraction=1.,
            val_fraction=0.2
        )

        dialogue_model = SparseTransformerDM(amazon_name)
        freeze_hf_model(dialogue_model.model, learner_config.finetune_layers)
        
        model = RankerListwise(
            dialogue_model=dialogue_model,
            dropout_prob=0.02
        )
    elif args.model == 'listwise-hssa':
        learner_config = ListwiseLearnerConfig(
            batch_size=192,
            warmup_period=200,
            do_periodic_warmup=False,
            lr=3e-5,
            finetune_layers=3,
            train_fraction=1.,
            val_fraction=0.2
        )

        config = HSSAConfig()

        dialogue_model = HSSADM(
            hf_model_name=amazon_name,
            config=config,
            pool_utterance_level=True
        )
        freeze_hf_model(dialogue_model.model, learner_config.finetune_layers)
        
        model = RankerListwise(
            dialogue_model=dialogue_model,
            dropout_prob=0.02
        )
    
    # ======= DEFINE LEARNER =======

    if args.weights_from is not None:
        learner = ListwiseLearner.load_from_checkpoint(
            checkpoint_path=args.weights_from,
            model=model,
            config=learner_config
        )
    else:
        learner = ListwiseLearner(model, learner_config)

    # ======= DEFINE DATA =======

    def collate_fn(batch):
        return batch
    
    import os
    root_dir = os.environ['ROOT_DIR']
    path = os.path.join(root_dir, 'data', 'source')

    from mylib.utils.data import DialogueDataset
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset=DialogueDataset(
            path=os.path.join(path, 'train'),
            fraction=learner_config.train_fraction
        ),
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=DialogueDataset(
            path=os.path.join(path, 'val'),
            fraction=learner_config.val_fraction
        ),
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    # ======= TRAIN =======

    from mylib.utils.training import train

    train(learner, train_loader, val_loader, args)
