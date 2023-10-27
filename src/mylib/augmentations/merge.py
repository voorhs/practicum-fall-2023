from ..modeling.dialogue import UtteranceTransformerDMConfig, UtteranceTransformerDM
from ..modeling.listwise import DecoderUtteranceSorter, Decoder
import numpy as np
from tqdm import tqdm


def load_listwise_decoder(ckpt_path, device):
    head_dropout_prob = 0.02
    encoder_name = 'sentence-transformers/all-mpnet-base-v2'
    config = UtteranceTransformerDMConfig(
        num_attention_heads=4,
        attention_probs_dropout_prob=0.02,
        n_layers=4,
        encoder_name=encoder_name,
        embed_turn_ids=False,
        is_casual=False
    )
    _dialogue_model = UtteranceTransformerDM(config)

    _model = DecoderUtteranceSorter(
        dialogue_model=_dialogue_model,
        dropout_prob=head_dropout_prob,
        max_n_uts=20,
        decoder=Decoder(top_k=2)
    )

    return DecoderUtteranceSorter.from_checkpoint(
        path_to_ckpt=ckpt_path,
        model=_model,
        map_location=device
    )


class ListwiseShuffler:
    def __init__(
            self,
            decoder=Decoder(),
            ckpt_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/listwie-decoder-symmetric-t1/checkpoints/last.ckpt',
            device='cpu',
            thresh=-np.inf
        ):
        self.decoder = decoder
        self.thresh = thresh
        self.model = load_listwise_decoder(ckpt_path, device)

    def __call__(self, dialogues, batch_size=192):
        aug_dialogues_with_scores = []
        for i_batch in tqdm(range(0, len(dialogues), batch_size), desc='shuffling batches'):
            start = i_batch
            end = i_batch + batch_size
            batch = dialogues[start:end]
            aug_dialogues_with_scores.extend(self.model.augment(batch, self.decoder))
        
        return [aug if score >= self.thresh else None for aug, score in aug_dialogues_with_scores]
