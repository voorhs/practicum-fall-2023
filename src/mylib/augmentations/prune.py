from ..modeling.pairwise import ChainCosine, TargetEncoder, ContextEncoderConcat
from ..modeling.aux import mySentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch
from tqdm import tqdm


def _load_pairwise_cat(ckpt_path, device):
    context_size = 3
    encoder_name = 'aws-ai/dse-bert-large'

    _encoder = mySentenceTransformer(encoder_name)
    _target_encoder = TargetEncoder(_encoder)
    _context_encoder = ContextEncoderConcat(_encoder, context_size=context_size)
    _model = ChainCosine(
        target_encoder=_target_encoder,
        context_encoder=_context_encoder,
        projection_size=256,
        context_size=context_size,
    )

    return ChainCosine.from_checkpoint(
        path_to_ckpt=ckpt_path,
        model=_model,
        map_location=device
    ).eval()


class Pruner:
    def __init__(
            self,
            ckpt_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/pairwise-cat-speaker-issue/checkpoints/last.ckpt',
            device='cpu',
            thresh=-np.inf
        ):
        self.thresh = thresh
        self.model = _load_pairwise_cat(ckpt_path, device)

    def __call__(self, dialogues):
        res = []
        for dia in tqdm(dialogues, desc='cutting dialogues'):
            aug, score = self._cut(self.model, dia)
            res.append(aug if score >= self.thresh else None)
        return res

    @staticmethod
    def _cut(model, dia):
        """drops all clusters except the biggest one. applies transformation only to dialogues with 6 utterances at least"""
        if len(dia) < 6:
            return None, -np.inf
        end = len(dia) // 3
        start = 2
        variations = []
        for n_clusters in range(start, end+1):
            clusterwise_uts = _cluster(model, dia, n_clusters)
            ids = clusterwise_uts[np.argmax([len(clust) for clust in clusterwise_uts])]
            aug = [dia[i] for i in ids]
            score = model.score(aug)
            variations.append((aug, score))
        res, score = max(variations, key=lambda x: x[1])
        return res, score


@torch.no_grad()
def _cluster(model, dia, n_clusters):
    """clusters utterances within dia according to logits (similarities) from pairwise model"""
    batch = model.make_batch_from_dia(dia)
    similarities = model.get_logits(batch, temperature=1).cpu().numpy()
    
    # mask out similarities between utterances of same speaker
    # speaker = [item['speaker'] for item in dia]
    # context_speaker = np.array(speaker[:-1])[:, None]
    # target_speaker = np.array(speaker[1:])[None, :]
    # mask = (context_speaker != target_speaker) | np.eye(len(speaker)-1, dtype=np.bool_)
    # similarities[~mask] = -1e3

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average',
        metric='precomputed'
    ).fit_predict(similarities)

    labels = np.r_[labels[0], labels]

    res = [[] for _ in range(len(np.unique(labels)))]
    for i_ut, lab in enumerate(labels):
        res[lab].append(i_ut)
    return res
