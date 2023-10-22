from sentence_transformers import SentenceTransformer
import numpy as np


def sentence_encoder(utterances):
    """
    Encodes `utterances` with sentence_transformers library and saves embeddings to .npy file.

    Params
    ------
        utterances: list[str], all utterances from dataset
        path: str, where to save .npy file
    """
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1').to('cuda')
    return model.encode(utterances, show_progress_bar=True)


if __name__ == '__main__':
    import json

    utterances = json.load(open('clust-data/utterances.json', 'r'))
    
    np.save('clust-data/sentence_embeddings', sentence_encoder(utterances))