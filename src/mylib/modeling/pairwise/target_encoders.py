from ...utils.training import HParamsPuller
from ...modeling.aux import mySentenceTransformer
import torch
from torch import nn


class BaseTargetEncoder(nn.Module, HParamsPuller):
    def get_encoding_size(self):
        raise NotImplementedError()

    @property
    def device(self):
        raise NotImplementedError()
    

class TargetEncoder(BaseTargetEncoder):
    def __init__(self, sentence_encoder: mySentenceTransformer, n_speakers=2, speaker_embedding_dim=8):
        super().__init__()

        self.sentence_encoder = sentence_encoder
        self.n_speakers = n_speakers
        self.speaker_embedding_dim = speaker_embedding_dim

        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
    
    def forward(self, batch):
        uts = [item['utterance'] for item in batch]
        spe = [item['speaker'] for item in batch]
        
        sentence_embeddings = self.sentence_encoder(uts)
        speaker_ids = torch.tensor(spe, device=self.device)
        speaker_embeddings = self.speaker_embedding(speaker_ids)
        return torch.cat([torch.stack(sentence_embeddings), speaker_embeddings], dim=1)

    @property
    def device(self):
        return self.sentence_encoder.device
    
    def get_encoding_size(self):
        return self.speaker_embedding_dim + self.sentence_encoder.get_sentence_embedding_size()

