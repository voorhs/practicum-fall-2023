from ...utils.training import HParamsPuller
from ...modeling.aux import mySentenceTransformer
import torch
from torch import nn
import torch.nn.functional as F


class BaseContextEncoder(nn.Module, HParamsPuller):
    context_size: int
    
    def get_encoding_size(self):
        raise NotImplementedError()


class ContextEncoderConcat(BaseContextEncoder):
    def __init__(self, sentence_encoder: mySentenceTransformer, context_size, n_speakers=2, speaker_embedding_dim=8):
        super().__init__()

        self.sentence_encoder = sentence_encoder
        self.context_size = context_size
        self.speaker_embedding_dim = speaker_embedding_dim

        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)

    def forward(self, batch):
        uts = []
        lens = []
        spe = []
        for dia in batch:
            cur_uts = [item['utterance'] for item in dia]
            spe.append(dia[-1]['speaker'])
            uts.extend(cur_uts)
            lens.append(len(cur_uts))
        
        sentence_embeddings = self.sentence_encoder(uts)
        speaker_embeddings = self.speaker_embedding(torch.tensor(spe, dtype=torch.int, device=sentence_embeddings[0].device))
        d = self.sentence_encoder.get_sentence_embedding_size()
        res = []
        for i in range(len(batch)):
            start = sum(lens[:i])
            end = start + lens[i]
            n_zeros_to_pad = (self.context_size - lens[i]) * d
            enc = F.pad(torch.cat(sentence_embeddings[start:end] + [speaker_embeddings[i]]), pad=(n_zeros_to_pad, 0), value=0)
            res.append(enc)
        
        return res

    def get_encoding_size(self):
        return self.sentence_encoder.get_sentence_embedding_size() * self.context_size + self.speaker_embedding_dim


class ContextEncoderEMA(BaseContextEncoder):
    def __init__(self, sentence_encoder: mySentenceTransformer, context_size, tau):
        super().__init__()

        self.sentence_encoder = sentence_encoder
        self.context_size = context_size
        self.tau = tau

    def forward(self, batch):
        uts = []
        lens = []
        for dia in batch:
            cur_uts = [item['utterance'] for item in dia]
            uts.extend(cur_uts)
            lens.append(len(cur_uts))
        
        sentence_embeddings = self.sentence_encoder(uts)
        return self._ema(sentence_embeddings, lens, self.tau)
    
    @staticmethod
    def _ema(sentence_embeddings, lens, tau):
        res = []
        for i in range(len(lens)):
            start = sum(lens[:i])
            end = start + lens[i]
            embs = sentence_embeddings[start:end]
            
            last_ut = embs[-1]

            if lens[i] > 1:
                prev_uts = embs[-2]
            else:
                prev_uts = torch.zeros_like(last_ut)
            for prev_ut in embs[-3:-lens[i]-1:-1]:
                prev_uts = tau * prev_uts + (1 - tau) * prev_ut
            
            res.append(torch.cat([prev_uts, last_ut]))
        
        return res

    def get_encoding_size(self):
        return 2 * self.sentence_encoder.get_sentence_embedding_size()


class ContextEncoderDM(BaseContextEncoder):
    def __init__(self, dialogue_model, tau):
        super().__init__()
        self.tau = tau
        self.dialogue_model = dialogue_model
    
    def forward(self, batch):
        hidden_states = self.dialogue_model(batch)
        
        lens = [len(context) for context in batch]
        
        sentence_embeddings = []
        for hs, length in zip(hidden_states, lens):
            sentence_embeddings.extend(torch.unbind(hs, dim=0)[:length])
        encodings = ContextEncoderEMA._ema(sentence_embeddings, lens, self.tau)
        
        return encodings

    def get_encoding_size(self):
        return 2 * self.dialogue_model.model.config.hidden_size

