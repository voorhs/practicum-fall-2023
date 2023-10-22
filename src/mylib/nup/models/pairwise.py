import torch.nn as nn
from .train_utils import LightningCkptLoadable, HParamsPuller
import torch
import torch.nn.functional as F
import numpy as np
from .aux import mySentenceTransformer, Projector


class ChainCosine(nn.Module, LightningCkptLoadable, HParamsPuller):
    def __init__(self, target_encoder: mySentenceTransformer, context_encoder, projection_size, context_size, temperature=1, k=1, hard_negative=False):
        super().__init__()

        self.projection_size = projection_size
        self.context_size = context_size
        self.temperature = temperature
        self.k = k
        self.hard_negative = hard_negative

        self.target_encoder = target_encoder
        self.context_encoder = context_encoder
        
        self.context_projector = Projector(
            input_size=self.context_encoder.get_encoding_size(),
            output_size=self.projection_size
        )
        self.target_projector = Projector(
            input_size=self.target_encoder.get_encoding_size(),
            output_size=self.projection_size
        )

    @property
    def device(self):
        return self.target_encoder.model.device
    
    def get_encodings(self, batch):
        if self.context_size is None:
            context_slice = slice(None, None, None)
        else:
            context_slice = slice(-self.context_size, None, None)

        context_batch = []
        target_batch = []
        for pair in batch:
            context_batch.append(pair['context'][context_slice])
            target_batch.append(pair['target'])
        
        target_encodings = self.target_encoder(target_batch)
        context_encodings = self.context_encoder(context_batch)

        context_encodings = self.context_projector(context_encodings)
        target_encodings = self.target_projector(target_encodings)

        return context_encodings, target_encodings
    
    def get_logits(self, batch, temperature):
        context_encodings, target_encodings = self.get_encodings(batch)
        return context_encodings @ target_encodings.T / temperature
    
    def forward(self, batch):
        logits = self.get_logits(batch, self.temperature)
        
        if self.hard_negative:
            mask = torch.eye(len(batch), dtype=torch.bool, device=logits.device)
            
            positive = torch.exp(logits[mask])
            negative = torch.exp(logits[~mask]).view(len(batch), len(batch) - 1)
            negative = (negative * negative).sum(dim=-1) / negative.mean(dim=-1)
            loss_r = (positive / (negative + positive)).log().neg().mean()

            positive = torch.exp(logits.T[mask])
            negative = torch.exp(logits.T[~mask]).view(len(batch), len(batch) - 1)
            negative = (negative * negative).sum(dim=-1) / negative.mean(dim=-1)
            loss_c = (positive / (negative + positive)).log().neg().mean()
        else:
            labels = torch.arange(len(batch), device=logits.device)
            loss_r = F.cross_entropy(logits, labels, reduction='mean')
            loss_c = F.cross_entropy(logits.T, labels, reduction='mean')
        loss = (loss_r + loss_c) / 2

        topk_accuracy = [i in top for i, top in enumerate(torch.topk(logits, k=self.k, dim=1).indices)]
        topk_accuracy = np.mean(topk_accuracy)

        return loss, topk_accuracy

    @torch.no_grad()
    def score(self, dialogue, temperature=1):
        batch = self.make_batch_from_dia(dialogue)
        logits = self.get_logits(batch, temperature)
        return F.softmax(logits, dim=1).diag().log10().mean().cpu().item()
    
    @staticmethod
    def make_batch_from_dia(dialogue):
        batch = []
        for i in range(1, len(dialogue)):
            batch.append({
                'context': dialogue[:i],
                'target': dialogue[i]
            })
        return batch
        

class TargetEncoder(nn.Module, HParamsPuller):
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
        return self.sentence_encoder.model.device
    
    def get_encoding_size(self):
        return self.speaker_embedding_dim + self.sentence_encoder.get_sentence_embedding_size()


class ContextEncoderConcat(nn.Module, HParamsPuller):
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


class ContextEncoderEMA(nn.Module, HParamsPuller):
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


class ContextEncoderDM(nn.Module, HParamsPuller):
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


class ChainCosine2(ChainCosine):
    def get_encodings(self, batch):
        context_slice = slice(-self.context_size, None, None)
        target_slice = slice(-self.context_size+1, None, None)

        context_batch = []
        target_batch = []
        for pair in batch:
            context_batch.append(pair['context'][context_slice])
            target_batch.append(pair['context'][target_slice] + [pair['target']])
        
        target_encodings = self.target_encoder(target_batch)
        context_encodings = self.context_encoder(context_batch)

        context_encodings = self.context_projector(context_encodings)
        target_encodings = self.target_projector(target_encodings)

        return context_encodings, target_encodings
    
    @staticmethod
    def make_batch_from_dia(dialogue):
        batch = []
        for i in range(1, len(dialogue)):
            batch.append({
                'context': dialogue[:i],
                'target': dialogue[i]
            })
        return batch
   