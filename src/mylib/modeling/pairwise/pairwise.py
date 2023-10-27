import torch.nn as nn
from ...utils.training import LightningCkptLoadable, HParamsPuller
import torch
import torch.nn.functional as F
import numpy as np
from ...modeling.aux import Projector
from .context_encoders import BaseContextEncoder
from .target_encoders import BaseTargetEncoder


class Pairwise(nn.Module, LightningCkptLoadable, HParamsPuller):
    def __init__(
            self,
            target_encoder: BaseTargetEncoder,
            context_encoder: BaseContextEncoder,
            projection_size,
            temperature=1,
            k=1,
            hard_negative=False
        ):
        super().__init__()

        self.projection_size = projection_size
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
        return self.target_encoder.device
    
    def get_encodings(self, batch):
        if self.context_encoder.context_size is None:
            context_slice = slice(None, None, None)
        else:
            context_slice = slice(-self.context_encoder.context_size, None, None)

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
        

class SymmetricPairwise(Pairwise):
    def get_encodings(self, batch):
        context_slice = slice(-self.context_encoder.context_size, None, None)
        target_slice = slice(-self.context_encoder.context_size+1, None, None)

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
   