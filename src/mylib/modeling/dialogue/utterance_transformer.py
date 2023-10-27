from dataclasses import dataclass, asdict
from ..aux import myTransformerConfig, mySentenceTransformer, Projector, myTransformerBlock
from torch import nn
import torch
from .base_dialogue_model import BaseDialogueModel


@dataclass
class UtteranceTransformerDMConfig(myTransformerConfig):
    encoder_name: str = None
    max_dia_len: int = None
    embed_turn_ids: bool = True
    is_casual: bool = False


class UtteranceTransformerDM(nn.Module, BaseDialogueModel):
    def __init__(self, config: UtteranceTransformerDMConfig):
        super().__init__()

        self.config = config
        self.encoder = mySentenceTransformer(model_name=config.encoder_name)

        sentence_embedding_dimension = self.encoder.get_sentence_embedding_size()
        projection_size = sentence_embedding_dimension // 2
        if self.config.embed_turn_ids:
            self.hidden_size = projection_size + 16
            self.turn_ids_embedding = nn.Embedding(config.max_dia_len, 8)
        else:
            self.hidden_size = projection_size + 8

        self.speaker_embeddings = nn.Embedding(2, 8)
        self.projector = Projector(sentence_embedding_dimension, projection_size)

        config.hidden_size = self.hidden_size
        config.intermediate_size = 4 * self.hidden_size
        self.transformer = nn.ModuleList([myTransformerBlock(config) for _ in range(config.n_layers)])
        
    def get_hparams(self):
        return asdict(self.config)

    @property
    def device(self):
        return self.encoder.device

    def forward(self, batch):
        dia_lens = [len(dia) for dia in batch]

        inputs = []
        for dia in batch:
            speaker_ids = torch.tensor([item['speaker'] for item in dia], device=self.device, dtype=torch.long)
            speaker_embeddings = self.speaker_embeddings(speaker_ids)

            sentence_embeddings = self.encoder([item['utterance'] for item in dia])
            sentence_embeddings = self.projector(sentence_embeddings)

            utterance_embeddings = [sentence_embeddings, speaker_embeddings]
            
            if self.config.embed_turn_ids:
                turn_ids = torch.arange(len(dia), device=self.device)
                turn_ids_embeddings = self.turn_ids_embedding(turn_ids)
                utterance_embeddings += [turn_ids_embeddings]
            
            utterance_embeddings = torch.cat(utterance_embeddings, dim=1)
            utterance_embeddings = torch.unbind(utterance_embeddings, dim=0)
            inputs.append(utterance_embeddings)
        
        T = max(dia_lens)
        if self.config.is_casual:
            attention_mask = []
            for length in dia_lens:
                mask = torch.tril(torch.ones(T, T, device=self.device))
                flat = torch.tensor(length * [1] + (T-length) * [0], device=self.device)
                mask *= flat.view(1, -1)
                attention_mask.append(mask)
            attention_mask = torch.stack(attention_mask, dim=0)
        else:
            attention_mask = torch.tensor([length * [1] + (T-length) * [0] for length in dia_lens], device=self.device)
        
        padded_inputs = torch.stack([torch.stack(inp + (T-len(inp)) * (inp[0].new_zeros(self.hidden_size),)) for inp in inputs])

        # (B, T, H)
        hidden_states = padded_inputs
        for layer in self.transformer:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states

    def get_hidden_size(self):
        return self.hidden_size
