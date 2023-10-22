from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing import List
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import math


class mySentenceTransformer(nn.Module):
    """Imitation of SentenceTransformers (https://www.sbert.net/)"""

    def __init__(
            self,
            model_name='sentence-transformers/all-mpnet-base-v2',
            model=None,
            pooling=True
        ):
        """If `pooling=False`, then instead of sentence embeddings forward will return list of token embeddings."""
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling

        if model is None:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, sentences: List[str]) -> List[torch.Tensor]:
        input = self.tokenizer(sentences, padding='longest', return_tensors='pt').to(self.model.device)
        output = self.model(
            input_ids=input['input_ids'],
            attention_mask=input['attention_mask']
        )
        
        res = []
        for token_emb, attention in zip(output.last_hidden_state, input['attention_mask']):
            last_mask_id = len(attention)-1
            while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                last_mask_id -= 1
            embs = token_emb[:last_mask_id+1]
            if self.pooling:
                embs = torch.mean(embs, dim=0)
                embs = embs / torch.linalg.norm(embs)
            res.append(embs)

        return res

    def get_sentence_embedding_size(self):
        return self.model.config.hidden_size


class Projector(nn.Module):
    """Fully-Connected 2-layer Linear Model. Taken from linking prediction paper code."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, input_size)
        self.linear_2 = nn.Linear(input_size, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.final = nn.Linear(input_size, output_size)
        # self.orthogonal_initialization()

    def orthogonal_initialization(self):
        for l in [self.linear_1, self.linear_2]:
            torch.nn.init.orthogonal_(l.weight)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.stack(x)
        else:
            x = x.to(torch.float32)
        # x = x.cuda()
        x = x + F.gelu(self.linear_1(self.norm1(x)))
        x = x + F.gelu(self.linear_2(self.norm2(x)))

        return F.normalize(self.final(x), dim=-1)


@dataclass
class myTransformerConfig:
    hidden_size: int = None
    num_attention_heads: int = None
    attention_probs_dropout_prob: float = None
    intermediate_size: int = None
    n_layers: int = None


class mySelfAttention(nn.Module):
    def __init__(
            self,
            config: myTransformerConfig
        ):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.config = config

        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.norm = nn.LayerNorm(config.hidden_size)
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        change view from (B, T, H) to (B, n, T, h)
        - B batch size
        - T longest sequence size
        - H hidden size
        - n number of att heads
        - h single att head size
        """
        new_x_shape = x.size()[:-1] + (self.config.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask):
        """
        x: (B, T, H)
        attention_mask: (B, T) or (B, T, T), if 0 then ignore corresponding token
        """
        # (B, T, H)
        hidden_states = self.norm(x)

        # (B, T, H)
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        # (B, n, T, h)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif len(attention_mask.shape) == 3:
            attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(f'strange shape of attention mask: {attention_mask.shape}')

        # (B, n, T, T)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores.masked_fill(attention_mask==0, -torch.inf)

        # (B, n, T, T)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # (B, n, T, h)
        c = torch.matmul(attention_probs, v)

        # (B, T, H)
        c = c.permute(0, 2, 1, 3).contiguous()
        new_c_shape = c.size()[:-2] + (self.config.hidden_size,)
        c = c.view(*new_c_shape)

        # (B, T, H)
        return x + self.o(c)


class myFFBlock(nn.Module):
    def __init__(self, config: myTransformerConfig):
        super().__init__()
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.nonlinear = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, x):
        return x + self.linear2(self.nonlinear(self.linear1(self.norm(x))))


class myTransformerBlock(nn.Module):
    def __init__(self, config: myTransformerConfig):
        super().__init__()
        
        self.att = mySelfAttention(config)
        self.ff = myFFBlock(config)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attention_mask):
        x = self.att(x, attention_mask)
        x = self.ff(x)
        return self.norm(x)
