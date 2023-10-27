from torch import nn
from ...utils.training import HParamsPuller
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
import torch
from transformers.models.mpnet.modeling_mpnet import create_position_ids_from_input_ids
from .base_dialogue_model import BaseDialogueModel


class SparseTransformerDM(nn.Module, HParamsPuller, BaseDialogueModel):
    def __init__(self, hf_model_name):
        super().__init__()

        self.hf_model_name = hf_model_name

        self.model = AutoModel.from_pretrained(hf_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    @property
    def device(self):
        return self.model.device
    
    def _tokenize(self, batch, device):
        input_ids, uts_grouped_by_turn_tokenized, max_ut_lens = self._group_uts(batch)
        extended_attention_mask = self._extended_att_mask(
            len(batch),
            uts_grouped_by_turn_tokenized,
            max_ut_lens
        )
        position_ids = self._position_ids(uts_grouped_by_turn_tokenized)
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": extended_attention_mask.to(device),
            "position_ids": position_ids.to(device)
        }

    @staticmethod
    def _group_uts(tokenizer, batch):
        """group utterances by turns in order to tokenize and pad them jointly"""
        uts_grouped_by_turn = defaultdict(list)
        max_n_utterances = max(len(dia) for dia in batch)

        for dia in batch:
            for i, item in enumerate(dia):
                uts_grouped_by_turn[i].append(f"[{item['speaker']}] {item['utterance']}")
            for i in range(len(dia), max_n_utterances):
                uts_grouped_by_turn[i].append('')

        uts_grouped_by_turn_tokenized = [None for _ in range(max_n_utterances)]
        max_ut_lens = [None for _ in range(max_n_utterances)]
        for i, uts in uts_grouped_by_turn.items():
            tokens = tokenizer(uts, padding='longest', return_tensors='pt')
            uts_grouped_by_turn_tokenized[i] = tokens
            max_ut_lens[i] = tokens['input_ids'].shape[1]
    
        # (N_uts, T)
        input_ids = torch.cat([group['input_ids'] for group in uts_grouped_by_turn_tokenized], dim=1)
        return input_ids, uts_grouped_by_turn_tokenized, max_ut_lens

    @staticmethod
    def _extended_att_mask(
            batch_size,
            uts_grouped_by_turn_tokenized,
            max_ut_lens
        ):
        """mask of size (B, T, T)"""
        attention_mask = []
        for i in range(batch_size):
            masks_per_utterance = []
            for j, group in enumerate(uts_grouped_by_turn_tokenized):
                mask = group['attention_mask'][i]
                T = max_ut_lens[j]
                masks_per_utterance.append(mask[None, :].expand(T, T))
            attention_mask.append(torch.block_diag(*masks_per_utterance))
        attention_mask = torch.stack(attention_mask, dim=0)
        
        # allow CLS tokens attend to each other
        res = attention_mask
        for i in range(len(max_ut_lens)):
            cls_idx = sum(max_ut_lens[:i])
            res[:, :, cls_idx] = 1
        return res

    @staticmethod
    def _position_ids(
            uts_grouped_by_turn_tokenized,
            padding_idx=1   # padding_idx that is used in MPNet
        ): 
        """assign positions within each utterance"""
        position_ids = []
        for group in uts_grouped_by_turn_tokenized:
            ids = create_position_ids_from_input_ids(group['input_ids'], padding_idx=padding_idx)
            position_ids.append(ids)
        position_ids = torch.cat(position_ids, dim=1)
        
        return position_ids
    
    def forward(self, batch):
        device = self.device

        inputs, uts_lens = self._tokenize(batch, device)
        outputs = self.model(**inputs)

        hidden_states = []
        for i in range(len(uts_lens)):
            # take final representation of <s> token
            j = sum(uts_lens[:i])
            hidden_states.append(outputs.last_hidden_state[:, j, :])
        
        # (B, T, H)
        hidden_states = torch.stack(hidden_states, dim=1)

        return hidden_states

    def get_hidden_size(self):
        return self.model.config.hidden_size
