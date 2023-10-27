from torch import nn
from transformers import AutoModel, AutoTokenizer
from ...utils.training import HParamsPuller
from .base_dialogue_model import BaseDialogueModel


class BaselineDialogueEncoder(nn.Module, HParamsPuller, BaseDialogueModel):
    def __init__(self, hf_model_name):
        super().__init__()

        self.hf_model_name = hf_model_name

        self.model = AutoModel.from_pretrained(hf_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    @property
    def device(self):
        return self.model.device

    @staticmethod
    def _parse(dia):
        return [f'{item["speaker"]} {item["utterance"]}' for item in dia]

    @staticmethod
    def _tokenize(tokenizer, batch):
        sep = tokenizer.sep_token
        parsed = [sep.join(BaselineDialogueEncoder._parse(dia)) for dia in batch]
        inputs = tokenizer(parsed, padding='longest', return_tensors='pt')
        return inputs
    
    def forward(self, batch):
        inputs = self._tokenize(self.tokenizer, batch).to(self.device)
        hidden_states = self.model(**inputs).last_hidden_state      # (B, T, H)
        encodings = hidden_states[:, 0, :]                          # (B, H)
        return encodings

    def get_hidden_size(self):
        return self.model.config.hidden_size
