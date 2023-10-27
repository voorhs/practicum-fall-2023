from torch import nn
from ..hssa import HSSAModel, HSSAConfig, HSSATokenizer
from .base_dialogue_model import BaseDialogueModel


class HSSADM(nn.Module, BaseDialogueModel):
    def __init__(
            self,
            hf_model_name,
            config: HSSAConfig,
            encode_utterances=False,
            encode_dialogue=False
        ):
        super().__init__()

        self.hf_model_name = hf_model_name
        self.config = config

        if encode_dialogue and encode_utterances:
            raise ValueError('either dialogue or utterance encodings can be demanded')
        self.encode_dialogue = encode_dialogue
        self.encode_utterances = encode_utterances

        self.model = HSSAModel.from_pretrained(hf_model_name, config=config)
        self.tokenizer = HSSATokenizer.from_pretrained(hf_model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    @property
    def device(self):
        return self.model.device

    def forward(self, batch):
        """
        returned shape:

        - without pooling: (B, T, S, H) --- each token for each utterance for each dia in batch
        - pool_utterance_level: (B, T, H) --- each utterance for each dia in batch
        - pool_dialogue_level: (B, H) --- each dia
        """
        if self.encode_dialogue:
            # add cls utterance
            batch = [[{'speaker': None, 'utterance': ''}] + dia for dia in batch]
        
        tokenized = self.tokenizer(batch).to(self.device)
        # (B, T, S, H)
        hidden_states = self.model(**tokenized)
        
        if self.encode_dialogue:
            # cls token of entire dialogue
            hidden_states = hidden_states[:, 0, 0, :]
        elif self.encode_utterances:
            # cls tokens of utterances
            hidden_states = hidden_states[:, :, 0, :]
        
        return hidden_states
    
    def get_hidden_size(self):
        return self.config.hidden_size

    def get_hparams(self):
        res = self.config.to_dict()
        res['hf_model_name'] = self.hf_model_name
        return res
