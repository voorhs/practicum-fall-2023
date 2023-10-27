from pprint import pformat
from typing import List, Union
from copy import copy


class BaseDictDType:
    content: dict

    def asdict(self) -> dict:
        raise NotImplemented()
    
    @staticmethod
    def get_train_sample(dct):
        raise NotImplemented()


class Dialogue(BaseDictDType):
    def __init__(
            self,
            utterances: List[str],
            speakers: List[int],
            source_dataset_name: str,
            idx_within_source: int,
            idx: int,
            **fields
        ):
        """add any extra `fields` if extra info is needed to be saved"""

        if len(utterances) != len(speakers):
            raise ValueError('`utterances` and `speakers` must be the same length')
        
        self.content = [
            {'utterance': ut, 'speaker': sp} for ut, sp in zip(utterances, speakers)
        ]
        
        self.source_dataset_name = source_dataset_name
        self.idx_within_source = idx_within_source
        self.idx = idx
        
        for key, val in fields.items():
            setattr(self, key, val)
    
    def asdict(self):
        return vars(self)
    
    def __repr__(self):
        return pformat(self.asdict(), indent=2)
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, idx_or_slice):
        res = copy(self)
        res.content = self.content[idx_or_slice]
        return res

    @staticmethod
    def get_train_sample(dct):
        return dct['content']
    

class ContextResponsePair(BaseDictDType):
    def __init__(
            self,
            context: Union[Dialogue, dict],
            response: Union[Dialogue, dict],
            idx: int = None,
            **fields
        ):
        """add any extra `fields` if extra info is needed to be saved"""

        if isinstance(context, Dialogue) and isinstance(response, Dialogue):
            self.content = {
                'context': context.content,
                'response': response.content
            }
            self.idx_within_source = context.idx_within_source
            self.idx = idx
        elif isinstance(context, dict) and isinstance(response, dict):
            self.content = {
                'context': context,
                'response': response
            }
        else:
            raise ValueError(f'context and response must be the same data type, got {type(context)} and {type(response)}')
        
        for key, val in fields.items():
            setattr(self, key, val)
    
    def asdict(self):
        return vars(self)
    
    def __repr__(self):
        return pformat(self.asdict(), indent=2)
    
    @staticmethod
    def from_dict(dct):
        return ContextResponsePair(
            context=dct['content']['context'],
            response=dct['content']['response'],
        )
    
    @staticmethod
    def get_train_sample(dct):
        return dct['content']
