import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, pipeline
import numpy as np
from typing import List
from tqdm import tqdm


def to_uts(dialogues):
    res = []
    for dia in dialogues:
        for item in dia:
            res.append(item['utterance'])
    return res

def to_dia(uts, dialogues):
    dia_lens = [len(dia) for dia in dialogues]
    res = []
    for i, length in enumerate(dia_lens):
        start = sum(dia_lens[:i])
        end = start + length
        cur_res = []
        for j, ut in enumerate(uts[start:end]):
            cur_res.append({
                'speaker': dialogues[i][j]['speaker'],
                'utterance': ut
            })
        res.append(cur_res)
    return res


class Inserter:
    def __init__(
            self,
            fraction=0.1,
            score_threshold=0.005,
            k=5,
            mask_utterance_level=False,
            fill_utterance_level=True,
            model='xlnet-base-cased',
            device='cpu',
            forbidden_tokens=None
        ):
        """
        Params
        ------
        - fraction: float in (0,1), fraction of words by which to increase the length of the dialogues
        - score_thresold: float, lower bound for probability of filled token
        - k: int, parameter for topk sampling
        - mask_utterance_level: bool, whether to mask dialogues as whole or mask each utterance separately
        - fill_utterance_level: bool or int > 1, whether to fill masks in dialogues as whole or process each utterance separately or use context of previous utterances
        - model: str, fill-mask model from hugging face
        - forbidden_tokens: list[str], list of all tokens which won't be used as insertions
        """

        self.fraction = fraction
        self.score_threshold = score_threshold
        self.k = k
        self.mask_utterance_level = mask_utterance_level
        self.fill_utterance_level = fill_utterance_level
        self.model = model
        self.device = device

        if forbidden_tokens is None:
            nltk.download('stopwords')
            forbidden_tokens = stopwords.words('english')
            forbidden_tokens.extend(AutoTokenizer.from_pretrained(self.model).all_special_tokens)
        self.forbidden_tokens = forbidden_tokens

    def _insert(self, words):
        """Insert <mask> after each space"""
        n = len(words)
        size = np.ceil(n * self.fraction).astype(int)
        i_places = np.sort(np.random.choice(n, size=size, replace=False)) + np.arange(size)
        for i in i_places:
            words.insert(i, '<mask>')
        return words

    def _insert_masks_dialogue_level(self, dialogues) -> List[str]:
        """
        Insert <mask> into random places of dialogues

        Params
        ------
        - dialogues: list[list[str]] 

        Return
        ------
        list of dialogues, where each dialogue is a single string with \\n delimiter between utterances
        """

        res = []
        
        for dia in dialogues:
            original = '\n'.join(dia)
            words = self._insert(original.split(' '))
            res.append(' '.join(words))
        
        return res

    def _insert_masks_utterance_level(self, dialogues) -> List[str]:
        """
        Insert <mask> into random places of dialogues

        Params
        ------
        - dialogues: list[list[str]] 

        Return
        ------
        list of dialogues, where each dialogue is a single string with \\n delimiter between utterances
        """

        res = []

        for dia in dialogues:
            masked = []
            for ut in dia:
                words = self._insert(ut.split(' '))
                masked.append(' '.join(words))
                
            res.append('\n'.join(masked))
        
        return res

    def _is_not_forbidden(self, word: str) -> bool:
        word = word.lower()
        flags = []
        flags.append(word in self.forbidden_tokens)
        flags.append(not word.isalpha())
        return not any(flags)

    def _choose_confident(self, fill_res):
        """
        Drop predicted tokens which have low score or are included into forbidden tokens.
        
        Params
        ------
        - fill_res: predicted tokens for single <mask>
        """
        words = []
        scores = []
        for word, score in map(lambda x: (x['token_str'], x['score']), fill_res):
            if len(words) == self.k:
                break
            if score < self.score_threshold:
                continue
            if self._is_not_forbidden(word):
                words.append(word)
                scores.append(score)
        return words, scores
    
    def _replace_masks(self, text, outputs) -> str:
        """Replace <mask> with predicted tokens."""
        for words, scores in outputs:
            i = text.find('<mask>')
            if len(words) == 0:
                if text[i-1] == ' ':
                    text = text[:i-1] + text[i+6:]
                else:
                    text = text[:i] + text[i+7:]
            else:
                probs = np.array(scores) / sum(scores)
                to_insert = words[int(np.random.choice(len(words), 1, p=probs))]
                text = text[:i] + to_insert + text[i+6:]
        return text

    def _fill_masks_dialogue_level(self, masked_dialogues) -> List[str]:
        """
        Apply MLM to fill <mask> in given dialogues.

        Params
        ------
        - masked_dialogues: list of dialogues i.e. strings with \\n delimiter betweem utterances
 
        Return
        ------
        list of utterances merged into single list
        """

        dataset_fill_results = self._to_mlm(masked_dialogues)
        res = []
        for dia, dia_fill_results in zip(masked_dialogues, dataset_fill_results):
            # choose only confident predictions
            outputs = [self._choose_confident(mask_fill_results) for mask_fill_results in dia_fill_results]
                
            # insert predictions
            res.extend(self._replace_masks(dia, outputs).split('\n'))
        
        return res

    def _fill_masks_utterance_level(self, masked_dialogues) -> List[str]:
        """
        Apply MLM to fill <mask> in given dialogues.

        Params
        ------
        - masked_dialogues: list of dialogues, where each dialogue is a single string with \\n delimiter between utterances

        Return
        ------
        list of utterances merged into single list
        """

        # get single list of utterances
        utterances = []
        for dia in masked_dialogues:
            utterances.extend(dia.split('\n'))
        
        # separate those with and without <mask>
        i_uts_without_mask = []
        uts_with_mask = []
        for i, ut in enumerate(utterances):
            if ut.find('<mask>') == -1:
                i_uts_without_mask.append(i)
            else:
                uts_with_mask.append(ut)

        # feed to MLM
        dataset_fill_results = self._to_mlm(uts_with_mask)
        
        # insert predictions to utterances with <mask>
        res = []
        for ut, ut_fill_results in zip(uts_with_mask, dataset_fill_results):
            if isinstance(ut_fill_results[0], dict):
                ut_fill_results = [ut_fill_results]
            candidates = [self._choose_confident(mask_fill_results) for mask_fill_results in ut_fill_results]
            res.append(self._replace_masks(ut, candidates))

        # merge masked and untouched utterances
        for i in i_uts_without_mask:
            res.insert(i, utterances[i])

        return res
    
    def _fill_masks_context_level(self, masked_dialogues, context_length) -> List[str]:
        """
        Apply MLM to fill <mask> in given dialogues.

        Params
        ------
        - masked_dialogues: list of dialogues, where each dialogue is a single string with \\n delimiter between utterances

        Return
        ------
        list of utterances merged into single list
        """

        # get single list of utterances
        context_list = []
        for dia in masked_dialogues:
            dia = dia.split('\n')
        
            # join consequetive utterances into single string
            context_list.extend(['\n'.join(dia[i:i+context_length]) for i in range(0, len(dia), context_length)])

        # separate those with and without <mask>
        i_cont_without_mask = []
        cont_with_mask = []
        for i, ut in enumerate(context_list):
            if ut.find('<mask>') == -1:
                i_cont_without_mask.append(i)
            else:
                cont_with_mask.append(ut)

        # feed to MLM
        dataset_fill_results = self._to_mlm(cont_with_mask)
        if len(cont_with_mask) != len(dataset_fill_results):
            raise RuntimeError("Something's wrong with MLM mask filling")
        
        # insert predictions to contexts with <mask>
        res = []
        for i, (context, context_fill_results) in enumerate(zip(cont_with_mask, dataset_fill_results)):
            if isinstance(context_fill_results[0], dict):
                context_fill_results = [context_fill_results]
            candidates = [self._choose_confident(mask_fill_results) for mask_fill_results in context_fill_results]
            res.append(self._replace_masks(context, candidates))

        # merge masked and untouched utterances
        for i in i_cont_without_mask:
            res.insert(i, context_list[i])
        
        # roll out contexts
        res_uts = []
        for context in res:
            res_uts.extend(context.split('\n'))

        return res_uts

    def _to_mlm(self, strings_with_masks, batch_size=16):
        model = pipeline(
            'fill-mask',
            model=self.model,
            batch_size=batch_size,
            device=self.device
        )
        res = []
        for start in tqdm(range(0, len(strings_with_masks), batch_size), desc='batches'):
            end = start + batch_size
            batch = strings_with_masks[start:end]
            outputs = model(batch, top_k=100)
            if len(batch) == 1:
                # hf pipeline implementation employs terrible idea that returned output doesn't follow nested structure of input list if the latter contains single element 
                # https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/fill_mask.py#L271
                outputs = [outputs]
            res.extend(outputs)
        return res

    # @staticmethod
    # def _load_dialogues():
    #     dialogues = json.load(open('aug-data/original.json', 'r'))
    #     res = []
    #     for dia in dialogues:
    #         res.append([item['utterance'] for item in dia])
    #     return res, dialogues

    # def from_file_system(self, name):
    #     """
    #     Add words to random places of dialogues.
        
    #     Reads data from 'aug-data/original.json'. Saves result to f'aug-data/{name}.json'.
    #     """

    #     # load data
    #     dialogues, original = self._load_dialogues()

    #     filled = self._augment(dialogues)
        
    #     BackTranslator._save(filled, original, name)
    
    def __call__(self, dialogues):
        """
        Add words to random places of dialogues from 'aug-data/original.csv'.

        Params
        ------
        - dialogues: list[list[str]]

        Return
        ------
        - list[list[str]]
        """
        uts = []
        for dia in dialogues:
            uts.append([item['utterance'] for item in dia])

        filled = self._augment(uts)
        
        #! replace to some generic data type of a dialogue
        res = to_dia(filled, dialogues)

        return res
    
    def _augment(self, dialogues):
        if self.mask_utterance_level:
            masked = self._insert_masks_utterance_level(dialogues)
        else:
            masked = self._insert_masks_dialogue_level(dialogues)
        
        if isinstance(self.fill_utterance_level, int):
            filled = self._fill_masks_context_level(masked, self.fill_utterance_level)
        elif self.fill_utterance_level:
            filled = self._fill_masks_utterance_level(masked)
        else:
            filled = self._fill_masks_dialogue_level(masked)
        return filled

