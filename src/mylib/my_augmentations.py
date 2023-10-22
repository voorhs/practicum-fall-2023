import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, pipeline
from typing import List, Literal
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


class BackTranslator:
    """Back Translate each utterance (separately). Preserves intent."""

    def __init__(self, language, device='cpu'):
        self.language = language
        self.device = device

    @staticmethod
    def _load_utterances():
        dialogues = json.load(open('aug-data/original.json', 'r'))
        utterances = []
        for dia in dialogues:
            utterances.extend([item['utterance'] for item in dia])
        return utterances, dialogues

    @staticmethod
    def _save(augmented, original, name):
        """
        Params
        ------
        - augmented: list of utterances
        - original: list of lists of dicts with keys 'utterance', 'speaker'
        - name: name of output .json file
        """
        i = 0
        res = []
        for dia in original:
            aug_dia = []
            for item in dia:
                aug_dia.append({'utterance': augmented[i], 'speaker': item['speaker']})
                i += 1
            res.append(aug_dia)
        json.dump(res, open(f'aug-data/{name}.json', 'w'))

    def from_file_system(self, name='back_trans_hf'):
        """
        Params
        ------
        - name: str, name of output .json file
        """

        uts = self._load_utterances()
        back_translated = self._augment(uts)
        self._save(back_translated, dialogues, name)

    def from_argument(self, dialogues):
        """
        Params
        ------
        - dialogues: list[list[str]]
        """

        uts = to_uts(dialogues)
        forth = self._forth(uts)
        back = self._back(forth)
        
        ru = to_dia(forth, dialogues)
        res = to_dia(back, dialogues)
        return res, ru

    def _augment(self, uts):
        return self._back(self._forth(uts))

    def _forth(self, uts, batch_size=16):
        model = pipeline(
            f'translation_en_to_{self.language}',
            model=f'Helsinki-NLP/opus-mt-en-{self.language}',
            batch_size=batch_size,
            device=self.device
        )
        res = []
        for start in tqdm(range(0, len(uts), batch_size), desc='forward translating batches'):
            end = start + batch_size
            batch = uts[start:end]
            res.extend([a['translation_text'] for a in model(batch)])
        return res
    
    def _back(self, uts, batch_size=16):
        model = pipeline(
            f'translation_{self.language}_to_en',
            model=f'Helsinki-NLP/opus-mt-{self.language}-en',
            batch_size=batch_size,
            device=self.device
        )
        res = []
        for start in tqdm(range(0, len(uts), batch_size), desc='backward translating batches'):
            end = start + batch_size
            batch = uts[start:end]
            res.extend([a['translation_text'] for a in model(batch)])
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

    @staticmethod
    def _load_dialogues():
        dialogues = json.load(open('aug-data/original.json', 'r'))
        res = []
        for dia in dialogues:
            res.append([item['utterance'] for item in dia])
        return res, dialogues

    def from_file_system(self, name):
        """
        Add words to random places of dialogues.
        
        Reads data from 'aug-data/original.json'. Saves result to f'aug-data/{name}.json'.
        """

        # load data
        dialogues, original = self._load_dialogues()

        filled = self._augment(dialogues)
        
        BackTranslator._save(filled, original, name)
    
    def from_argument(self, dialogues):
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


class Replacer(Inserter):
    def __init__(
            self,
            k=3,
            fill_utterance_level=True,
            model='xlnet-base-cased',
            device='cpu',
            forbidden_tokens=None
        ):
        super().__init__(
            fraction=1,
            score_threshold=0,
            k=k,
            mask_utterance_level=False,
            fill_utterance_level=fill_utterance_level,
            model=model,
            device=device,
            forbidden_tokens=forbidden_tokens
        )
        self.replaced_tokens = []

    def _insert(self, words):
        for i, word in enumerate(words):
            if self._is_not_forbidden(word):
                self.replaced_tokens.append(word)
                words[i] = '<mask>'
        return words
    
    def _replace_masks(self, text, outputs):
        for words, scores in outputs:
            i = text.find('<mask>')
            to_insert = self.replaced_tokens.pop(0)
            if len(words) > 0:
                probs = np.array(scores) / sum(scores)
                to_insert = words[int(np.random.choice(len(words), 1, p=probs))]
            text = text[:i] + to_insert + text[i+6:]
        return text


class LlamaMaskFiller:
    def __init__(self, llm, tokenizer, masking: Literal['replace', 'insert', 'head', 'tail'], fraction=0.1, as_json=True):
        self.masking = masking
        self.fraction = fraction
        self.as_json = as_json

        self.llm = llm
        self.tokenizer = tokenizer

        print(self)

    def _prompt(self, dialogue, speaker):
        """
        Make prompt for mask filling task.

        Params
        ------
            dialogue: list[str], list of dialogue utterances
            speaker: list[int], list of IDs of speaker for each utterance
            as_json: whether to give result as a json (works very poorly)
        """

        if self.as_json:
            utterances = json.dumps({'utterances': dialogue, 'speaker': speaker})
            system = """You work as a function for dialogue completion. Input is a json dictionaty with keys 'utterances' and 'speaker'. Former item is a list of dialogue utterances, latter item is a list of IDs of speakers corresponding to each utterance.

You need to replace all [[LOST]] tokens in utterances with generated meaningful utterances that match the context of the dialogue. Firstly, construct json list `utterances`, then look through it and construct `speaker` to ensure that it contains all speakers IDs corresponding to resulting utterances in a correct order and count. Ensure that json format is correct.

You must not give any comments, greetings or explanations, only resulting json following the format of input."""

        else:
            utterances = '\n'.join([f'[{"AB"[i]}] ```{ut}```' for i, ut in zip(speaker, dialogue)])
            system = """You work as a function with specific input text and desired output text. Do not give any comments, greetings or explanations, only desired output.

Specific input is a dialogue prodived by a user. Each turn of the dialogue begins with [A] or [B] denoting the speaker, followed by the utterance in triple quotes. Some utterances are lost, this is indicated by the [[LOST]] token.

You need to replace all [[LOST]] tokens with generated meaningful utterances that match the context of the dialogue.

Desired output is the utterances generated in the places where [[LOST]] tokens were. It means you must not rewrite entire dialogue."""

        return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

Specific input:
\"\"\"
{utterances}
\"\"\"
[/INST]"""

    @staticmethod
    def _replace(dialogue, speaker, fraction):
        dialogue = dialogue[:]
        n = len(dialogue)
        size = np.ceil(n * fraction).astype(int)
        i_places = np.random.choice(n, size=size, replace=False)
        for i in range(n):
            if i in i_places:
                dialogue[i] = '[[LOST]]'
            else:
                dialogue[i] = f'{dialogue[i]}'
        return dialogue, speaker

    @staticmethod
    def _tail(dialogue: List[str], speaker, fraction):
        dialogue = dialogue[:]
        speaker = speaker[:]
        n = len(dialogue)
        size = np.ceil(n * fraction).astype(int)
        role = speaker[-1]
        for i in range(n, n+size):
            dialogue.insert(i, '[[LOST]]')
            role = 1-role
            speaker.insert(i, role)
            
        return dialogue, speaker

    @staticmethod
    def _insert(dialogue: List[str], speaker, fraction):
        dialogue = dialogue[:]
        n = len(dialogue)
        size = np.ceil(n * fraction).astype(int)
        i_places = np.sort(np.random.choice(n, size=size, replace=False)) + np.arange(size) * 2
        for i in i_places:
            dialogue.insert(i, '[[LOST]]')
            dialogue.insert(i, '[[LOST]]')
            role = speaker[i-1]
            speaker.insert(i, role)
            speaker.insert(i, 1-role)
        return dialogue, speaker
    
    @staticmethod
    def _head(dialogue: List[str], speaker, fraction=0.2):
        dialogue = dialogue[:]
        speaker = speaker[:]
        n = len(dialogue)
        size = np.ceil(n * fraction).astype(int)
        role = speaker[0]
        for i in range(size):
            dialogue.insert(i, '[[LOST]]')
            role = 1-role
            speaker.insert(i, role)
            
        return dialogue, speaker

    @staticmethod
    def _load_dialogues():
        dialogues = json.load(open('aug-data/original.json', 'r'))
        res_ut = []
        res_sp = []
        for dia in dialogues:
            res_ut.append([item['utterance'] for item in dia])
            res_sp.append([item['speaker'] for item in dia])
        return res_ut, res_sp

    def from_file_system(self, name):
        dialogues, speakers = self._load_dialogues()

        if self.masking == 'replace':
            masker = LlamaMaskFiller._replace
        elif self.masking == 'insert':
            masker = LlamaMaskFiller._insert
        elif self.masking == 'head':
            masker = LlamaMaskFiller._head
        elif self.masking == 'tail':
            masker = LlamaMaskFiller._tail
        else:
            raise ValueError(f'Unknown masking: {self.masking}')

        prompt_list = []
        for dia, spe in zip(dialogues, speakers):
            dia, spe = masker(dia, spe, self.fraction)
            prompt_list.append(self._prompt(dia, spe))

        sequences = self.llm(
            prompt_list,
            num_return_sequences=1,
            max_new_tokens=1024,
            batch_size=2,
            
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=1
        )

        json.dump(sequences, open(f'aug-data/{name}-raw.json', 'w'))


class LlamaSummarizer:
    def __init__(self, penalty_length, llm, tokenizer, as_json=True):
        self.penalty_length = penalty_length
        self.as_json = as_json
        self.llm = llm
        self.tokenizer= tokenizer

        print(self)
    
    def _prompt(self, dialogue, speaker):
        """
        Make prompt for mask filling task.

        Params
        ------
            dialogue: list[str], list of dialogue utterances
            speaker: list[int], list of IDs of speaker for each utterance
            as_json: whether to give result as a json (works very poorly)
        """

        if self.as_json:
            utterances = json.dumps({'utterances': dialogue, 'speaker': speaker})
            system = """You work as a function for dialogue completion. Input is a json dictionaty with keys 'utterances' and 'speaker'. Former item is a list of dialogue utterances, latter item is a list of IDs of speakers corresponding to each utterance.

You need to replace all [[LOST]] tokens in utterances with generated meaningful utterances that match the context of the dialogue. Firstly, construct json list `utterances`, then look through it and construct `speaker` to ensure that it contains all speakers IDs corresponding to resulting utterances in a correct order and count. Ensure that json format is correct.

You must not give any comments, greetings or explanations, only resulting json following the format of input."""

        else:
            utterances = '\n'.join([f'[{"AB"[i]}] ```{ut}```' for i, ut in zip(speaker, dialogue)])
            system = f"""You work as a function for dialogue text transformation with specific input text and desired output text.

Specific input is a dialogue prodived by a user. Each turn of the dialogue begins with [A] or [B] denoting the speaker, followed by the utterance in triple quotes.

You need to summarize the dialogue by making new one. In total from all speakers, new dialogue maximum number of turns must be two times less than original number of turns. For example, if there are 12 turns in the original dialogue, then there are no more than 6 turns in the new dialogue. New dialogue must preserve meaningfulness and general context of original dialogue.

Desired output is a resulting dialogue following the input format of the original dialogue."""

        return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

Specific input:
\"\"\"
{utterances}
\"\"\"
[/INST]"""

    def from_file_system(self, name):
        dialogues, speakers = LlamaMaskFiller._load_dialogues()

        prompt_list = []
        for dia, spe in zip(dialogues, speakers):
            prompt_list.append(self._prompt(dia, spe))
    
        sequences = self.llm(
            prompt_list,
            num_return_sequences=1,
            max_new_tokens=1024,
            batch_size=2,
            
            do_sample=True,
            top_k=50,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.4,
            length_penalty=self.penalty_length
        )

        json.dump(sequences, open(f'aug-data/{name}-raw.json', 'w'))


class LlamaVerbose(LlamaSummarizer):
    def _prompt(self, dialogue, speaker):
        """
        Make prompt for mask filling task.

        Params
        ------
            dialogue: list[str], list of dialogue utterances
            speaker: list[int], list of IDs of speaker for each utterance
            as_json: whether to give result as a json (works very poorly)
        """

        if self.as_json:
            utterances = json.dumps({'utterances': dialogue, 'speaker': speaker})
            system = """You work as a function for dialogue completion. Input is a json dictionaty with keys 'utterances' and 'speaker'. Former item is a list of dialogue utterances, latter item is a list of IDs of speakers corresponding to each utterance.

You need to expand the dialogue by making new one. In total from all speakers, new dialogue minimum number of turns must be 1.5 times more than original number of turns. For example, if there are 12 turns in the original dialogue, then there are at least 18 turns in the new dialogue. New dialogue must preserve meaningfulness and general context of original dialogue.
Firstly, construct json list `utterances`, then look through it and construct `speaker` to ensure that it contains all speakers IDs corresponding to resulting utterances in a correct order and count. Ensure that json format is correct.

You must not give any comments, greetings or explanations, only resulting json following the format of input."""

        else:
            utterances = '\n'.join([f'[{"AB"[i]}] ```{ut}```' for i, ut in zip(speaker, dialogue)])
            system = f"""You work as a function for dialogue text transformation with specific input text and desired output text. Do not give any comments or explanation, only resulting desired output text.

Specific input is a dialogue prodived by a user. Each turn of the dialogue begins with [A] or [B] denoting the speaker, followed by the utterance in triple quotes.

You need to expand the dialogue by making new one. In total from all speakers, new dialogue minimum number of turns must be two times more than original number of turns. For example, if there are 6 turns in the original dialogue, then there are at least 12 turns in the new dialogue. New dialogue must preserve meaningfulness and general context of original dialogue.

Desired output is a resulting dialogue following the input format of the original dialogue."""

        return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

Specific input:
\"\"\"
{utterances}
\"\"\"
[/INST]"""


class LlamaParaphraser(LlamaSummarizer):
    def __init__(self, style: Literal['formal', 'informal', 'technical', 'persuasive', 'creative', 'poetic', 'playful'], llm, tokenizer, as_json=True):
        self.style = style
        super().__init__(0, llm, tokenizer, as_json)

    def _prompt(self, dialogue, speaker, as_json=False):
        """
        Make prompt for mask filling task.

        Params
        ------
            dialogue: list[str], list of dialogue utterances
            speaker: list[int], list of IDs of speaker for each utterance
            as_json: whether to give result as a json (works very poorly)
        """

        style_descr = {
            'formal': 'formal style. This type of text is characterized by a more serious and professional tone, often used in formal letters, business proposals, and academic papers.',
            'informal': 'informal style. This type of text is characterized by a more casual and relaxed tone, often used in everyday conversations, social media, and text messages.',
            'technical': 'technical style. This type of text is characterized by the use of technical terms and jargon, often used in instruction manuals, technical reports, and scientific papers.',
            'persuasive': 'persuasive style. This type of text is characterized by the use of rhetorical devices and persuasive techniques, often used in sales and marketing materials, persuasive essays, and opinion pieces.',
            'creative': 'creative style. This type of text is characterized by imaginative and expressive language, often used in poetry, fiction, and creative nonfiction',
            'poetic': 'poetic style. This type of text is characterized by imaginative language and creative expression, often used in poetry, song lyrics, and spoken word performances.',
            'playful': "playful style. This style of dialogue involves humor, wit, and lighthearted teasing. It's characterized by a relaxed and joyful atmosphere, and a willingness to have fun and enjoy each other's company.",
        }

        if as_json:
            utterances = json.dumps({'utterances': dialogue, 'speaker': speaker})
            system = f"""You work as a function for dialogue completion. Input is a json dictionaty with keys 'utterances' and 'speaker'. Former item is a list of dialogue utterances, latter item is a list of IDs of speakers corresponding to each utterance.

You need to construct new dialogue by paraphrasing original dialogue to {style_descr[self.style]}. New dialogue must preserve meaningfulness and general context of original dialogue.
Firstly, construct json list `utterances`, then look through it and construct `speaker` to ensure that it contains all speakers IDs corresponding to resulting utterances in a correct order and count. Ensure that json format is correct.

You must not give any comments, greetings or explanations, only resulting json following the format of input."""

        else:
            utterances = '\n'.join([f'[{"AB"[i]}] ```{ut}```' for i, ut in zip(speaker, dialogue)])
            system = f"""You work as a function for dialogue text transformation with specific input text and desired output text. Do not give any comments or explanation, only resulting desired output text.

Specific input is a dialogue prodived by a user. Each turn of the dialogue begins with [A] or [B] denoting the speaker, followed by the utterance in triple quotes.

You need to construct new dialogue by paraphrasing original dialogue to {style_descr[self.style]}. New dialogue must preserve meaningfulness and general context of original dialogue.

Desired output is a resulting dialogue following the input format of the original dialogue."""

        return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

Specific input:
\"\"\"
{utterances}
\"\"\"
[/INST]"""


from nup.models.dialogue import UtteranceTransformerDMConfig, UtteranceTransformerDM
from nup.models.listwise import DecoderUtteranceSorter, Decoder
import torch


def load_listwise_decoder(ckpt_path, device):
    head_dropout_prob = 0.02
    encoder_name = 'sentence-transformers/all-mpnet-base-v2'
    config = UtteranceTransformerDMConfig(
        num_attention_heads=4,
        attention_probs_dropout_prob=0.02,
        n_layers=4,
        encoder_name=encoder_name,
        embed_turn_ids=False,
        is_casual=False
    )
    _dialogue_model = UtteranceTransformerDM(config)

    _model = DecoderUtteranceSorter(
        dialogue_model=_dialogue_model,
        dropout_prob=head_dropout_prob,
        max_n_uts=20,
        decoder=Decoder(top_k=2)
    )

    return DecoderUtteranceSorter.from_checkpoint(
        path_to_ckpt=ckpt_path,
        model=_model,
        map_location=device
    )


class ListwiseShuffler:
    def __init__(
            self,
            decoder=Decoder(),
            ckpt_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/listwie-decoder-symmetric-t1/checkpoints/last.ckpt',
            device='cpu',
            thresh=-np.inf
        ):
        self.decoder = decoder
        self.thresh = thresh
        self.model = load_listwise_decoder(ckpt_path, device)

    def from_file_system(self, name):
        dialogues = json.load(open('aug-data/original.json', 'r'))
        res = self.from_argument(dialogues)
        json.dump(res, open(f'aug-data/{name}.json', 'w'))

    def from_argument(self, dialogues, batch_size=192):
        aug_dialogues_with_scores = []
        for i_batch in tqdm(range(0, len(dialogues), batch_size), desc='shuffling batches'):
            start = i_batch
            end = i_batch + batch_size
            batch = dialogues[start:end]
            aug_dialogues_with_scores.extend(self.model.augment(batch, self.decoder))
        
        return [aug if score >= self.thresh else None for aug, score in aug_dialogues_with_scores]


from nup.models.pairwise import ChainCosine, TargetEncoder, ContextEncoderConcat
from nup.models.aux import mySentenceTransformer
from sklearn.cluster import AgglomerativeClustering


def load_pairwise_cat(ckpt_path, device):
    context_size = 3
    encoder_name = 'aws-ai/dse-bert-large'

    _encoder = mySentenceTransformer(encoder_name)
    _target_encoder = TargetEncoder(_encoder)
    _context_encoder = ContextEncoderConcat(_encoder, context_size=context_size)
    _model = ChainCosine(
        target_encoder=_target_encoder,
        context_encoder=_context_encoder,
        projection_size=256,
        context_size=context_size,
    )

    return ChainCosine.from_checkpoint(
        path_to_ckpt=ckpt_path,
        model=_model,
        map_location=device
    ).eval()


class PairwiseCutter:
    def __init__(
            self,
            ckpt_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/pairwise-cat-speaker-issue/checkpoints/last.ckpt',
            device='cpu',
            thresh=-np.inf
        ):
        self.thresh = thresh
        self.model = load_pairwise_cat(ckpt_path, device)

    def from_file_system(self, name):
        dialogues = json.load(open('aug-data/original.json', 'r'))
        
        res = []
        for dia in tqdm(dialogues, desc='cutting dialogues'):
            aug, score = self._cut(self.model, dia)
            res.append(aug if score >= self.thresh else None)
            
        json.dump(res, open(f'aug-data/{name}.json', 'w'))

    def from_argument(self, dialogues):
        res = []
        for dia in tqdm(dialogues, desc='cutting dialogues'):
            aug, score = self._cut(self.model, dia)
            res.append(aug if score >= self.thresh else None)
        return res

    @staticmethod
    def _cut(model, dia):
        """drops all clusters except the biggest one. applies transformation only to dialogues with 6 utterances at least"""
        if len(dia) < 6:
            return None, -np.inf
        end = len(dia) // 3
        start = 2
        variations = []
        for n_clusters in range(start, end+1):
            clusterwise_uts = PairwiseCutter._cluster(model, dia, n_clusters)
            ids = clusterwise_uts[np.argmax([len(clust) for clust in clusterwise_uts])]
            aug = [dia[i] for i in ids]
            score = model.score(aug)
            variations.append((aug, score))
        res, score = max(variations, key=lambda x: x[1])
        return res, score

    @staticmethod
    @torch.no_grad()
    def _cluster(model, dia, n_clusters):
        """clusters utterances within dia according to logits (similarities) from pairwise model"""
        batch = model.make_batch_from_dia(dia)
        similarities = model.get_logits(batch, temperature=1).cpu().numpy()
        
        # mask out similarities between utterances of same speaker
        # speaker = [item['speaker'] for item in dia]
        # context_speaker = np.array(speaker[:-1])[:, None]
        # target_speaker = np.array(speaker[1:])[None, :]
        # mask = (context_speaker != target_speaker) | np.eye(len(speaker)-1, dtype=np.bool_)
        # similarities[~mask] = -1e3

        labels = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='average',
            metric='precomputed'
        ).fit_predict(similarities)

        labels = np.r_[labels[0], labels]

        res = [[] for _ in range(len(np.unique(labels)))]
        for i_ut, lab in enumerate(labels):
            res[lab].append(i_ut)
        return res


import random
class PairwiseShuffler:
    def __init__(
            self,
            ckpt_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/pairwise-cat-speaker-issue/checkpoints/last.ckpt',
            device='cpu',
            thresh=-np.inf
        ):
        self.thresh = thresh
        self.model = load_pairwise_cat(ckpt_path, device)

    def from_file_system(self, name):
        dialogues = json.load(open('aug-data/original.json', 'r'))
        
        res = []
        for dia in tqdm(dialogues, desc='cutting dialogues'):
            aug, score = self._shuffle(self.model, dia)
            res.append(aug if score >= self.thresh else None)

        json.dump(res, open(f'aug-data/{name}.json', 'w'))
    
    def from_argument(self, dialogues):
        res = []
        for dia in tqdm(dialogues, desc='shuffling dialogues'):
            aug, score = self._shuffle(self.model, dia)
            res.append(aug if score >= self.thresh else None)
        return res

    @staticmethod
    @torch.no_grad()
    def _shuffle(model, dia):
        if len(dia) < 12:
            return None, -np.inf
        end = len(dia) // 3
        start = 4
        variations = []
        for n_clusters in range(start, end+1):
            clusterwise_uts = PairwiseCutter._cluster(model, dia, n_clusters)
            for i_try in range(n_clusters):
                random.shuffle(clusterwise_uts)
                aug = []
                for ut_ids in clusterwise_uts:
                    aug.extend([dia[i] for i in ut_ids])
                score = model.score(aug)
                variations.append((aug, score))
        res, score = max(variations, key=lambda x: x[1])
        return res, score


if __name__ == "__main__":

    # inserter = Inserter(
    #     fraction=0.5,
    #     score_threshold=0.005,
    #     k=5,
    #     mask_utterance_level=True,
    #     fill_utterance_level=2,
    #     model='microsoft/mpnet-base',
    #     device='cuda'
    # )
    # inserter.from_file_system('inserter')
    
    # replacer = Replacer(
    #     k=3,
    #     fill_utterance_level=2,
    #     model='microsoft/mpnet-base',
    #     device='cuda'
    # )
    # replacer.from_file_system('replacer')

    # back_translator = BackTranslator(
    #     language='ru',
    #     device='cuda'
    # )
    # back_translator.from_file_system('back_translator')

    # model = 'meta-llama/Llama-2-13b-chat-hf'

    # tokenizer = AutoTokenizer.from_pretrained(model)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # llm = pipeline(
    #     "text-generation",
    #     model=AutoModelForCausalLM.from_pretrained(
    #         model,
    #         device_map='auto',
    #         load_in_4bit=True
    #     ),
    #     tokenizer=tokenizer
    # )

    # LlamaMaskFiller(llm, tokenizer, 'replace', fraction=0.2).from_file_system('llm_replacer')
    # LlamaMaskFiller(llm, tokenizer, 'insert', fraction=0.2).from_file_system('llm_inserter')
    # LlamaMaskFiller(llm, tokenizer, 'head', fraction=0.2).from_file_system('llm_head')
    # LlamaMaskFiller(llm, tokenizer, 'tail', fraction=0.2).from_file_system('llm_tail')

    # LlamaSummarizer(-5, llm, tokenizer).from_file_system('llm_summarizer')
    # LlamaVerbose(+5, llm, tokenizer).from_file_system('llm_verbose')
    # LlamaParaphraser('formal', llm, tokenizer).from_file_system('llm_formal')
    # LlamaParaphraser('informal', llm, tokenizer).from_file_system('llm_informal')
    # LlamaParaphraser('technical', llm, tokenizer).from_file_system('llm_technical')
    # LlamaParaphraser('persuasive', llm, tokenizer).from_file_system('llm_persuasive')
    # LlamaParaphraser('creative', llm, tokenizer).from_file_system('llm_creative')
    # LlamaParaphraser('playful', llm, tokenizer).from_file_system('llm_playful')
    
    # listwise_shuffler = ListwiseShuffler(
    #     ckpt_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/listwise-utterance-transformer-amazon-resumed/checkpoints/last.ckpt',
    #     device='cuda:0'
    # )
    # listwise_shuffler.from_file_system('listwise_shuffler')

    import torch
    torch.set_float32_matmul_precision('medium')    

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', dest='method', required=True, choices=[
        'back-translate', 'insert', 'replace',
        'listwise-shuffler', 'pairwise-cutter', 'pairwise-shuffler'
    ])
    ap.add_argument('--path-out', dest='path_out', default=None)
    ap.add_argument('--cuda', dest='cuda', default='0')
    ap.add_argument('--path-in', dest='path_in', default='/home/alekseev_ilya/dialogue-augmentation/nup/dialogues/train')
    args = ap.parse_args()

    # from dataclasses import dataclass
    # @dataclass
    # class Args:
    #     method = 'listwise-shuffler'
    #     cuda = '2'
    #     path_in = '/home/alekseev_ilya/dialogue-augmentation/nup/dialogues/train'
    #     path_out = None
    # args = Args()

    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    json_chunks = sorted([filename for filename in os.listdir(args.path_in) if filename.endswith('.json')])
    if args.path_out is None:
        args.path_out = os.path.join(os.getcwd(), 'augmented', args.method)
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out)

    if args.method == 'back-translate':
        augmenter = BackTranslator(language='ru', device='cuda')
    elif args.method == 'insert':
        augmenter = Inserter(
            fraction=0.5,
            score_threshold=0.005,
            k=5,
            mask_utterance_level=True,
            fill_utterance_level=2,
            model='microsoft/mpnet-base',
            device='cuda'
        )
    elif args.method == 'replace':
        augmenter = Replacer(
            k=3,
            fill_utterance_level=2,
            model='microsoft/mpnet-base',
            device='cuda'
        )
    elif args.method == 'listwise-shuffler':
        augmenter = ListwiseShuffler(device='cuda')
    elif args.method == 'pairwise-cutter':
        augmenter = PairwiseCutter(device='cuda')
    elif args.method == 'pairwise-shuffler':
        augmenter = PairwiseShuffler(device='cuda')
        exit()
    
    for chunk in tqdm(json_chunks[80:]):
        dialogues = json.load(open(os.path.join(args.path_in, chunk), 'r'))
        clean_dialogues = [dia for dia in dialogues if dia is not None]
        augmented = augmenter.from_argument(clean_dialogues)
        if args.method == 'back-translate':
            augmented, ru = augmented
            json.dump(ru, open(os.path.join(args.path_out, 'ru-' + chunk), 'w'))
        for i, dia in enumerate(dialogues):
            if dia is None:
                augmented.insert(i, None)
        json.dump(augmented, open(os.path.join(args.path_out, chunk), 'w'))
