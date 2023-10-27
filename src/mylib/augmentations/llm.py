from typing import Literal, List
import numpy as np
import json


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
