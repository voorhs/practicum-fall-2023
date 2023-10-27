from transformers import pipeline
from tqdm import tqdm

class BackTranslator:
    """Back Translate each utterance (separately). Preserves intent."""

    def __init__(self, language, device='cpu'):
        self.language = language
        self.device = device

    # @staticmethod
    # def _load_utterances():
    #     dialogues = json.load(open('aug-data/original.json', 'r'))
    #     utterances = []
    #     for dia in dialogues:
    #         utterances.extend([item['utterance'] for item in dia])
    #     return utterances, dialogues

    # @staticmethod
    # def _save(augmented, original, name):
    #     """
    #     Params
    #     ------
    #     - augmented: list of utterances
    #     - original: list of lists of dicts with keys 'utterance', 'speaker'
    #     - name: name of output .json file
    #     """
    #     i = 0
    #     res = []
    #     for dia in original:
    #         aug_dia = []
    #         for item in dia:
    #             aug_dia.append({'utterance': augmented[i], 'speaker': item['speaker']})
    #             i += 1
    #         res.append(aug_dia)
    #     json.dump(res, open(f'aug-data/{name}.json', 'w'))

    # def from_file_system(self, name='back_trans_hf'):
    #     """
    #     Params
    #     ------
    #     - name: str, name of output .json file
    #     """

    #     uts = self._load_utterances()
    #     back_translated = self._augment(uts)
    #     self._save(back_translated, dialogues, name)

    def __call__(self, dialogues):
        """
        Params
        ------
        - dialogues: list[list[str]]
        """

        #! replace to some generic data type of a dialogue
        uts = to_uts(dialogues)
        forth = self._forth(uts)
        back = self._back(forth)
        
        #!
        ru = to_dia(forth, dialogues)
        #!
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

