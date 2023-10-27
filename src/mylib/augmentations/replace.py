from .insert import Inserter


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
