if __name__ == "__main__":
    from datasets import load_dataset
    import json

    dataset = load_dataset("multi_woz_v22")
    n_dialogues = 15
    dialogues = []
    for dia in dataset['validation']['turns'][:n_dialogues]:    
        dialogues.append([{'utterance': ut, 'speaker': sp} for ut, sp in zip(dia['utterance'], dia['speaker'])])

    import os
    if not os.path.exists('aug-data'):
        os.makedirs('aug-data')
    json.dump(dialogues, open('aug-data/original.json', 'w'))