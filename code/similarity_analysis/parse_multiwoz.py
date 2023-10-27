from datasets import load_dataset
import json
from tqdm import tqdm
import os


def parse():
    """Parse data for DGAC clustering."""

    dataset = load_dataset("multi_woz_v22")
    
    utterances = []
    lengths = []
    speaker = []
    dialogues = []
    for dia in tqdm(dataset['train']['turns'], desc='Parsing MultiWOZ'):
        utterances.extend(dia['utterance'])
        lengths.append(len(dia['utterance']))
        speaker.extend(dia['speaker'])
        dialogues.append(dia['utterance'])
    
    if not os.path.exists('clust-data'):
        os.makedirs('clust-data')
    json.dump(utterances, open('clust-data/utterances.json', 'w'))
    json.dump(lengths, open('clust-data/rle.json', 'w'))
    json.dump(speaker, open('clust-data/speaker.json', 'w'))
    json.dump(dialogues, open('clust-data/dialogues.json', 'w'))


if __name__ == '__main__':
    parse()