import plotly.express as px
from sklearn.manifold import TSNE
import pickle
from dgac_clustering import Clusters
import json
import numpy as np
import pandas as pd
from redlines import Redlines
from IPython.display import Markdown, display
from Levenshtein import distance, ratio, jaro
from typing import Literal
from difflib import HtmlDiff
import html2markdown
import os


def show_clusters(is_system) -> None:
    """
    Show clusters, obtained from train multiwoz with DGAC clustering. Scatter plot with utterances appearing on hover.
    Also assigns names to clusters using TF-IDF and saves them to clust-data directory as .json files.

    Params
    ------
        is_system: bool, flag, indicating what part of clusters to visualize, user of system
    """

    # load necessary data
    clusterer: Clusters = pickle.load(open('clust-data/dgac_clusterer.pickle', 'rb'))
    X = np.load('clust-data/sentence_embeddings.npy')
    speaker = np.array(json.load(open('clust-data/speaker.json', 'r')))

    mask = (speaker == is_system)
        
    projected = pd.DataFrame(TSNE().fit_transform(X[mask]))

    # get names of clusters via tf-idf
    texts = [txt for i, txt in enumerate(json.load(open('clust-data/utterances.json', 'r'))) if speaker[i] == is_system]
    labels = clusterer.labels[speaker == 1] - is_system * clusterer.n_clusters // 2
    apndx = 'system' if is_system else 'user'
    cluster_names = json.load(open(f'clust-data/cluster-tfidf-names-{apndx}.json', 'w'))

    projected['clust_name'] = '-'
    for i in range(clusterer.n_clusters // 2):
        projected.loc[labels == i, 'clust_name'] = cluster_names[i]
    
    # visualize
    projected['label'] = labels
    projected['text'] = texts
    fig = px.scatter(
        projected, x=0, y=1, color='label',
        hover_name='clust_name',
        hover_data={'label': True, 'text': True},
        title=f'{apndx} utterances'
    )

    fig.update_layout()
    fig.show()


def show_augmented(i, name) -> None:
    """
    Show difference between original dialogue and augmented.
    
    Params
    ------
    - i: int, index of dialogue to construct from aug-data/utterances.json
    - name: file name to extract augmented dialogue from as f'aug-data/{name}.csv'
    """
    original = json.load(open('aug-data/original.json', 'r'))[i]
    augmented = json.load(open(f'aug-data/{name}.json', 'r'))[i]
    
    speaker_alias = "AB"
    orig = '\n'.join([f'[{speaker_alias[item["speaker"]]}] {item["utterance"]}' for item in original])
    aug = '\n'.join([f'[{speaker_alias[item["speaker"]]}] {item["utterance"]}' for item in augmented])

    display(Markdown(Redlines(orig, aug).output_markdown))


def show_similarities(
        i, name,
        func,
        view: Literal['redlines', 'two-sided'] = 'redlines',
        with_intent=True,
        display_mkdown=True,
        return_raw=False
    ):
    """
    Show difference between original dialogue and augmented with cluster name provided for each utterance and dialogues similarity.
    
    Params
    ------
    - i: int, index of dialogue to construct from aug-data/utterances.json
    - name: name of json from `aug-data`
    - func: similarity function over bag of nodes vectorization 
    """

    # load texts
    orig_obj = json.load(open('aug-data/original.json', 'r'))[i]
    aug_obj = json.load(open(f'aug-data/{name}.json', 'r'))[i]

    # load cluster labels
    orig_labels = json.load(open(f'aug-data/clust-labels-original.json'))[i]
    aug_labels = json.load(open(f'aug-data/clust-labels-{name}.json'))[i]

    # load cluster names
    names = json.load(open(f'clust-data/cluster-tfidf-names-user.json'))
    names.extend(json.load(open(f'clust-data/cluster-tfidf-names-system.json')))

    # parse
    def parse_item(item_lab):
        item, lab = item_lab
        speaker_alias = "AB"
        spe = f"[{speaker_alias[item['speaker']]}]"
        intent = f"[label: {lab}] [name: {names[lab]}]"
        ut = item['utterance']
        if with_intent:
            return ' '.join([spe, intent, ut])
        return ' '.join([spe, ut])

    orig = list(map(parse_item, zip(orig_obj, orig_labels)))
    aug = list(map(parse_item, zip(aug_obj, aug_labels)))

    # display some edit distance
    orig_txt = ' '.join(orig)
    aug_txt = ' '.join(aug)
    levenstein = distance(orig_txt, aug_txt)
    similarity_ratio = ratio(orig_txt, aug_txt)
    similarity_jaro = jaro(orig_txt, aug_txt)
    sim = f'{levenstein=}, {similarity_ratio=:.3f}, {similarity_jaro=:.3f}\n\n'
    if with_intent:
        orig_vecs = np.load(f'aug-data/vectors-original.npy')[i]
        aug_vecs = np.load(f'aug-data/vectors-{name}.npy')[i]
        intent_similarity = func(orig_vecs, aug_vecs)
        sim = f'{intent_similarity=:.3f}, ' + sim

    if view == 'redlines':
        raw = Redlines('\n'.join(orig), '\n'.join(aug)).output_markdown
    elif view == 'two-sided':
        raw = HtmlDiff(wrapcolumn=70).make_table(orig, aug)
        raw = html2markdown.convert(raw)
    else:
        raise ValueError('unexpected `view` value')
    
    raw = sim + raw

    if display_mkdown:
        display(Markdown(raw))

    if return_raw:
        return raw


def make_demo(
        name,
        func,
        view: Literal['redlines', 'two-sided'] = 'redlines',
        with_intent=True,
        n_dialogues=15
    ):
    """Save all aug visualisations to .md file"""
    res = f"# Demo of {name}\n"
    for i in range(n_dialogues):
        mkdown = show_similarities(i, name, func, view, with_intent, display_mkdown=False, return_raw=True)
        res += f'\n## dialogue \# {i}\n{mkdown}\n'
    if not os.path.exists('demo'):
        os.makedirs('demo')
    with open(f'demo/{name}.md', 'w') as f:
        f.write(res)
