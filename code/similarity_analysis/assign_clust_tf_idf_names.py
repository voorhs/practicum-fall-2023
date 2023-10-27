from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np
import pickle
from dgac_clustering import Clusters
import json


def assign_cluster_names(labels, texts) -> List[str]:
    n_clusters = len(np.unique(labels))    

    cluster_utterances = []
    for i in range(n_clusters):
        cluster_texts = [txt for j, txt in enumerate(texts) if labels[j] == i]
        cluster_utterances.append('. '.join(cluster_texts))

    vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
    vec = vectorizer.fit_transform(cluster_utterances).toarray()

    titles = np.zeros_like(vec)
    for i, topk in enumerate(np.argpartition(vec, kth=[-1, -2, -3], axis=1)[:, -3:]):
        titles[i, topk] = vec[i, topk]
    
    return [', '.join(title) for title in vectorizer.inverse_transform(titles)]


if __name__ == '__main__':
    # load necessary data
    clusterer: Clusters = pickle.load(open('clust-data/dgac_clusterer.pickle', 'rb'))
    speaker = np.array(json.load(open('clust-data/speaker.json', 'r')))        

    # separate system and user utterances from each other
    system_uts = []
    user_uts = []
    for i, ut in enumerate(json.load(open('clust-data/utterances.json', 'r'))):
        if speaker[i] == 0:
            user_uts.append(ut)
        elif speaker[i] == 1:
            system_uts.append(ut)
        else:
            raise ValueError(f'something\'s wrong with "clust-data/speaker.json", speaker value must be either 0 (user) or 1 (system), but got {speaker[i]}')
    
    # get names via tf-idf
    system_labels = clusterer.labels[speaker == 1] - clusterer.n_clusters // 2
    user_labels = clusterer.labels[speaker == 0]
    system_names = assign_cluster_names(system_labels, system_uts)
    user_names = assign_cluster_names(user_labels, user_uts)

    json.dump(system_names, open(f'clust-data/cluster-tfidf-names-system.json', 'w'))
    json.dump(user_names, open(f'clust-data/cluster-tfidf-names-user.json', 'w'))
