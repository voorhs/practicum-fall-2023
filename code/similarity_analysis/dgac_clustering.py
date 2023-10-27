from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial.distance import cdist


class Clusters:
    ''' 
        Two-stage utternces clustering (DGAC paper).

        Attributes
        ----------
        - embeddings_user, embedding_system: trained w2v embeddings of clusters
        - centroids_user, centroids_system: mean sentence embeddings of utterances for each cluster
        - labels: cluster labels of train utterances
    '''

    def __init__(self, n_first_clusters, n_second_clusters=None):
        self.n_first_clusters = n_first_clusters
        self.n_second_clusters = n_second_clusters

    def _first_clustering(self, X, speaker):
        '''
        KMeans clustering over sentence embeddings of utterances.

        Params
        ------
            X: np.ndarray of size (n_utterances, emb_size)
        Return
        ------
            centroids: np.ndarray of size (self.n_first_clusters, emb_size)
            labels: np.ndarray of size (n_utterances,)
        '''

        kmeans_user = KMeans(n_clusters=self.n_first_clusters, n_init=5).fit(X[speaker == 0])
        kmeans_system = KMeans(n_clusters=self.n_first_clusters, n_init=5).fit(X[speaker == 1])
        return (
            kmeans_user.cluster_centers_,
            kmeans_user.labels_,
            kmeans_system.cluster_centers_,
            kmeans_system.labels_
        )

    def _cluster_embeddings(self, labels, dialogues_rle):
        """
        Train word2vec: word = cluster labels, sentence = dialogue.

        Params
        ------
            labels: np.ndarray of size (n_utterances,), cluster labels of each utterance from `dialogues`
            dialogues: list[list[str]] where list[str] is a single dialogue
        
        Return
        ------
            np.ndarray of size (n_clusters, 100)
        """
        i = 0
        array_for_word2vec = []

        for dia_len in dialogues_rle:
            array_for_word2vec.append([str(clust_label) for clust_label in labels[i:i+dia_len]])
            i += dia_len

        w2v_model = Word2Vec(
            sentences=array_for_word2vec,
            sg=0,
            min_count=1,
            workers=4,
            window=10,
            epochs=20
        )
        
        n_clusters = len(np.unique(labels))
        return np.stack([w2v_model.wv[str(i)] for i in range(n_clusters)])

    def _second_clustering(self, X, speaker, first_embeddings_user, first_embeddings_system, first_labels_user, first_labels_system):
        """
        KMeans clustering over word2vec embeddings of first stage clusters.

        Params
        ------
            X: np.ndarray of size (n_utterances, emb_size)
            first_embeddings: np.ndarray of size (self.n_first_clusters, 100)
            first_labels: np.ndarray of size (n_utterances,)
        
        Return
        ------
            centroids: np.ndarray of size (self.n_second_clusters, emb_size)
            labels: np.ndarray of size (n_utterances,)
        """
        
        # for user
        is_user = (speaker == 0)

        kmeans_user = KMeans(
            n_clusters=self.n_second_clusters,
            n_init=5,
            algorithm="elkan"
        ).fit(first_embeddings_user)

        second_labels_user = kmeans_user.labels_[first_labels_user]
        
        centroids_user = []
        for i in range(self.n_second_clusters):
            centroids_user.append(X[is_user][second_labels_user == i].mean(axis=0))

        # for system
        kmeans_user = KMeans(
            n_clusters=self.n_second_clusters,
            n_init=5,
            algorithm="elkan"
        ).fit(first_embeddings_system)

        second_labels_system = kmeans_user.labels_[first_labels_system]
        
        centroids_system = []
        for i in range(self.n_second_clusters):
            centroids_system.append(X[~is_user][second_labels_user == i].mean(axis=0))

        return np.stack(centroids_user), second_labels_user, np.stack(centroids_system), second_labels_system

    def fit(self, X, speaker, dialogues_rle):
        '''
        Params
        ------
            X: np.ndarray of size (n_utterances, emb_size)
        '''
        is_user = (speaker == 0)

        print("First stage of clustering has begun...")
        self._first_centroids_user, self._first_labels_user, self._first_centroids_system, self._first_labels_system = self._first_clustering(X, speaker)
        
        first_labels = np.empty(len(X), dtype=np.int16)
        first_labels[is_user] = self._first_labels_user
        first_labels[~is_user] = self._first_labels_system + self.n_first_clusters
        first_labels = first_labels

        print('Training first stage word2vec...')
        first_embeddings = self._cluster_embeddings(first_labels, dialogues_rle)
        self._first_embeddings_user, self._first_embeddings_system = np.split(first_embeddings, 2)

        self.embeddings_user = self._first_embeddings_user
        self.embeddings_system = self._first_embeddings_system
        self.centroids_user = self._first_centroids_user
        self.centroids_system = self._first_centroids_system
        self.labels_user = self._first_labels_user
        self.labels_system = self._first_labels_system
        self.labels = first_labels
        self.n_clusters = len(np.unique(self.labels))

        if self.n_second_clusters is None:
            return self
        
        print("Second stage of clustering has begun...")
        self._second_centroids_user, self._second_labels_user, self._second_centroids_system, self._second_labels_system = self._second_clustering(
            X,
            speaker,
            self._first_embeddings_user,
            self._first_centroids_system,
            self._first_labels_user,
            self._first_labels_system
        )

        second_labels = np.empty(len(X), dtype=np.int16)
        second_labels[is_user] = self._second_labels_user
        second_labels[~is_user] = self._second_labels_system + self.n_second_clusters
        second_labels = second_labels
        
        print('Training second stage word2vec...')
        second_embeddings = self._cluster_embeddings(second_labels, dialogues_rle)
        self._second_embeddings_user, self._second_embeddings_system = np.split(second_embeddings, 2)

        self.embeddings_user = self._second_embeddings_user
        self.embeddings_system = self._second_embeddings_system
        self.centroids_user = self._second_centroids_user
        self.centroids_system = self._second_centroids_system
        self.labels_user = self._second_labels_user
        self.labels_system = self._second_labels_system
        self.labels = second_labels
        self.n_clusters = len(np.unique(self.labels))

        return self

    def predict(self, X, speaker):
        """
        Predict cluster label for given utterances embeddings.
        
        Params
        ------
            X: np.ndarray of size (n_utterances, emb_size)
            speaker: iterable of size (n_utterances,)
        
        Return
        ------
            np.ndarray of size (n_utterances,)
        """
        is_user = (np.array(speaker) == 0)

        labels_user = cdist(X[is_user], self.centroids_user, metric='euclidean').argmin(axis=1)
        labels_system = cdist(X[~is_user], self.centroids_system, metric='euclidean').argmin(axis=1)

        return labels_user, labels_system


if __name__ == "__main__":
    import json
    import pickle
    from datetime import datetime

    clusterer = Clusters(
        n_first_clusters=100,
        n_second_clusters=15,
    )

    X = np.load('clust-data/sentence_embeddings.npy')
    speaker = np.array(json.load(open('clust-data/speaker.json', 'r')))
    dialogues_rle = json.load(open('clust-data/rle.json', 'r'))

    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    clusterer.fit(X, speaker, dialogues_rle)

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    with open('clust-data/dgac_clusterer.pickle', 'wb') as handle:
        pickle.dump(clusterer, handle, protocol=pickle.HIGHEST_PROTOCOL)