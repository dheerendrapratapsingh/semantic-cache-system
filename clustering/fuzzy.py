from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyCluster:

    def __init__(self, n_clusters=20):

        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=42
        )

    def fit(self, embeddings):

        self.model.fit(embeddings)

    def get_cluster_distribution(self, embedding):

        probs = self.model.predict_proba([embedding])

        return probs[0]

    def dominant_cluster(self, embedding):

        probs = self.get_cluster_distribution(embedding)

        return int(np.argmax(probs))