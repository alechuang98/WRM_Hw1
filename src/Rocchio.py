from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Rocchio(object):
    def __init__(self, alpha, beta, gamma, top_k, last_k):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k
        self.last_k = last_k
        print("Rocchio object init finish")

    def update(self, query, tfidf):
        cosSim = cosine_similarity(tfidf, query.reshape(1, -1))
        rnk = np.flip(cosSim.argsort(axis=0), axis=0)
        dr = tfidf[rnk[: self.top_k, :]].mean(axis=0)
        dn = tfidf[rnk[-self.last_k :, :]].mean(axis=0)
        return query * self.alpha + dr * self.beta - dn * self.gamma