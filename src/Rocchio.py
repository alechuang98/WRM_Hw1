from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import xml.etree.ElementTree as et
import os
import Param

class Rocchio(object):
    def __init__(self, alpha, beta, gamma, top_k, last_k, model, d):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k
        self.last_k = last_k
        self.indx2file = None
        with open(os.path.join(model, "file-list"), "r") as f:
            lines = f.readlines()
            self.indx2file = [line.strip(" \n") for line in lines]
        self.d = d
        print("Rocchio object init finish")

    def update_old(self, query, tfidf):
        cosSim = cosine_similarity(tfidf, query.reshape(1, -1))
        rnk = np.flip(cosSim.argsort(axis=0), axis=0)
        dr = tfidf[rnk[: self.top_k, :]].mean(axis=0)
        dn = tfidf[rnk[-self.last_k :, :]].mean(axis=0)
        return query * self.alpha + dr * self.beta - dn * self.gamma

    def update(self, result):
        rnk = np.flip(result.argsort(axis=0), axis=0)
        
        freq = {}
        for i in rnk[: self.top_k]:
            result[i] += 1e9
            tree = et.parse(os.path.join(self.d, self.indx2file[i]))
            root = tree.getroot()
            title = root.find("doc/title").text.strip("\n")
            for c in Param.stopWords:
                title = title.replace(c, "")
        return result
                
            