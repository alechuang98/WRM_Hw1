import Parser
import Param
import os
import math
import numpy as np
import logging

class VSM(object):
    def __init__(self, model, k, b):
        self.k = k
        self.b = b
        self.inverted = Parser.getInverted(os.path.join(model, "inverted-file"), cache=Param.CACHE)
        self.fileLen = Parser.getFileLen(self.inverted)
        self.avgLen = np.sum(self.fileLen) / Param.FILE_NUM
        # print("VSM object init finish")

    def getTF(self, query):
        ctd = np.zeros((Param.FILE_NUM, len(query)))
        qr = [Parser.hashBigram(x[0], x[1]) for x in query]
        for key, value in self.inverted.items():
            if key in qr:
                indx = qr.index(key)
                for v in value:
                    ctd[v[0]][indx] += v[1]
        tf = (self.k + 1) * ctd / (ctd.T + self.k * (1 - self.b + self.b * self.fileLen / self.avgLen)).T
        # tf = np.nan_to_num((tf.T / self.k  ).T, 0)
        tf = np.nan_to_num(tf, 0)
        return tf

    def getIDF(self, query):
        qr = [Parser.hashBigram(x[0], x[1]) for x in query]
        idf = np.zeros(len(qr))
        for i, token in enumerate(qr):
            if token not in self.inverted:
                idf[i] = 0
            else:
                idf[i] = math.log((Param.FILE_NUM - len(self.inverted[token]) + 0.5) / (len(self.inverted[token]) + 0.5) + 1)
        return idf