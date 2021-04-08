import os
import time
import pickle
import xml.etree.ElementTree as et
import numpy as np
import Param

def hashBigram(a, b):
    if b == -1:
        b = 0
    return a + b * (Param.TOTAL_VOCAB + 1)

def getInverted(file, tmpPath="tmp/inverted-file.pkl", cache=0):
    res = {}
    if cache == 1 and os.path.isfile(tmpPath):
        with open(tmpPath, "rb") as f:
            res = pickle.load(f)
    else:
        with open(file, "r", encoding="utf8") as f:
            while line := f.readline():
                token = [int(x) for x in line.strip("\n").split(" ")]
                key = hashBigram(token[0], token[1])
                res[key] = []
                for i in range(token[2]):
                    info = f.readline()
                    inf = [int(x) for x in info.strip("\n").split(" ")]
                    res[key].append(inf)
        if cache == 1:
            with open(tmpPath, "wb") as f:
                pickle.dump(res, f)
    return res

def getFileLen(inverted):
    res = np.zeros(Param.FILE_NUM).astype('int32')
    for key, value in inverted.items():
        if key <= Param.TOTAL_VOCAB:
            for v in value:
                res[v[0]] += v[1]
    return res
