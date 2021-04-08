from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import numpy as np
import Param
import os

class Eval(object):
    def __init__(self, model, ansPath=None):
        self.model = model
        self.ansPath = ansPath
        self.file2indx = {}
        self.indx2file = []
        with open(os.path.join(self.model, "file-list"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                val = line.strip("\n").split(os.sep)[-1].lower()
                self.indx2file.append(val)
                self.file2indx[val] = i
        self.fileNum = len(self.indx2file)
        if self.ansPath:
            self.ans = []
            with open(self.ansPath, "r") as f:
                lines = f.readline()
                lines = f.readlines()
                for i, line in enumerate(lines):
                    self.ans.append(np.zeros(self.fileNum).astype(bool))
                    for file in line.strip("\n").split(",")[1].split(" "):
                        self.ans[i][self.file2indx[file]] = True
            self.ans = np.array(self.ans)
                    
    
    def getResult(self, query, tfidf):
        return np.sum(tfidf, axis=1)
    
    def test(self, results):
        total = 0
        idd = 0
        for x, y in zip(results, self.ans):
            print(idd, average_precision_score(y, x))
            total += average_precision_score(y, x)
            idd += 1
        return total / self.ans.shape[0]
    
    def output(self, results, filePath):
        lines = []
        lines.append("query_id,retrieved_docs\n")
        for i in range(len(results)):
            line = "%03d," % (i + 11)
            rnk = np.flip(results[i].argsort(axis=0), axis=0)
            line += " ".join([self.indx2file[x] for x in rnk[: Param.MAX_RESULT].ravel().tolist()]) + "\n"
            lines.append(line)
        with open(filePath, "w") as f:
            f.writelines(lines)
        