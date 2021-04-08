import argparse
import time
import numpy as np
import Param
from VSM import VSM
from QueryParser import QueryParser
from Rocchio import Rocchio
from Eval import Eval
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action='store_true', default=False, help="Revelence feedback")
    parser.add_argument("-i", type=str, default="queries/query-train.xml", help="Input query")
    parser.add_argument("-o", type=str, default="results/res_train.xml", help="Output")
    parser.add_argument("-m", type=str, default="model")
    parser.add_argument("-d", type=str, default="CIRB010")

    args = parser.parse_args()
    t0 = time.time()
    queryParser = QueryParser(args.m)
    vsm = VSM(args.m, k=Param.K, b=Param.B)
    rocchio = Rocchio(Param.ALPHA, Param.BETA, Param.GAMMA, Param.TOP_K, Param.LAST_K, args.m, args.d)
    ev = Eval(args.m, ansPath="queries/ans_train.csv")
    print("init time", time.time() - t0)
    queries = queryParser.getQueries(args.i)

    tfidf = [None] * len(queries)
    q = [None] * len(queries)
    res = [None] * len(queries)

    for j, query in enumerate(queries):
        tf = vsm.getTF(query)
        idf = vsm.getIDF(query)
        tfidf[j] = tf * idf
        q[j] = idf
    for j in range(len(q)):
        res[j] = ev.getResult(q[j], tfidf[j])
    print("[Iter %2d]: %3f" % (-1, ev.test(res)))

    for i in range(Param.ITERS):
        t0 = time.time()
        for j in range(len(q)):
            res[j] = rocchio.update(res[j])
        print("[Iter %2d]: %3f | Using %3f second" % (i, ev.test(res), time.time() - t0))


if __name__ == "__main__":
    main()