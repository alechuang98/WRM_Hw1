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
    parser.add_argument("-k", type=float, required=True)
    parser.add_argument("-b", type=float, required=True)

    args = parser.parse_args()
    t0 = time.time()
    queryParser = QueryParser(args.m)
    vsm = VSM(args.m, k=args.k, b=args.b)
    rocchio = Rocchio(Param.ALPHA, Param.BETA, Param.GAMMA, Param.TOP_K, Param.LAST_K, args.m, args.d)
    ev = Eval(args.m, ansPath="queries/ans_train.csv")
    queryIds, queries = queryParser.getQueries(args.i)

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
    

    if args.r:
        for i in range(Param.ITERS):
            t0 = time.time()
            for j in range(len(q)):
                res[j] = rocchio.update(res[j], vsm, queryParser)

    print("[b = %2f | k = %2f]: %3f" % (args.b, args.k, ev.test(res)))

if __name__ == "__main__":
    main()