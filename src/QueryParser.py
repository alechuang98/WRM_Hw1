import os
import xml.etree.ElementTree as et
import logging

class QueryParser(object):
    def __init__(self, model):
        self.dic = {}
        with open(os.path.join(model, "vocab.all"), "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # english doesn't matter
                self.dic[line.strip("\n")] = i
        print("QueryParser object init finish")

    def string2index(self, string):
        return [self.dic[x] for x in string]

    def getQueries(self, filePath="queries/query-train.xml"):
        tree = et.parse(filePath)
        root = tree.getroot()
        res = []
        for t in root.findall("topic"):
            query = [t.find("title").text]
            concepts = t.find("concepts").text.strip("。 \n").split("、")
            for concept in concepts:
                converted = self.string2index(concept)
                for i in range(len(converted) - 1):
                    query.append([converted[i], converted[i + 1]])

                if concept == "ＢＯＴ":
                    query.append([self.dic["BOT"], -1])
                if  concept == "ＮＢＡ":
                    query.append([self.dic["NBA"], -1])
            res.append(query)
        return res
