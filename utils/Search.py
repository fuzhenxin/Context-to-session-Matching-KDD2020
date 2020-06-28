#encoding=utf-8
import sys, os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
import random
import codecs

class Search():
    def __init__(self, file_name):
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        directory = SimpleFSDirectory(Paths.get(os.path.join(base_dir, file_name)))
        self.reader = DirectoryReader.open(directory)
        self.searcher = IndexSearcher(self.reader)
        self.analyzer = StandardAnalyzer()
        self.error_count = 0
        #import pdb; pdb.set_trace()

    def search_one(self, line, search_type, return_count=1):
        line = line.replace("?", "").replace("OR", "").replace("AND", "").replace("NOT", "")
        line = QueryParser.escape(line)
        search_types = ["all", "context", "last", "response"]
        query = QueryParser(search_types[search_type], self.analyzer).parse(line)
        scoreDocs = self.searcher.search(query, return_count).scoreDocs
        if len(scoreDocs)==0:
            self.error_count += 1
            print("Error in search: ",line)
            return None
        res = []
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            score = scoreDoc.score
            res.append((doc.get("all"), doc.get("context"), doc.get("last"), doc.get("response"), score))
        return res


def test(file_name):
    search = Search(file_name)
    test_cases = ["好 的 亲退 了"]
    for test_case in test_cases:
        print(test_case)
        for j in range(3):
            print("test_type: ",j)
            res = search.search_one(test_case, j)
            for i in res[0]:
                print(i)

if __name__== "__main__":
    test("data/ecommerce")