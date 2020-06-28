#encoding=utf-8

import lucene
import random
import codecs
import sys, os, lucene, threading, time
from datetime import datetime

from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory

import jieba

class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)

class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, lines, storeDir, analyzer):

        if not os.path.exists(storeDir):
            os.mkdir(storeDir)

        store = SimpleFSDirectory(Paths.get(storeDir))
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        self.indexDocs(lines, writer)
        ticker = Ticker()
        print('commit index')
        threading.Thread(target=ticker.run).start()
        writer.commit()
        writer.close()
        ticker.tick = False
        print('done')

    def indexDocs(self, lines, writer):

        ts = []
        for i in range(4):
            t1 = FieldType()
            t1.setStored(True)
            t1.setTokenized(True)
            t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
            t1.storeTermVectors()
            ts.append(t1)

        file_names = ["all", "context", "last", "response"]
        for line in lines:
            doc = Document()
            for i in range(4): 
                doc.add(Field(file_names[i], line[i], ts[i]))
            writer.addDocument(doc)


def get_data(file_name):
    f = codecs.open(file_name, "r", "utf-8")
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line[0]=="1"]
    print("Before remove duplicated: ", len(lines))
    line_all_dict = dict()
    lines_tmp = []
    for line in lines:
        if line[-20:] in line_all_dict:
            continue
        else:
            lines_tmp.append(line)
            line_all_dict[line[-20:]] = 1
    lines = lines_tmp
    print("After remove duplicated: ", len(lines))
    # all context last response
    res = [(" ".join(line.split("\t")[1:]) , "\t".join(line.split("\t")[1:-1]), line.split("\t")[-2], line.split("\t")[-1]) for line in lines]
    #res = res[:-20000]
    return res
    

def build_index():
    print('lucene', lucene.VERSION)
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    lines = get_data("../../data/ecommerce/train.txt")
    IndexFiles(lines, "data/ecommerce", StandardAnalyzer())


if __name__=="__main__":
    build_index()

