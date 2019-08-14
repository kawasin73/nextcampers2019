import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def data_loader(f_name, l_name):
    with open(f_name, mode='r', encoding='utf-8') as f:
        data = list(set(f.readlines()))
        label = [l_name for i in range(len(data))]
        return data, label

def csv_write(f_name, label, data):
    with open(f_name, mode='w', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        for l, d in zip(label, data):
            w.writerow([d, l])


XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_2.csv'
XSS2_TRAIN_FILE = 'dataset/train_level_2.csv'
XSS2_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'

NON_LABEL_FILE = 'dataset/level_3_non_label.csv'
DOC2VEC_MODEL_FILE = 'dataset/doc2vec'

STOP_WORDS = []

FMT_TAG = "</*[a-zA-Z0-9]+|>"
FMT_HTML_ESCAPE = "&[a-zA-Z0-9]+;"
FMT_SYMBOL = "=|:|;|\"|\\\\\\\\|\\\\|\(|\)|`|&|#"

FORMAT = "(%s|%s|%s)" %(FMT_TAG, FMT_HTML_ESCAPE, FMT_SYMBOL)

def filter_not_script(w):
    return (w[0] != "<") or (w == "<script")

def parse_text(text):
    text = text.lower()
    parsed = re.split(FORMAT, text.rstrip("\n"))
    # remove white space in head and tail
    parsed = map(lambda x : x.strip(), parsed)
    # remove empty string
    parsed = filter(None, parsed)
    # filter not <script tag
    parsed = filter(filter_not_script, parsed)
    # remove ">"
    parsed = filter(lambda x : x != ">", parsed)
    return list(parsed)

def run():
    data, _ = data_loader(NON_LABEL_FILE, "none")
    # https://medium.com/@MSalnikov/text-clustering-with-k-means-and-tf-idf-f099bcf95183

    tfidf_vectorizer = TfidfVectorizer(tokenizer=parse_text)
    tfidf = tfidf_vectorizer.fit_transform(data)
    kmeans = KMeans(n_clusters=2).fit_predict(tfidf)

    csv_write("dataset/cluster", kmeans, [text.rstrip("\n") for text in data])


if __name__ == '__main__':
    run()
