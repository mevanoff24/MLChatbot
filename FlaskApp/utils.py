import os 
import numpy as np
import pandas as pd
from pathlib import Path
import csv
import pickle
import annoy
from datetime import datetime
from collections import defaultdict
import sqlite3
import re
import html
from sklearn.metrics.pairwise import linear_kernel


from config import * 

import nltk
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))


def avg_word_vectors(question, embeddings, dim):
    words_embedding = [embeddings[word] for word in question.lower().split() if word in embeddings]
    if not words_embedding:
        return np.zeros(dim)
    words_embedding = np.array(words_embedding).astype(np.float32)
    return words_embedding.mean(axis=0)

def average_tfidf_vectors(question, embeddings, dim, vect, idf_scores):
    # get idf weights
    split_question = [word for word in question.lower().split() if word in embeddings]
    if not split_question:
        return np.zeros(dim).astype(np.float32)
    words_embedding = np.zeros((dim, len(split_question))).astype(np.float32)
    for i, token in enumerate(split_question):
        if token in embeddings:
            embed_score = embeddings[token]
        else: embed_score = 0
        idf_score = idf_scores[token]
        # word vectors multiply by their TF-IDF scores
        words_embedding[:, i] = embed_score * idf_score    
    return words_embedding.mean(axis=1)


def get_embeddings(filename):
    embeddings = {}
    with open(MODEL_PATH/filename, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        embed_list = list(reader)
    for line in embed_list:
        embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
        
    dim = len(embeddings['code'])
    return embeddings, dim


def unpickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



re1 = re.compile(r'  +')
# def clean_title(text, remove_html=False, other=False):
def clean_text(text, remove_html=False, other=False):
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    
    if remove_html:
        x = re.sub(r'<code>[^>]*</code>', '', text)
        x = re.sub(r'<[^>]*>', '', x)
        x = re.sub(r'[^A-Za-z0-9]', ' ', x)
    text = text.lower()
    text = text.replace('π', 'pi').replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
            'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '').replace('<unk>','u_n').replace(' @.@ ','.').replace(
            ' @-@ ','-').replace('\\', ' \\ ').replace('"',"'").replace('\n', ' ').replace('\r', ' ')
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])
    return re1.sub(' ', html.unescape(text).strip())



def get_top_preds(X, clf, class_map, k=3, cutoff=0.8):
        preds = clf.predict_proba(X)
        sorted_preds = np.argsort(preds)[:,::-1][0][:3]
        top_scores = [preds[0][i] for i in sorted_preds]
        if top_scores[0] > cutoff:
            # if top_scores above threshold only need to return top 
            return [class_map[sorted_preds[0]]]
        else:
            return [class_map[p] for p in sorted_preds]


def tfidf_cosine_similarities(X, question, topk=5):
    tfid_vectorizer = unpickle(MODEL_PATH/'tf_idf_python_title_stopwords.pkl')

    tfidf_X = tfid_vectorizer.transform(X)
    tfidf_question = tfid_vectorizer.transform([question])

    cosine_similarities = linear_kernel(tfidf_question, tfidf_X).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-topk:-1]
    print(related_docs_indices)
    return final_df.loc[related_docs_indices]

