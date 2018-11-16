import os 
import numpy as np
import pandas as pd
from pathlib import Path
import csv
import pickle
from datetime import datetime
from collections import defaultdict
import re
import html
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


from config import * 

import nltk
from nltk.corpus import stopwords
# stopword set 
stopwords_set = set(stopwords.words('english'))


def avg_word_vectors(question, embeddings, dim):
    """Compute the average word vector based on individual word vectors trained from StarSpace"""
    words_embedding = [embeddings[word] for word in question.lower().split() if word in embeddings]
    # return 0s if no word included
    if not words_embedding:
        return np.zeros(dim)
    words_embedding = np.array(words_embedding).astype(np.float32)
    # simple mean return
    return words_embedding.mean(axis=0)


def average_tfidf_vectors(question, embeddings, dim, vect, idf_scores):
    """Average the weighted word vectors trained from StarSpace with the weight of a word given by tf-idf"""
    # get idf weights
    split_question = [word for word in question.lower().split() if word in embeddings]
    # return 0s if no word included
    if not split_question:
        return np.zeros(dim).astype(np.float32)
    # initialize empty array
    words_embedding = np.zeros((dim, len(split_question))).astype(np.float32)
    for i, token in enumerate(split_question):
        if token in embeddings:
            # get vector
            embed_score = embeddings[token]
        else: embed_score = 0
        # get tf-idf score
        idf_score = idf_scores[token]
        # word vectors weighted by their TF-IDF scores
        words_embedding[:, i] = embed_score * idf_score 
    # mean return 
    return words_embedding.mean(axis=1)


def avg_word_vectors_OOV(question, embeddings, fasttext_embeddings, dim):
    """
    Compute the average word vector based on individual word vectors trained from FastText
    If the word is not included in the FastText word vectors we find the most similar vector from FastText
    """
    words_embedding = []
    for word in question.lower().split():
        if word in embeddings:
            words_embedding.append(embeddings[word])
        else:
            # compute similarity between this unknown word and other FastText word vectors
            cosine_similarities = linear_kernel([fasttext_embeddings[word]], np.array(word_list)[:, 1:])
            related_docs_indices = cosine_similarities.argsort()[:-10:-1]
            words_embedding.append(word_list[related_docs_indices[0][0]][1:])
            
    if not words_embedding:
        return np.zeros(dim)
    words_embedding = np.array(words_embedding).astype(np.float32)
    # mean return 
    return words_embedding.mean(axis=0)


def average_tfidf_vectors_OOV(question, embeddings, fasttext_embeddings, dim, vect, idf_scores):
    """
    Compute the tf-idf weighted average word vector based on individual word vectors trained from FastText
    If the word is not included in the FastText word vectors we find the most similar vector from FastText
    """
    split_question = [word for word in question.lower().split()]
    
    if not split_question:
        return np.zeros(dim).astype(np.float32)
    
    words_embedding = np.zeros((dim, len(split_question))).astype(np.float32)
    
    for i, token in enumerate(split_question):
        if token in embeddings:
            embed_score = embeddings[token]
        else: 
            # compute similarity between this unknown word and other FastText word vectors
            cosine_similarities = linear_kernel([fasttext_embeddings[token]], np.array(word_list)[:, 1:])
            related_docs_indices = cosine_similarities.argsort()[:-10:-1]
            embed_score = word_list[related_docs_indices[0][0]][1:]
        # get idf weights
        if idf_scores[token]:
            idf_score = idf_scores[token]
        else: idf_score = 1
        # word vectors multiply by their TF-IDF scores
        words_embedding[:, i] = embed_score * idf_score 
    # mean return
    return words_embedding.mean(axis=1)


def get_embeddings(filename, array=False):
    """Load Word Embedding into memory. Return dictionary and embedding dim"""
    embeddings = {}
    with open(MODEL_PATH/filename, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        embed_list = list(reader)
    for line in embed_list:
        embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
        
    dim = len(embeddings['code'])
    if array:
        return embeddings, embed_list, dim
    return embeddings, dim


def unpickle(filename):
    """Unpickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


re1 = re.compile(r'  +')
def clean_text(text, remove_html=False, other=False):
    """
    Clean text field. Remove puctuations, weird chars, stopwords
    
    Args:
        text (str): text
        remove_html (boolean): Remove all 'code' aspects of text if True
    
    Returns:
        text (str): Cleaned text
    """
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
    """
    Predict and return the top k predictions and confidence scores 
    from Tag Classifier 
    
    Args:
        clf (object): sklearn model with `predict_proba` attribute
        class_map (dict): Sklearn label encoding mapping of class index to class name
        k (int): number of predictions to return
        cutoff (float): If model is this confident, we only return top1 prediction
    """
    # predict probabilities 
    preds = clf.predict_proba(X)
    # sort by top k
    sorted_preds = np.argsort(preds)[:,::-1][0][:k]
    top_scores = [preds[0][i] for i in sorted_preds]
    print(top_scores)
    if top_scores[0] > cutoff:
        # if top_scores above threshold only need to return top 
        return [class_map[sorted_preds[0]]], top_scores
    else:
        return [class_map[p] for p in sorted_preds], top_scores


def joined(row):
    """Append row to numpy array"""
    arr = []
    return np.append(arr, row)


def compute_similarity(question, X):
    """Compute the linear kernel between X and question"""
    return linear_kernel(question, X).flatten()


def matching_word(row, question):
    """Return True if token (word) is in question"""
    for token in question.split():
        if token in row.split():
            return True
    return False
           
    
def tfidf_cosine_similarities(df, col, question, topk=5):
    """
    Tf-idf fit on `col`, transform user question and compute linear kernel between `col` and question
    Add column to `df` and return `df`
    """
    tfid_vectorizer = TfidfVectorizer(token_pattern='(\S+)', min_df=1, max_df=0.9, ngram_range=(1,1))
    tfidf_X = tfid_vectorizer.fit_transform(df[col])
    tfidf_question = tfid_vectorizer.transform([question]).todense()
    df['tfidf_' + col] = compute_similarity(tfidf_question, tfidf_X)
    return df


def symbolic_ngrams(df, col, question, pca_dim=50, topk=5):
    """
    Fit BOW and PCA model on `col`, transform user question, and compute linear kernel between `col` and question
    Add column to `df` and return `df`
    """
    vect = CountVectorizer(analyzer='char', ngram_range=(1, 5), min_df=1)
    X = vect.fit_transform(df[col]).todense()
    pca = PCA(pca_dim)
    pca_X = pca.fit_transform(X)
    vect_question = pca.transform(vect.transform([question]).todense())
    df['ngram_' + col] = compute_similarity(vect_question, pca_X)
    return df

def tfidf_symbolic_ngrams_cosine_similarities(df, col, question, pca_dim=50, topk=5):
    """
    Fit tf-idf, BOW, and PCA model on `col`, transform user question, 
    Compute linear kernel between `col` and question
    Add column to `df` and return `df`
    """
    # tfidf 
    tfid_vectorizer = TfidfVectorizer(token_pattern='(\S+)', min_df=1, max_df=0.9, ngram_range=(1,1))
    tfidf_X = tfid_vectorizer.fit_transform(df[col]).todense()
    
    # symbolic n-grams (1 - 5)
    vect = CountVectorizer(analyzer='char', ngram_range=(1, 5), min_df=1, max_df=0.9)
    count_X = vect.fit_transform(df[col]).todense()
    X = np.concatenate((tfidf_X, count_X), axis=1)
    
    # PCA 
    pca = PCA(pca_dim)
    pca_X = pca.fit_transform(X)
    
    # question
    tfidf_question = tfid_vectorizer.transform([question]).todense()
    count_question = vect.transform([question]).todense()
    transformed_question = np.concatenate((tfidf_question, count_question), axis=1)
    transformed_question = pca.transform(transformed_question)
    
    # cosine similarity 
    df['tfidf_ngram_combo_' + col] = compute_similarity(transformed_question, pca_X)
    return df
