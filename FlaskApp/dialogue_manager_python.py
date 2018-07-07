import os 
import numpy as np
import pandas as pd
from pathlib import Path
import csv
import pickle
# import nmslib
import annoy
from datetime import datetime
from collections import defaultdict
# from IPython.display import FileLink
import sqlite3
import nltk
import re
import html


from config import * 
from utils import *

# DB_NAME = 'StackOverflow_python.db'


# connection = sqlite3.connect(DB_NAME)
# c = connection.cursor()


DATA_PATH = Path('../data/')
MODEL_PATH = Path('../models/')
FLASK_PATH = Path('ui')
FLASK_PATH.mkdir(exist_ok=True)



# # utils file 

# # from nltk.corpus import stopwords
# # stopwords_set = set(stopwords.words('english'))


# def avg_word_vectors(question, embeddings, dim):
#     words_embedding = [embeddings[word] for word in question.lower().split() if word in embeddings]
#     if not words_embedding:
#         return np.zeros(dim)
#     words_embedding = np.array(words_embedding).astype(np.float32)
#     return words_embedding.mean(axis=0)

# def average_tfidf_vectors(question, embeddings, dim, vect, idf_scores):
#     # get idf weights
#     split_question = [word for word in question.lower().split() if word in embeddings]
#     if not split_question:
#         return np.zeros(dim).astype(np.float32)
#     words_embedding = np.zeros((dim, len(split_question))).astype(np.float32)
#     for i, token in enumerate(split_question):
#         if token in embeddings:
#             embed_score = embeddings[token]
#         else: embed_score = 0
#         idf_score = idf_scores[token]
#         # word vectors multiply by their TF-IDF scores
#         words_embedding[:, i] = embed_score * idf_score    
#     return words_embedding.mean(axis=1)


# def get_embeddings(filename):
#     embeddings = {}
#     with open(MODEL_PATH/filename, newline='') as f:
#         reader = csv.reader(f, delimiter='\t')
#         embed_list = list(reader)
#     for line in embed_list:
#         embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
        
#     dim = len(embeddings['code'])
#     return embeddings, dim


# def unpickle(filename):
#     with open(filename, 'rb') as f:
#         return pickle.load(f)



# re1 = re.compile(r'  +')
# # def clean_title(text, remove_html=False, other=False):
# def clean_text(text, remove_html=False, other=False):
    
#     replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
#     bad_symbols_re = re.compile('[^0-9a-z #+_]')
    
#     if remove_html:
#         x = re.sub(r'<code>[^>]*</code>', '', text)
#         x = re.sub(r'<[^>]*>', '', x)
#         x = re.sub(r'[^A-Za-z0-9]', ' ', x)
#     text = text.lower()
#     text = text.replace('π', 'pi').replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
#             'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
#             '<br />', "\n").replace('\\"', '').replace('<unk>','u_n').replace(' @.@ ','.').replace(
#             ' @-@ ','-').replace('\\', ' \\ ').replace('"',"'").replace('\n', ' ').replace('\r', ' ')
#     text = replace_by_space_re.sub(' ', text)
#     text = bad_symbols_re.sub('', text)
#     text = ' '.join([x for x in text.split() if x and x not in stopwords_set])
#     return re1.sub(' ', html.unescape(text).strip())


# # def get_top_preds(X, clf, k=3):
# #     preds = clf.predict_proba(X)
# #     return np.argsort(preds)[:,::-1][0][:3]

# def get_top_preds(X, clf, class_map, k=3, cutoff=0.8):
#         preds = clf.predict_proba(X)
#         sorted_preds = np.argsort(preds)[:,::-1][0][:3]
#         top_scores = [preds[0][i] for i in sorted_preds]
#         if top_scores[0] > cutoff:
#             # if top_scores above threshold only need to return top 
#             return [class_map[sorted_preds[0]]]
#         else:
#             return [class_map[p] for p in sorted_preds]

        
# def merge_parent_comment(df_parent, df_comments, results, distances):
#     distance_df = pd.DataFrame.from_dict(dict(zip(results, distances)), orient='index')
#     distance_df.columns = ['distances']
#     parent_len = len(df_parent)
#     df_parent = df_parent.merge(distance_df, how='left', left_on='comment_id', right_index=True)
#     assert len(df_parent) == parent_len 
    
#     final_df = df_parent[['comment_id', 'comment', 'title', 'score', 'distances']].merge(
#             df_comments[['parent_id', 'comment', 'score']], how='left', left_on='comment_id', right_on='parent_id', 
#             suffixes=('_parent', '_comment')).dropna(subset=['comment_comment']).sort_values('distances')
#     return final_df.reset_index(drop=True)


# from sklearn.metrics.pairwise import linear_kernel

# def tfidf_cosine_similarities(X, question, topk=5):
#     tfid_vectorizer = unpickle(MODEL_PATH/'tf_idf_python_title_stopwords.pkl')

#     tfidf_X = tfid_vectorizer.transform(X)
#     tfidf_question = tfid_vectorizer.transform([question])

#     cosine_similarities = linear_kernel(tfidf_question, tfidf_X).flatten()
#     related_docs_indices = cosine_similarities.argsort()[:-topk:-1]
#     print(related_docs_indices)
#     return final_df.loc[related_docs_indices]


# -------------------------------------------------------------------------------------------------------------------



class DialogueManager(object):
    def __init__(self, data_path, model_path):
        self.model_path = model_path
        self.data_path = data_path
        self.thread_embeddings_path = model_path/'thread_embeddings_by_tags'
        self.knn_path = model_path/'knn_embeddings_path'
        self.word_embeddings, self.dim = get_embeddings('starspace_embedding100_ngram2.tsv')
        self.parent_comment_map = unpickle(data_path/'parent_comment_map.pkl')
        
#         self.tag_classifier = unpickle(model_path/'LR_tag_classifier_all.pkl')
#         self.class_map = unpickle(data_path/'class_map.pkl')
        
        self.tfid_vectorizer = unpickle(model_path/'tf_idf_python_title_stopwords.pkl')
        self.idf_scores = defaultdict(lambda:0, zip(self.tfid_vectorizer.get_feature_names(), 
                                                    self.tfid_vectorizer.idf_))
        
        self.parent_comment_map = unpickle(data_path/'parent_comment_map.pkl')
        
        
    def __get_embeddings_by_tag(self, tag):
        
#         embeddings_files = [self.thread_embeddings_path/tag for tag in tags]
#         for file in embeddings_files:
#             ids, vectors = unpickle(embeddings_files)
        
        # tfidf-avg
        tag_path = 'python_only.pkl'
        embeddings_file = self.thread_embeddings_path/tag_path
        ids, vectors = unpickle(embeddings_file)
        return ids, vectors
    
    def __create_nmslib_index(self, a, space, load=True, filepath=None, save=False):
        M = 25
        efC = 100
        
        num_threads = 4
        index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 
                             'post': 0, 'skip_optimized_index':1}
        query_time_params = {'efSearch': efC}

        index = nmslib.init(space=space)
        if load:
            # only need to init if indexed is saved
            index.loadIndex(filepath)
#             index.setQueryTimeParams(query_time_params)
            return index
        else:
            index.addDataPointBatch(a)
            index.createIndex()
#             index.setQueryTimeParams(query_time_params)
            if save: index.saveIndex(filepath)
            return index
        
    def __create_annoy_index(self, data, space='angular', n_trees=30, load=True, filepath=None, save=False):
                
        index = annoy.AnnoyIndex(self.dim, metric=space)
        if load:
            # only need to init if indexed is saved
            index.load(filepath)
        else:
            for i, vect in enumerate(data):
                index.add_item(i, vect)
            index.build(n_trees)            
            if save: index.save(filepath)
        return index

    
    def get_similar(self, question, question_to_vec=average_tfidf_vectors, topk=5, space='angular', 
                    load=True, save=False, return_dist=True, *args, **kwargs):
        
#         tags = self.get_tags(question)
        # need to update for multiple tags
#         tag = tags[0]
        print('get vects')
        start = datetime.now()
#         tag
        thread_ids, thread_vectors = self.__get_embeddings_by_tag(tag=None)
#         candidate2vecs = np.load(MODEL_PATH/'candidate2vecs_python_title.npy')
        print(datetime.now() - start)
        print('create index')
        start = datetime.now()
#         tag_path = tag + '.bin'
#         tag_path = 'python_only.bin'
        tag_path = 'python_only.annoy'
#         index = self.__create_nmslib_index(thread_vectors, space=space, load=load, 
#                                            filepath=str(self.knn_path/tag_path), save=save)
        index = self.__create_annoy_index(thread_vectors, space=space, load=load, 
                                           filepath=str(self.knn_path/tag_path), save=save)
        print(datetime.now() - start)
        print('question creation')

        question2vec = question_to_vec(question, self.word_embeddings, self.dim, vect=self.tfid_vectorizer, 
                                        idf_scores=self.idf_scores, *args, **kwargs)
        print('query')
        start = datetime.now()
#         idxs, distances = index.knnQuery([question2vec], k=topk)
        idxs, distances = index.get_nns_by_vector(question2vec, n=topk, include_distances=return_dist)
        print(datetime.now() - start)
        output = [thread_ids[i] for i in idxs]
        if return_dist: output = output, distances
        return output
    
    
    def get_comments(self, post_ids, connection):
        # need to weight by distance 
        # need to get for multiple tags, combine and weight by distance
        df_parent = self.get_df(np.array(post_ids).flatten().tolist(), connection=connection)

        knns = [j for i in post_ids if i in self.parent_comment_map for j in self.parent_comment_map[i]]
        df_comments = self.get_df(knns, connection=connection)

        return df_parent, df_comments 
    
    def get_tags(self, question, k=3): 
        cleaned_question = clean_text(question)
        features = self.tfid_vectorizer.transform([cleaned_question])
        preds = get_top_preds(features, self.tag_classifier, self.class_map, k)
        return preds
    
    def clean_output(self, df):
        # get rid of bad (negative / 0 scores)
        # only show good comments (based on distance?)
        # clean html
        # print in cool format
        pass
    
    def merge_parent_comment(self, df_parent, df_comments, results, distances):
        distance_df = pd.DataFrame.from_dict(dict(zip(results, distances)), orient='index')
        distance_df.columns = ['distances']
        parent_len = len(df_parent)
        df_parent = df_parent.merge(distance_df, how='left', left_on='comment_id', right_index=True)
        assert len(df_parent) == parent_len 

        final_df = df_parent[['comment_id', 'comment', 'title', 'score', 'distances']].merge(
                df_comments[['parent_id', 'comment', 'score']], how='left', left_on='comment_id', right_on='parent_id', 
                suffixes=('_parent', '_comment')).dropna(subset=['comment_comment']).sort_values('distances')
        return final_df.reset_index(drop=True)

    def get_df(self, ids, connection):
        neighbor_length = '?,' * len(ids)
        df = pd.read_sql("SELECT * FROM posts WHERE comment_id IN ({})".format(neighbor_length[:-1]), 
                                         connection, params=tuple(ids))
        return df
    
    