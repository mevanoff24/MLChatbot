# utils file 

import nltk
import pickle
import re
import numpy as np
import html
import annoy
from pathlib import Path
import os 
import pandas as pd
import csv
import nmslib
from datetime import datetime
from collections import defaultdict
import sqlite3

# DB_NAME = 'StackOverflow.db'

# connection = sqlite3.connect(DB_NAME)
# c = connection.cursor()


from config import * 
from utils import *


DATA_PATH = Path('../data/')
MODEL_PATH = Path('../models/')


class DialogueManager(object):
    def __init__(self, data_path, model_path):
        self.model_path = model_path
        self.data_path = data_path
        self.thread_embeddings_path = model_path/'thread_embeddings_by_tags'
        self.knn_path = model_path/'knn_embeddings_path'
        self.word_embeddings, self.dim = get_embeddings('starspace_embedding100_ngram2.tsv')
        self.parent_comment_map = unpickle(data_path/'parent_comment_map.pkl')
        
        self.tag_classifier = unpickle(model_path/'LR_tag_classifier_all.pkl')
        self.class_map = unpickle(data_path/'class_map.pkl')
        
        self.tfid_vectorizer = unpickle(model_path/'tf_idf.pkl')
        self.idf_scores = defaultdict(lambda:0, zip(self.tfid_vectorizer.get_feature_names(), 
                                                    self.tfid_vectorizer.idf_))
        
        self.parent_comment_map = unpickle(data_path/'parent_comment_map.pkl')
        
        
    def __get_embeddings_by_tag(self, tags):
        
        all_ids, all_vectors = [], []
        embeddings_files = [str(self.thread_embeddings_path/tag) for tag in tags]
        for file in embeddings_files:
            tag_path = file + '.pkl'
            ids, vectors = unpickle(tag_path)
            all_ids.append(ids)
            all_vectors.append(vectors)
        return all_ids, all_vectors
        
    
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
        
        tags = self.get_tags(question)
        print('get vects for tags {}'.format(tags))
        start = datetime.now()
        thread_ids, thread_vectors = self.__get_embeddings_by_tag(tags)

        print(datetime.now() - start)
        print('create index')
        start = datetime.now()

        question2vec = question_to_vec(question, self.word_embeddings, self.dim, vect=self.tfid_vectorizer, 
                                        idf_scores=self.idf_scores, *args, **kwargs)
        output = []
        distances = []

        for i in range(len(tags)):
            tag = tags[i]
            thread_vector = thread_vectors[i]
            thread_id = thread_ids[i]
            tag_path = tag + '.bin'
            index = self.__create_annoy_index(thread_vector, space=space, load=load, 
                                           filepath=str(self.knn_path/tag_path), save=save)
            idxs, distance = index.get_nns_by_vector(question2vec, n=topk, include_distances=return_dist)
            output.append([thread_id[i] for i in idxs])
            distances.append([distance])

        if return_dist: return output, distances
        else: return output

   
        
    
    def get_comments(self, post_ids, connection):
#         tags_repeat = [np.repeat(tag, len(post_ids[0])) for tag in tags]
        post_ids = np.array(post_ids).flatten()
        # need to weight by distance 
        # need to get for multiple tags, combine and weight by distance
        df_parent = self.get_df(post_ids.tolist(), connection=connection)

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
        results = np.array(results).flatten()
#         print(distances)
        distances = np.array(distances).flatten()
#         print(distances)
        distance_df = pd.DataFrame.from_dict(dict(zip(results, distances)), orient='index')
        distance_df.columns = ['distances']
        parent_len = len(df_parent)
        df_parent = df_parent.merge(distance_df, how='left', left_on='comment_id', right_index=True)
        assert len(df_parent) == parent_len 

        final_df = df_parent[['comment_id', 'comment', 'title', 'score', 'distances', 'tags']].merge(
                df_comments[['parent_id', 'comment', 'score']], how='left', left_on='comment_id', right_on='parent_id', 
                suffixes=('_parent', '_comment')).dropna(subset=['comment_comment']).sort_values('distances')
        return final_df.reset_index(drop=True)

    def get_df(self, ids, connection):
        neighbor_length = '?,' * len(ids)
        df = pd.read_sql("SELECT * FROM posts WHERE comment_id IN ({})".format(neighbor_length[:-1]), 
                                         connection, params=tuple(ids))
        return df
    
    