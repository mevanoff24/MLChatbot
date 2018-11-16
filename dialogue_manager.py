# python3

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
# import nmslib
from datetime import datetime
from collections import defaultdict
import sqlite3

# DB_NAME = 'StackOverflow.db'

# connection = sqlite3.connect(DB_NAME)
# c = connection.cursor()

# 
from config import * 
from utils import *


DATA_PATH = Path('../data/')
MODEL_PATH = Path('../models/')


class DialogueManager(object):
    """
    Dialogue Manager Class 
    
    Attributes:
        model_path (Path object): pathlib file path to model directory  
        data_path (Path object): pathlib file path to data directory
        thread_embeddings_path (str): Path to thread embeddings 
        knn_path (Path object): Path to Nearest Neighbors
        word_embeddings (array): Word embeddings
        dim (int): Dimensions of word embeddings
        parent_comment_map (dict): Dictionary mapping parent question to list of answers
        tag_classifier (object): Sklearn model
        class_map (dict): Sklearn label encoding mapping of class index to class name
        tfid_vectorizer (obect): Pre-trained tf-idf vectorizer
        idf_scores (defaultdict): Mapping from tf-idf vectorizer to inverse document frequency vector
        min_max (object): MinMaxScaler object for weigh features 
    """
    
    def __init__(self, data_path, model_path):
        """
        Args: 
            data_path (Path object): pathlib file path to data directory
            model_path (Path object): pathlib file path to model directory
        """
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
        
    def __get_embeddings_by_tag(self, tags):
        """
        Get embedding IDs and vectors based on the predicted tag from the tag classifier 
        
        Args: 
            tags (list): Predicted tags from tag classifier
        
        Returns:
            all_ids (list): List of ids from thread embeddings
            all_vectors (list): List of word vectors from thread embeddings
        """
        all_ids, all_vectors = [], []
        embeddings_files = [str(self.thread_embeddings_path/tag) for tag in tags]
        # loop through embeddings files and append to lists
        for file in embeddings_files:
            tag_path = file + '.pkl'
            ids, vectors = unpickle(tag_path)
            all_ids.append(ids)
            all_vectors.append(vectors)
        return all_ids, all_vectors
        
    
    def __create_nmslib_index(self, a, space, load=True, filepath=None, save=False, use_query_params=False):
        """
        Create similarity search (Currently Not being used (create_annoy_index() used below)) 
        """
        M = 25
        efC = 100 
        num_threads = 4
        index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 
                             'post': 0, 'skip_optimized_index': 1}
        query_time_params = {'efSearch': efC}

        index = nmslib.init(space=space)
        if load:
            # only need to init if indexed is saved
            index.loadIndex(filepath)
            if use_query_params:
                index.setQueryTimeParams(query_time_params)
            return index
        else:
            index.addDataPointBatch(a)
            index.createIndex()
            if use_query_params:
                index.setQueryTimeParams(query_time_params)
            if save: index.saveIndex(filepath)
            return index
        
    def __create_annoy_index(self, data, space='angular', n_trees=30, load=True, filepath=None, save=False):
        """
        Create or Load Approximate Nearest Neighbors index 
        
        Args: 
            data (array): Thread word vectors 
            space (str): Distance (metric) function can be "angular", "euclidean", "manhattan", "hamming", or "dot"
            n_trees (int): Number of trees in a forest. More trees gives higher precision when querying.
            load (boolean): Load model (True) -- Create model (False)
            filepath (str): Path to Nearest Neighbors
            save (boolean): Save model (True) -- Only used if load=False
        
        Returns:
            index (object): Annoy object
        """     
        index = annoy.AnnoyIndex(self.dim, metric=space)
        if load:
            # only need to init if index is saved
            index.load(filepath)
        else:
            for i, vect in enumerate(data):
                # add data
                index.add_item(i, vect)
            # build moel 
            index.build(n_trees) 
            # save indexes
            if save: index.save(filepath)
        return index

    
    def get_similar(self, question, question_to_vec=average_tfidf_vectors, topk=5, space='angular', 
                    load=True, save=False, return_dist=True, *args, **kwargs):
        """
        Get most similar responses to the users question based on Td-idf average word vectors
        
        Args: 
            question (str): Question asked by User 
            question_to_vec (func): Function to compute average word vector scores. 
                                    Can be `average_tfidf_vectors` or `avg_word_vectors`
            topk (int): Number of closest items to return from NN model 
            space (str): Distance (metric) function can be "angular", "euclidean", "manhattan", "hamming", or "dot"
            load (boolean): Load model (True) -- Create model (False)
            save (boolean): Save model (True) -- Only used if load=False
            return_dist (boolean): If include_distances to True, it will return a 2 element tuple with two lists 
                                   in it: the second one containing all corresponding distances.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            output (list): Indicies of Nearest Neighbors to User question 
            distances (list): Distances of indicies of Nearest Neighbors to User question 
                              -- Only returned if return_dist=True
            tags (list): Predicted tags from Tag Classifier 
            scores (list): Confidence scores from Tag Classifier 
        """ 
        # get top predicted tags and scores from tag classifier 
        tags, scores = self.get_tags(question)
        print(tags)
        print('get vects for tags {}'.format(tags))
        start = datetime.now()
        # get word embedding vectors based on predicted tags 
        thread_ids, thread_vectors = self.__get_embeddings_by_tag(tags)

        print(datetime.now() - start)
        print('create index')
        start = datetime.now()
        # create average word vector for user Question 
        question2vec = question_to_vec(question, self.word_embeddings, self.dim, vect=self.tfid_vectorizer, 
                                        idf_scores=self.idf_scores, *args, **kwargs)
        output = []
        distances = []

        for i in range(len(tags)):
            # corresponding tag
            tag = tags[i]
            # corresponding word vector 
            thread_vector = thread_vectors[i]
            # corresponding id 
            thread_id = thread_ids[i]
            tag_path = tag + '.bin'
            # create or load Approximate Nearest Neighbor model 
            index = self.__create_annoy_index(thread_vector, space=space, load=load, 
                                           filepath=str(self.knn_path/tag_path), save=save)
            idxs, distance = index.get_nns_by_vector(question2vec, n=topk, include_distances=return_dist)
            # append to lists 
            output.append([thread_id[i] for i in idxs])
            distances.append([distance])

        if return_dist: return output, distances, tags, scores
        else: return output, tags, scores
        
        
    def get_comments(self, post_ids, connection):
        """
        Get Parent and Response from DB based on Approximate Nearest Neighbor model
        
        Args: 
            post_ids (list): topk Thread Ids from NN model 
            connection: DB connection instance 
        
        Returns:
            df_parent (pandas df): Parent comments dataframe
            df_comments (pandas df): Response comments dataframe
        """     
        post_ids = np.array(post_ids).flatten()
        # get most similar parent comments from DB
        df_parent = self.get_df(post_ids.tolist(), connection=connection)
        # only keep comments that we have the actual parent for 
        knns = [j for i in post_ids if i in self.parent_comment_map for j in self.parent_comment_map[i]]
        df_comments = self.get_df(knns, connection=connection)
        return df_parent, df_comments 

    def get_tags(self, question, k=3):
        """
        Get top predicted tags and confidence scores from Tag Classifier model
        
        Args: 
            question (list): topk Thread Ids from NN model 
            k (int): Number of top `k` predictions and confidence scores to return from model
        
        Returns:
            preds (list): Programming language predictions 
            scores (list): Confidence scores for predicted programming language
        """    
        # clean the users question 
        cleaned_question = clean_text(question)
        # get tf-idf features for question 
        features = self.tfid_vectorizer.transform([cleaned_question])
        # get top `k` predictions and confidence scores from the Tag Classifier model
        preds, scores = get_top_preds(features, self.tag_classifier, self.class_map, k)
        return preds, scores
    
    
    def merge_parent_comment(self, df_parent, df_comments, results, distances, highest=False):
        """
        Merge the parent comments and responses into one pandas dataframe 
        
        Args: 
            df_parent (pandas df): Parent comments dataframe
            df_comments (pandas df): Response comments dataframe
            results (list): Indicies of Nearest Neighbors to user question
            distances (list): Distances of Nearest Neighbors
            highest (boolean): only keep highest score if True
            
        Returns:
            final_df (pandas df): Merged dataframes from parent and response comments from DB
        """ 
        results = np.array(results).flatten()
        distances = np.array(distances).flatten()
        # create new distance dataframe from Indicies of Nearest Neighbors
        distance_df = pd.DataFrame.from_dict(dict(zip(results, distances)), orient='index')
        distance_df.columns = ['distances']
        parent_len = len(df_parent)
        # merge parent dataframe to distance dataframe 
        df_parent = df_parent.merge(distance_df, how='left', left_on='comment_id', right_index=True)
        assert len(df_parent) == parent_len 
        # merge parent dataframe to answer dataframe and only keep certain columns (--clean this code up--)
        final_df = df_parent[['comment_id', 'comment', 'title', 'score', 'distances', 'tags']].merge(
                df_comments[['parent_id', 'comment', 'score']], how='left', left_on='comment_id', right_on='parent_id', 
                suffixes=('_parent', '_comment')).dropna(subset=['comment_comment']).sort_values('distances')
        # only keep highest score if True
        if highest:
            final_df = final_df.groupby('parent_id', group_keys=False).apply(lambda x: x.loc[x.score_comment.idxmax()])

        return final_df.reset_index(drop=True)
  
    def get_df(self, ids, connection):
        """
        Get data from Database and return pandas dataframe
        
        Args: 
            ids (list): Comment Ids to pull from DB
            connection: DB connection instance        
        Returns:
            df (pandas df): Pandas dataframe based on comment id 
        """ 
        # create length of neighbor desired 
        neighbor_length = '?,' * len(ids)
        # select comment_ids from ids 
        df = pd.read_sql("SELECT * FROM posts WHERE comment_id IN ({})".format(neighbor_length[:-1]), 
                                         connection, params=tuple(ids))
        return df
    
    def joined_comment_comment(self, df, concat=True):
        """
        Join child comments to parent comments and sort by distances 
        
        Args: 
            df (pandas df): Comment Ids to pull from DB
            concat (boolean): Concatenate comment and score if True     
        
        Returns:
            tmp (pandas df): Pandas dataframe based on joined 
        """ 
        # concatenate comment and score into one field if True 
        if concat:
            df['comment_comment'] = df['comment_comment'] + '::' + df['score_comment'].map(str)

        cols = ['comment_id', 'comment_parent', 'title', 'score_parent', 'distances', 'tags', 'parent_id']
        # sort by 'parent_id' and 'score_comment' and groupby `cols` append comments
        tmp = pd.DataFrame(df.sort_values(['parent_id', 'score_comment'], ascending=False).groupby(
                                                cols).apply(lambda x: joined(x.comment_comment))).reset_index()
        tmp.rename(columns=({0: 'combined_comments'}), inplace=True)
        # sort by distances 
        tmp.sort_values('distances', inplace=True)
        return tmp
    
    
    def weigh_features(self, df, question, tags, scores):
        """
        Join child comments to parent comments and sort by distances 
        
        Args: 
            df (pandas df): Comment Ids to pull from DB
            question (string): Cleaned question inputed from user
            tags (list): Predicted programming tags
            scores (list): Predicted programming confidence scores
        
        Returns:
            tmp (pandas df): Pandas dataframe based on joined 
        """ 
        # optimal params found by gridSearch using discounted cumulative gain as metric 
        param_dict = {'distance': 0.92, 'tfidf_title': 0.74, 'ngram_title': 0.21, 'tfidf_ngram_combo_title': 0.33}
        # only use 'title' and 'parent' fields for text features (found from validation set)
        # since this `df` is not too big we can compute all these 'features'
        for col in ['title', 'comment_parent']:
            # clean text
            col = clean_text(col)
            # tf-idf fit on `col`, transform user question and compute linear kernel between `col` and question
            df = tfidf_cosine_similarities(df, col, question, topk=100)
            # fit BOW and PCA model on `col`, transform user question, and compute linear kernel between `col` and question
            df = symbolic_ngrams(df, col, question, topk=100)
            # fit tf-idf, BOW, and PCA model on `col`, transform user question, 
            # and compute linear kernel between `col` and question
            df = tfidf_symbolic_ngrams_cosine_similarities(df, col, question, topk=100)
        # scaler to scale created features 
        self.min_max = MinMaxScaler()
        all_desired_cols = list(df.select_dtypes(include='float').columns)
        for col in all_desired_cols:
            # use fit_transform, since don't reall have training data
            df[col] = self.min_max.fit_transform(df[[col]])
        
        df.tags = df.tags.map(lambda x: x.replace('-', '_'))
        tag_scores = dict(zip(tags, scores))

        def __tag_weight(x):
            """Map tag to tag scores"""
            X = [tag_scores[tag] for tag in tags if tag in x if tag_scores[tag]]
            return X[0] if len(X) > 0 else 0
        
        # tag scores to weights 
        df['tag_score'] = df['tags'].map(lambda x: __tag_weight(x))
        # Linear combination of all the created features multiplied by 'weights' of optimal combinations
        df['final_weight'] = ((1-df['distances']) * param_dict['distance']) + \
                             (df['tfidf_title'] * param_dict['tfidf_title']) + \
                             (df['ngram_title'] * param_dict['ngram_title']) + \
                             (df['tfidf_ngram_combo_title'] * param_dict['tfidf_ngram_combo_title']) + \
                             (df['tfidf_comment_parent'] * param_dict['tfidf_title']) + \
                             (df['ngram_comment_parent'] * param_dict['ngram_title']) + \
                             (df['tfidf_ngram_combo_comment_parent'] * param_dict['tfidf_ngram_combo_title'] + \
                             df['tag_score'])
        # sort by best weight 
        df.sort_values('final_weight', ascending=False, inplace=True)
        return df
    
    