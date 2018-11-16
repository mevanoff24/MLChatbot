from flask import Flask, render_template, request, jsonify
from wtforms import Form, TextAreaField, validators
import sys
import os
from pathlib import Path
import pandas as pd 
import numpy as np 

sys.path.append(os.path.abspath("../"))

# python only
# from dialogue_manager_python import DialogueManager
# all tags 
from dialogue_manager import DialogueManager

from config import * 
from utils import *

# DB_NAME = '../StackOverflow_python.db'
# DB_NAME = '../StackOverflow.db'
DB_NAME = '../StackOverflow_newline_score.db'


# load model class 
dm = DialogueManager(DATA_PATH, MODEL_PATH)

app = Flask(__name__)


class SubmissionForm(Form):
    question_form = TextAreaField('', 
                    [validators.DataRequired(),
                    validators.length(min=1)])


@app.route('/')
def index():
    form = SubmissionForm(request.form)
    return render_template('main.html', form=form)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # connect to DB
    connection = sqlite3.connect(DB_NAME)
    c = connection.cursor()
    form = SubmissionForm(request.form)
    if request.method == 'POST' and form.validate():
        question = request.form['question_form']
        print(question)
        # clean user question 
        clean_question = clean_text(question)
        # get most similar neighbors, distances, tags, and confidence scores 
        results, distances, tags, scores = dm.get_similar(clean_question, topk=10, load=True, save=False, return_dist=True)
        # get parent and comments from DB
        df_parent, df_comments = dm.get_comments(results, connection=connection)
        # join df_parent and df_comments
        df = dm.merge_parent_comment(df_parent, df_comments, results, distances)
        df = dm.joined_comment_comment(df)
        try:
            # weigh and sort top responsese based on levenshtein distance, Tf-Idf similarity and symbolic n-grams with PCA.
            df = dm.weigh_features(df, clean_question, tags, scores)
        except Exception as e:
            df = df
        df = df[['title', 'comment_parent', 'distances', 'tags', 'combined_comments']]
        data = []
        # to pass to view 
        for title, parent_comment, answer, distance in zip(df['title'], df['comment_parent'], df['combined_comments'], df['distances']):
            data.append((title, parent_comment, answer, np.round(distance, 2)))
        
        connection.close()
        return render_template('predict.html', 
                    tables=[df.to_html()], 
                    titles=df.columns.values, 
                    original_question=question,
                    data=data,
                    )





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)


