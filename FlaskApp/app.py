from flask import Flask, render_template, request, jsonify
from wtforms import Form, TextAreaField, validators
import sys
import os
from pathlib import Path


import pandas as pd 
import numpy as np 




sys.path.append(os.path.abspath("../"))

from dialogue_manager_python import DialogueManager
from config import * 
from utils import *

DB_NAME = '../StackOverflow_python.db'


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
    
    connection = sqlite3.connect(DB_NAME)
    print(connection)
    c = connection.cursor()
    print(c)
    form = SubmissionForm(request.form)
    if request.method == 'POST' and form.validate():
        question = request.form['question_form']
        print(question)
        # do all 
        clean_question = clean_text(question)
        results, distances = dm.get_similar(clean_question, topk=10, load=True, save=False, return_dist=True)
        df_parent, df_comments = dm.get_comments(results, connection=connection)
        df = dm.merge_parent_comment(df_parent, df_comments, results, distances)
        # get best results top 5? 
        

#         df = pd.read_csv(OUTPUT_PATH/'final_df.csv')
        df = df[['title', 'comment_parent', 'comment_comment', 'distances']]
#         df = df.loc[[4, 5, 6]]
        data = []
        for title, parent_comment, answer, distance in zip(df['title'], df['comment_parent'], df['comment_comment'], df['distances']):
            data.append((title, parent_comment, answer, np.round(distance, 2)))
        connection.close()
        return render_template('predict.html', 
                    tables=[df.to_html()], 
                    titles=df.columns.values, 
                    original_question=question,
                    data=data,
                    )





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)


