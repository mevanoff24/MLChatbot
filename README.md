# MLChatbot
Web app that allows users to ask a coding question and returns similar questions and answers


## Process
- Finding similaritites based on [Stack Overflow questions and answers datasets](https://archive.org/details/stackexchange)
- Runs a 'Tag Classifier' to predict the programming language to subset the database 
- Similarity based on Tf-Idf Average Word Vectors with pre-computed word vectors.
- Weigh 'similarity' score based on levenshtein distance, Tf-Idf similarity and symbolic n-grams with PCA. 
- Word vectors trained with [https://github.com/facebookresearch/StarSpace](https://github.com/facebookresearch/StarSpace)
- Approximate Nearest Neighbors lookup for quick response (query time 0:00:00.000383)

## Data 
- pass 

## Dependencies 
- python 3.6 


## In Progress
- Web Hosting - AWS Lambda (coming soon!)
- Generic ChatBot - Sequence to Sequence with Attention - Tensorflow
- 'Rogue' agent. Generative Language Model agent to produce a legible english response. Language model trained on stack overflow answers. Using a pre-trained Language model from Wiki Articles 



### TODO:
- add docstrings
- improve READEME (metrics, more thorough this file)
- put notebooks in specific folders
- more comments 
- push rec system to github


## Demo

Overview of project 


#### Question 

<img width="1269" alt="screen shot 2018-08-08 at 9 17 13 am" src="https://user-images.githubusercontent.com/8717434/43850126-f62510fe-9aeb-11e8-9ca7-d01937895d22.png">


#### Answer 

<img width="1275" alt="screen shot 2018-08-08 at 9 16 56 am" src="https://user-images.githubusercontent.com/8717434/43850199-21a886c0-9aec-11e8-9ce3-0a068c0327ce.png">

