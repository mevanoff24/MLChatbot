# MLChatbot
Web app that allows users to ask any programming, coding, AI, system question and returns similar questions and answers. The motivation behind this app is to allow users a 'one-shot response'. Instead of having to scroll through the multiple results and pages from a Google Search, this app returns everything in one click. This systems is built based on the combination of a 'Keyword Recognition-Based Chatbot' and a 'Natural Language Processing Chatbots'. It does have a 'pre-loaded' response system in place, but it also heavily utilizes a contextual understanding of a question towards its resolution. [[1]](https://www.taskus.com/blog/keyword-based-natural-language-processing-chatbots-mean/)


Fast prediction time is definitely needed. Large dataset. 

## Approach

When a users asks a question, this process is as follows
- first the question is cleaned, a [term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is applied to create features, then we predict the programming language that the user question refers to with a 'Tag Classifier'. We run the top 3 results ( top 1 depending on how confident the model is). This allows us not to have to run through the entire database of data and reduces the look up 90%. 
- Next, we create an the average word vector for User. As the user's question sentence represents a sentence by averaging the weighted word embeddings of all words, with the weight of a word is given by tf-idf [[2]](http://www.aclweb.org/anthology/S17-2100), [[3]](http://aclweb.org/anthology/P/P16/P16-1089.pdf). This helps learn the neuance of the question. We have trained custom word embeddings with [facebook's StarSpace](https://github.com/facebookresearch/StarSpace) to get some more 'finer-detail'. 
- We then get topk closest neighbors with Approximate Nearest Neighbor [[4]](https://arxiv.org/pdf/1806.09823.pdf). Based on pre-trained tf-idf average word vectors for each answer in our database. 



All the data is from Stack Overflow questions and answers datasets is cleaned and parsed into a large database.  

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

