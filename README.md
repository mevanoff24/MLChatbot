# MLChatbot
Web app that allows users to ask any programming, coding, AI, framework, system related Question and returns the most 'similar' questions and answers. The motivation behind this app is to allow users a 'one-shot' response. Instead of having to scroll through the multiple results and pages from a Google Search, this app returns everything in one click. This systems is built based on the combination of a 'Keyword Recognition-Based Chatbot' and a 'Natural Language Processing Chatbots'. It does have a 'pre-loaded' response system in place (data is parsed from [Stack Overflow]((https://archive.org/details/stackexchange))), but it also heavily utilizes a contextual understanding of a question towards its resolution. [[1]](https://www.taskus.com/blog/keyword-based-natural-language-processing-chatbots-mean/)

 
 
## Demo

Here might be a quick overview of of what the app does


#### Question 

<img width="1269" alt="screen shot 2018-08-08 at 9 17 13 am" src="https://user-images.githubusercontent.com/8717434/43850126-f62510fe-9aeb-11e8-9ca7-d01937895d22.png">


#### Answer 

<img width="1275" alt="screen shot 2018-08-08 at 9 16 56 am" src="https://user-images.githubusercontent.com/8717434/43850199-21a886c0-9aec-11e8-9ce3-0a068c0327ce.png">



## Approach

When a users asks a question, this process is as follows:
- The user's question is cleaned, a [term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is applied to create input features, then we predict the `tag` (programming language or framework) that the user's question refers to with a 'Tag Classifier'. We return the top 3 results (or top 1 depending on how confident the model is). This allows us not to have to run through the entire database of data and reduces the DB look up 90%. 
- Next, the user's question is represented by averaging the weighted word embeddings of all words with the weight of a word given by tf-idf [[2]](http://www.aclweb.org/anthology/S17-2100), [[3]](http://aclweb.org/anthology/P/P16/P16-1089.pdf). This helps learn the neuance of the question and puts more emphasis on rare words. Since programming, ML, etc. are non-typical words that wouldn't commonly occur in pre-trained word vectors such as [Word2vec](https://en.wikipedia.org/wiki/Word2vec) or [GloVe](https://nlp.stanford.edu/projects/glove/), we train custom word embeddings with [facebook's StarSpace](https://github.com/facebookresearch/StarSpace) to get some more 'finer-detail' in the embeddings. 
- We then get the 'topk' closest neighbors with an Approximate Nearest Neighbor algorithm [[4]](https://arxiv.org/pdf/1806.09823.pdf) comparing the transformed user's question to the pre-trained tf-idf weighted average word vectors from top `tags`. 
- After we return the top neighbors from our locality sensitive hashing algorihms, we do one last 'weighting' to sort the top responsese. We weigh this 'similarity' score based on [levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance), Tf-Idf similarity and [symbolic n-grams](https://sites.google.com/site/textdigitisation/ngrams) with [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis). 
- We then can finally output the top responsed based on this weighted 'similarity' score from these features.


## Results 
Average Query time 0.76484225 seconds 

### Metrics 

The first simple metric will be a number of correct hits for [some *K*](https://en.wikipedia.org/wiki/Iverson_bracket): **Hits@K**
```
dup_ranks = np.array(dup_ranks)
len(dup_ranks[dup_ranks <= k]) / len(dup_ranks)
```

The second one is a simplified [DCG metric](https://en.wikipedia.org/wiki/Discounted_cumulative_gain): **DCG@K**
```
dup_ranks = np.array(dup_ranks)
N = len(dup_ranks)
dup_ranks = dup_ranks[dup_ranks <= k]
np.sum((np.ones_like(dup_ranks)) / (np.log2(1.0 + dup_ranks))) / float(N)
```
Where `dup_ranks` is a list of duplicates' ranks, one rank per question. And `k` is the number of top-ranked elements. 



Based on the results of 

**input:**
```
test_embeddings(embeddings, 300, average_tfidf_vectors, vect=vect)
```
**output:**
```
DCG@   1: 0.453 | Hits@   1: 0.453
DCG@   5: 0.548 | Hits@   5: 0.631
DCG@  10: 0.567 | Hits@  10: 0.690
DCG@ 100: 0.602 | Hits@ 100: 0.861
DCG@ 500: 0.616 | Hits@ 500: 0.964
DCG@1000: 0.619 | Hits@1000: 1.000
```

**input:**
```
test_embeddings(embeddings, 100, average_tfidf_vectors, vect=vect)
```
**output:**
```
DCG@   1: 0.441 | Hits@   1: 0.441
DCG@   5: 0.533 | Hits@   5: 0.614
DCG@  10: 0.554 | Hits@  10: 0.678
DCG@ 100: 0.591 | Hits@ 100: 0.854
DCG@ 500: 0.605 | Hits@ 500: 0.965
DCG@1000: 0.609 | Hits@1000: 1.000
```
We actually went with the simpler model of 100 dimensional word vectors instead of the 300 dimensional word vectors for computational reasoning despite the marginal gain in DCG and Hits @ K. This greatly helped the average query time and memory requirements. 


----


## Data 

Most data is from th [Stack Overflow questions and answers datasets](https://archive.org/details/stackexchange)
- [stackoverflow.com-Comments.7z](https://archive.org/download/stackexchange/stackoverflow.com-Comments.7z) (3.7G)
- [stackoverflow.com-Posts](https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z) (12.4G)
- [stackoverflow.com-PostHistory.7z](https://archive.org/download/stackexchange/stackoverflow.com-PostHistory.7z) (21.7G)
- [stackoverflow.com-PostLinks.7z](https://archive.org/download/stackexchange/stackoverflow.com-PostLinks.7z) (71.4M)
- [stackoverflow.com-Tags.7z](https://archive.org/download/stackexchange/stackoverflow.com-Tags.7z) (747.1K)



## In Progress
- Web Hosting - AWS Lambda
- Generic ChatBot - Sequence to Sequence with Attention - Tensorflow. Use another classifier to predict if the user's response is a programming question or general chat (and use ChatBot for general chat)
- 'Rogue' agent. Generative Language Model agent to produce a legible english response. Language model trained on stack overflow answers. Using a pre-trained Language model from Wiki Articles 


## Requirements  
- python 3.6 
- flask
- pandas
- numpy
- scikit-learn 
- annoy
- pickle
- sqlite3

All can be easily installed with `pip` or `conda`

