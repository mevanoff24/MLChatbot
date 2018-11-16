# MLChatbot
Web app that allows users to ask any programming, coding, AI, system question and returns similar questions and answers. The motivation behind this app is to allow users a 'one-shot response'. Instead of having to scroll through the multiple results and pages from a Google Search, this app returns everything in one click. This systems is built based on the combination of a 'Keyword Recognition-Based Chatbot' and a 'Natural Language Processing Chatbots'. It does have a 'pre-loaded' response system in place (Stack Overflow answers), but it also heavily utilizes a contextual understanding of a question towards its resolution. [[1]](https://www.taskus.com/blog/keyword-based-natural-language-processing-chatbots-mean/)

 

## Approach

When a users asks a question, this process is as follows
- First the question is cleaned, a [term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is applied to create input features, then we predict the programming language that the user question refers to with a Logistic Regression 'Tag Classifier'. We return the top 3 results (top 1 depending on how confident the model is). This allows us not to have to run through the entire database of data and reduces the look up 90%. 
- Next, we create an the average word vector for User's question. As the user's question sentence represents a sentence by averaging the weighted word embeddings of all words, with the weight of a word is given by tf-idf [[2]](http://www.aclweb.org/anthology/S17-2100), [[3]](http://aclweb.org/anthology/P/P16/P16-1089.pdf). This helps learn the neuance of the question. We have trained custom word embeddings with [facebook's StarSpace](https://github.com/facebookresearch/StarSpace) to get some more 'finer-detail' in the embeddings.  
- We then get 'topk' closest neighbors with Approximate Nearest Neighbor [[4]](https://arxiv.org/pdf/1806.09823.pdf). Based on pre-trained tf-idf average word vectors for each response in our database. 
- After we return the top neighbors from our locality sensitive hashing, we do one last 'weighing' to sort the top responsese. We weigh this 'similarity' score based on [levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance), Tf-Idf similarity and [symbolic n-grams](https://sites.google.com/site/textdigitisation/ngrams) with [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis). 
- We then can finally output the top responsed based on this weighted 'similarity' score from these features. 




## Process
- Finding similaritites based on [Stack Overflow questions and answers datasets](https://archive.org/details/stackexchange)
- Runs a 'Tag Classifier' to predict the programming language to subset the database 
- Similarity based on Tf-Idf Average Word Vectors with pre-computed word vectors.
- Weigh 'similarity' score based on levenshtein distance, Tf-Idf similarity and symbolic n-grams with PCA. 
- Word vectors trained with [https://github.com/facebookresearch/StarSpace](https://github.com/facebookresearch/StarSpace)
- Approximate Nearest Neighbors lookup for quick response (query time 0:00:00.000383)

## Data 

Most data is from th [Stack Overflow questions and answers datasets](https://archive.org/details/stackexchange)
- [stackoverflow.com-Comments.7z](https://archive.org/download/stackexchange/stackoverflow.com-Comments.7z) (3.7G)
- [stackoverflow.com-Posts](https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z) (12.4G)
- [stackoverflow.com-PostHistory.7z](https://archive.org/download/stackexchange/stackoverflow.com-PostHistory.7z) (21.7G)
- [stackoverflow.com-PostLinks.7z](https://archive.org/download/stackexchange/stackoverflow.com-PostLinks.7z) (71.4M)
- [stackoverflow.com-Tags.7z](https://archive.org/download/stackexchange/stackoverflow.com-Tags.7z) (747.1K)



## In Progress
- Web Hosting - AWS Lambda (coming soon!)
- Generic ChatBot - Sequence to Sequence with Attention - Tensorflow
- 'Rogue' agent. Generative Language Model agent to produce a legible english response. Language model trained on stack overflow answers. Using a pre-trained Language model from Wiki Articles 



### TODO:
- add docstrings
- improve READEME (metrics, more thorough this file)
- put notebooks in specific folders
- more comments 



## Demo

Here might be a quick overview of of what the app does


#### Question 

<img width="1269" alt="screen shot 2018-08-08 at 9 17 13 am" src="https://user-images.githubusercontent.com/8717434/43850126-f62510fe-9aeb-11e8-9ca7-d01937895d22.png">


#### Answer 

<img width="1275" alt="screen shot 2018-08-08 at 9 16 56 am" src="https://user-images.githubusercontent.com/8717434/43850199-21a886c0-9aec-11e8-9ce3-0a068c0327ce.png">







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

