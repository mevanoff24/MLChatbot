# MLChatbot
Web app that allows users to ask a coding question and returns similar questions and answers


## Process
- Finding similaritites based on Stack Overflow questions and answers datasets
- Similarity based on Tf-Idf Average Word Vectors with pre-computed word vectors  
- Word vectors trained with [https://github.com/facebookresearch/StarSpace](https://github.com/facebookresearch/StarSpace)\
- Approximate Nearest Neighbors lookup for quick response (query time 0:00:00.000383)


## In Progress
- Update to answer all programming languages questions with a 'Tag Classifier'
- Generic ChatBot - Sequence to Sequence with Attention - Tensorflow
- 'Rogue' agent. Generative Language Model agent to produce a legible english response. Language model trained on stack overflow answers. Using a pre-trained Language model from Wiki Articles 
- Weigh 'similarity' score with levenshtein distance, Tf-Idf similarity and symbolic n-grams with PCA. 
- Web Hosting 
- Better UI experience (obviously)




## Demo


#### Question 
![image](https://user-images.githubusercontent.com/8717434/42413108-5908b464-81ce-11e8-9828-e05d647f0331.png)

#### Answer 
![image](https://user-images.githubusercontent.com/8717434/42413110-64130774-81ce-11e8-9d79-37f28e113375.png)



