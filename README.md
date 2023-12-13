# NLP Project 

The given dataset is from https://huggingface.co/datasets/argilla/medical-domain and contains Medical transcription data scraped from
mtsamples.com the dataset contains sample medical transcriptions for various medical specialities.

We were given 3 Tasks:
* Task 1: Data Exploration and Processing in 'Task1.ipynb'
  - Computing some basic statistics of the data, we found that the dataset is very unbalanced and that it includes not only Single-Labels but Multi-Labels and that the text lenght of the texts with certain Labels do vary a lot.
  - 
* Task 2: NER  in 'Task2.ipynb'
  - Using spaCy we investigated the NER types and noticed that many get missclassified due to the charateristics of the medical/domain specific nature of texts, we therefore needed to add/create our own labels and trained a classifer for this. 
  
* Task 3: BERT and GPT-Like Models in 'Task3.ipynb'
  - ..
