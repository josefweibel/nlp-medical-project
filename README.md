# NLP Project

The given dataset is from https://huggingface.co/datasets/argilla/medical-domain and contains Medical transcription data scraped from mtsamples.com the dataset contains sample medical transcriptions for various medical specialities.

We were given 3 tasks:
* Task 1: Data Exploration and Processing in 'Task1.ipynb'
  - Computing some basic statistics of the data, we found that the dataset is very unbalanced and that it includes not only Single-Labels but Multi-Labels and that the text lenght of the texts with certain Labels do vary a lot.

### Key Visualizations
- Distribution of Text Lengths
![Distribution of Text Lengths](/plots/distribution_of_text_lengths.png)
- Distribution of Labels
![Distribution of Labels](/plots/distribution_labels.png)
- Heatmap of Text Length vs. Class Distribution
![Heatmap of Text Length vs. Class Distribution](/plots/heatmap_text_length_class_distribution.png)


* Task 2: NER in 'Task2.ipynb'
  - Using spaCy we investigated the NER types and noticed that many get missclassified due to the charateristics of the medical/domain specific nature of texts, we therefore needed to add/create our own labels and trained a classifer for this.

* Task 3: Classification with Traditional ML BERT and GPT-Like Models in 'Task3.ipynb'
  - We tried to classify the texts into 40 different categories using different models.

These notebook provide details about our work. A sumamrisation of it can be found in the presentation stored in directory `presentations`.

Requirements are specificed in the `requirements.txt` or `Pipfile`.
