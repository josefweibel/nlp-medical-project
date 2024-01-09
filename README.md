# NLP Project

The given dataset is from https://huggingface.co/datasets/argilla/medical-domain and contains Medical transcription data scraped from mtsamples.com the dataset contains sample medical transcriptions for various medical specialities.

We were given 3 tasks:
### Task 1: Data Exploration and Processing in 'Task1.ipynb'
  - Computing some basic statistics of the data, we found that the dataset is very unbalanced and that it includes not only Single-Labels but Multi-Labels and that the text lenght of the texts with certain Labels do vary a lot.

##### Some key statistics are:
- Distribution of Text Lengths
<img src="/plots/distribution_of_text_lengths.png" width="400">
- Distribution of Labels
<img src="/plots/distribution_labels.png" width="400">
- Heatmap of Text Length vs. Class Distribution
<img src="/plots/heatmap_text_length_class_distribution.png" width="400">



### Task 2: NER in 'Task2.ipynb'
  - Using spaCy we investigated the NER types and noticed that many get missclassified due to the charateristics of the medical/domain specific nature of texts, we therefore needed to add/create our own labels and trained a classifer for this.
  - We also tried to train a NER model using spaCy, but due to the nature of the data we were not able to get good results.
  - In our second task, we focused on training a Named Entity Recognition (NER) model. The training progress plot shows how the model's loss decreased over time, indicating successful learning of the new label (drug)

##### Training Progress
<img src="/plots/training_NER.png" width="400">

##### Metrics Drug Label
<img src="/plots/metrics_drug_label.png" width="400">

### Task 3: Classification with Traditional ML BERT and GPT-Like Models in 'Task3.ipynb'
  - We tried to classify the texts into 40 different categories using different models.


#### F1-Scores by Model
- overall Scores 
![overall Scores](/plots/overall_scores.png)
- statistics for generated responses
![statistics for generated responses](/plots/generated_responses.png)

 The Baseline models show consistent performance with a good balance between precision and recall, while the fine-tuned models like BERT and ClinicalBERT demonstrate improvements in specific metrics. The Llama models exhibit varying degrees of performance across different medical text categories, with the Llama 2 (13B) model generally outperforming the Llama 2 (7B) model. Despite their advancements, these models still face challenges in accurately categorizing medical texts, as seen in the varying F1-scores. This evaluation serves as a critical assessment for future improvements in model training and application in the medical field.


These notebook provide details about our work. Further summarization and details on explanation can be found in the presentation stored in directory `presentations`.

Requirements are specificed in the `requirements.txt` or `Pipfile`.
