# DrLongHealth

## Development notes
This project uses git submodule. In order to add submodules when cloning repo, first clone the repo, then init submodules:

```bash
git clone https://github.com/rballachay/DrLongHealth
git submodule init
git submodule update
```

You will also need to have git LFS installed. Follow the instructions [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).  

## Introduction
Code for COMP545 Project: LLM modelling of LongHealth Clinical Notes. This project aims to test out multiple methods for improving the performance of the task of long-document question answering on the corpus LongHealth. We aim to achieve this objective using multiple different techniques:

1. Retriever-reader method using bi-encoder DPR to lookup passages in the clinical notes
- Find multiple starting models for encoder (use some zero-shot) and question-answering
- Fine-tune encoders on another open-source dataset
- Train 'reader' (question-answering model)

2. Long-document model with sparse attention 
- Find multiple starting models for long document QA 
- Fine-tune long-document models on another open-source dataset

3. Reproduce the results from the original paper 
- Use API's to provide us a baseline and ensure we are doing it propery

## Training Data
We use the emrQA dataset, which contains the clinical documents of 258 patients, with a total of approximately 223k QA pairs. The original emrQA dataset only contains questions and answers, however, since we want to train our models with multiple choices, we generated distractors for each question.

## Part 1: Retriever-reader method
This method will be built largely off of [_Efficient and Robust Question Answering from Minimal Context over Documents_](https://arxiv.org/abs/1805.08092). This will choose a different number of sentences depending upon the question being asked, using a threshold for the relevance of the top n retrieved sentences from the document. 



## Developing the retriever 

The first few training trials of the DPR retriever did not go particularly well. The training set of emrQA immediately did very well (>99% top-10 accuracy after 1 epoch), however it did not perform very well at all on the LongHealth dataset. This may be attributable to the fact that the questions in the LongHealth dataset are considerably more difficult. In addition, the formulation of the questions are different, as they are asking multiple choice questions. Here are examples of the questions from the emrQA dataset:


```
Question: "Has the patient ever had enteric-coated aspirin"
Answer: "2. Enteric-coated aspirin 325 mg p.o. daily"

Question: "Why did the patient need digoxin."
Answer: "fibrillation. The digoxin was also added for heart rate control"

Question: "Does this woman have a history of labor or maternal fever"
Answer: "There was no labor or maternal fever"
```


Compare these three question/answer pairs to those from the LongHealth dataset:


```
Question: What was the diagnosis for Mrs. Anderson during her inpatient treatment from 07/20/2023 to 09/12/2023?
Correct Answer: Seropneumothorax secondary to punction of a malignant pleural effusion with progressive pulmonary metastasis of a pancreatic head carcinoma
Information: Diagnosis**
Seropneumothorax secondary to punction of a malignant pleural effusion
with progressive pulmonary metastasis of a pancreatic head carc 

Question: Which of the following statements regarding Mrs. Anderson's chemotherapy is accurate?
Correct Answer: Gemcitabine/nab-paclitaxel was started in 07/22 after surgery.
Information: Nurse
-Status post Whipple surgery on 12/22
-3 months adjuvant CTx with gemcitabin/5 

Question: Based on the sequence of interventions and radiologic findings, which of the following statements is accurate regarding Mrs. Anderson's clinical progression?
Correct Answer: The pleural effusion that was initially drained led to an improvement in oxygen saturation, but was followed by the development of a seropneumothorax, necessitating a chest tube placement.
Information:  97% under 3L of O2. X-rays revealed a significant right-sided pleural
effusion, which was drained. After the procedure, the patient\'s
shortness of breath improved, with SpO2 at 95% under 2L of O2. However,
an x-ray follow-up displayed a seropneumothorax, which became more
evident over time, leading to the placement of a chest tube by the
thoracic side 
```
