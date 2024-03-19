# DrLongHealth
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
I think we should use the following dataset: 

## Part 1: Retriever-reader method
This method will be built largely off of [_Efficient and Robust Question Answering from Minimal Context over Documents_](https://arxiv.org/abs/1805.08092). This will choose a different number of sentences depending upon the question being asked, using a threshold for the relevance of the top n retrieved sentences from the document. 