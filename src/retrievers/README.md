# Retrievers

## Dense passage retrieval
From the paper [_Dense Passage Retrieval_](https://arxiv.org/abs/2004.04906). 

### Explanation of technique

The technique works with two encoders - a passage and a query encoder. The objective is to train the encoder of the question and passage so that the cosine distance between the passage and question embedding is minimized. What that means for this project is that if we train a model to understand the form of medical question answering (ostensibly on the _Publicly Shareable Clinical Large Language Model Built on Synthetic Clinical Notes_), then can use it to extract the most relevant passages out of our long form document without passing the entire tokenized document to our question-answering model (which could be up to 10,000 tokens long). A second part of this could be to use the whole end-to-end question answering system proposed by the same paper. 

### DPR in medical QA context

There are some examples of DPR used in a medical context. I will list these below:

1. [_RedHOT: A Corpus of Annotated Medical Questions, Experiences, and Claims on Social Media_](https://arxiv.org/abs/2210.06331). This work creates a dataset of reddit posts spanning a variety of medical questions. A DPR model is trained on the reddit posts and medical abstracts, so that the final model may act as a medical context retriever. 

2. [_General-Purpose Retrieval-Enhanced Medical Prediction Model Using Near-Infinite History_](https://arxiv.org/abs/2310.20204). While not DPR exactly, this uses a retrieval model over large amounts of EHR data to perform outcome estimation. This is a great example of using a retriever to get relevant info out of a long medical context.

3. [_A large-scale dataset of patient summaries for retrieval-based clinical decision support systems_](https://www.nature.com/articles/s41597-023-02814-8). A DPR model is used for lookup of similar patient records, including the MIMIC-IV dataset with clinical discharge notes, with an average document length of ~1500 words. They use a range of encoders, including bge-base-en-v1.5, MedCPT, PubMedBERT, BERT and SPECTER. 

### Objectives for this part of the project

1. 


## Retrieval-Augmented Generation 
From the paper [_Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_](https://arxiv.org/abs/2005.11401). 

### Explanation of the technique 

Retrieval-augmented generation (RAG) is an extension of DPR which combines a parametric question-answering model with a DPR retrieval from some information source like wikipedia. We could technically use this in addition to the other way we are using DPR, which is to retrieve relevant indices from the input corpus.