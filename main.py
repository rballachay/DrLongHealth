from models.dpr import DPRModel,inbatch_negative_sampling, get_topk_indices,contrastive_loss_criterion, recall_at_k
from src.utils import collate_longhealth, collate_emrQA
import torch
import random
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dpr():
    """
    Funtion that trains our DPR for retrieval of relevant passages
    """
    random.seed(0)

    N_EPOCHS = 128
    MAX_LENGTH = 16
    BATCH_SIZE = 16

    questions, answers = collate_emrQA()

    longhealth_docs, longhealth_qs, longhealth_infos = collate_longhealth()

    model = DPRModel(MAX_LENGTH)
    optimizer = model.optimizer

    results = {'training_loss':[],'epoch':[],'eval_recall':[]}
    # start training/eval loop
    for epoch in range(N_EPOCHS):
        model.train()

        training_loss = []
        for i in range(0,len(questions)//BATCH_SIZE): 
            questions_i = questions[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            answers_i = answers[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            answers_stack = model.embed_passages(answers_i)
            qa_stack = model.embed_questions(questions_i)

            # Implement in-batch negative sampling
            S = inbatch_negative_sampling(qa_stack, answers_stack)

            loss = contrastive_loss_criterion(S)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(float(loss))
            print(f"Training loss, {i}: {loss:.2f}")

            if i>10:
                break

        model.eval()

        eval_recall = []
        # we are evaluating patient-by-patient
        for patient_i in range(len(longhealth_docs)):
            # each question is a tuple of the question and the multuple choice answers
            patient_qs = [j[0] for j in longhealth_qs[patient_i]]

            # add the actual question answer passages back to the corpus, then shuffle
            patient_docs = longhealth_docs[patient_i] + longhealth_infos[patient_i]

            l_docs = len(longhealth_docs[patient_i])
            patient_idx = np.arange(l_docs,l_docs+len(longhealth_infos[patient_i]))
            
            patient_doc_mtx = model.embed_passages(patient_docs)
            qa_stack = model.embed_questions(patient_qs)

            indices, scores = get_topk_indices(qa_stack, patient_doc_mtx, k=10)  

            recall = recall_at_k(indices, patient_idx, k=10)   

            eval_recall.append(recall)

        results['training_loss'].append(np.mean(training_loss))
        results['eval_recall'].append(np.mean(eval_recall))
        results['epoch'].append(epoch+1)

        # save the current state of the model 
        torch.save(model.state_dict(), f"models/dpr_training_best.pth")
        pd.DataFrame(results).to_csv(f'results/dpr_training.csv',index=False)
        


if __name__=="__main__":
    train_dpr()