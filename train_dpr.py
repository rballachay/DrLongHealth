from models.dpr import DPRModel,inbatch_negative_sampling, get_topk_indices,contrastive_loss_criterion, recall_at_k, accuracy_at_k
from src.utils import collate_longhealth, collate_emrQA
import torch
import random
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dpr():
    """
    Funtion that trains our DPR for retrieval of relevant passages.

    Note that there is a major difference between emrQA and LongHealth, that being that 
    the questions asked of emrQA are relatively simple. 
    """
    random.seed(0)

    N_EPOCHS = 128
    MAX_LENGTH = 256
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 128

    questions_train, answers_train, questions_eval, answers_eval = collate_emrQA()

    longhealth_docs, longhealth_qs, longhealth_infos = collate_longhealth()

    model = DPRModel(MAX_LENGTH, device).to(device)
    optimizer = model.optimizer

    results = {'training_loss':[],'epoch':[],
    'eval_recall_10':[],'eval_recall_50':[],'eval_recall_100':[],
    'eval_acc_10':[],'eval_acc_50':[],'eval_acc_100':[],
    'test_recall_10':[],'test_recall_50':[],'test_recall_100':[],
    'test_acc_10':[],'test_acc_50':[],'test_acc_100':[],
    }
    # start training/eval loop
    for epoch in range(N_EPOCHS):
        model.train()

        training_loss = []
        for i in range(0,len(questions_train)//BATCH_SIZE): 
            questions_i = questions_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            answers_i = answers_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            answers_stack = model.embed_passages(answers_i).to(device)
            qa_stack = model.embed_questions(questions_i).to(device)

            # Implement in-batch negative sampling
            S = inbatch_negative_sampling(qa_stack, answers_stack)

            loss = contrastive_loss_criterion(S, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(float(loss))
            print(f"Training loss, {i}: {loss:.2f}")

        # delete these to clear up memory
        del answers_stack, qa_stack, S, loss

        model.eval()

        eval_recall_10 = []
        eval_recall_50 = []
        eval_recall_100 = []
        eval_acc_10 = []
        eval_acc_50 = []
        eval_acc_100 = []

        # 256 is the closest possible size to what our lookup will be
        for i in range(0,len(questions_eval)//BATCH_SIZE_VAL):  
            questions_i = questions_eval[i*BATCH_SIZE_VAL:(i+1)*BATCH_SIZE_VAL]
            answers_i = answers_eval[i*BATCH_SIZE_VAL:(i+1)*BATCH_SIZE_VAL]
            patient_doc_mtx = model.embed_passages(answers_i)
            qa_stack = model.embed_questions(questions_i)

            indices, scores = get_topk_indices(qa_stack, patient_doc_mtx, k=100)  

            eval_recall_10.append(recall_at_k(indices, np.arange(len(answers_i)), k=10))
            eval_recall_50.append(recall_at_k(indices, np.arange(len(answers_i)), k=50))
            eval_recall_100.append(recall_at_k(indices, np.arange(len(answers_i)), k=100))

            eval_acc_10.append(accuracy_at_k(indices, np.arange(len(answers_i)), k=10))
            eval_acc_50.append(accuracy_at_k(indices, np.arange(len(answers_i)), k=50))
            eval_acc_100.append(accuracy_at_k(indices, np.arange(len(answers_i)), k=100))

            del indices, scores, qa_stack, patient_doc_mtx

        test_recall_10 = []
        test_recall_50 = []
        test_recall_100 = []
        test_acc_10 = []
        test_acc_50 = []
        test_acc_100 = []

        # we are evaluating patient-by-patient
        for patient_i in range(len(longhealth_docs)):
            # each question is a tuple of the question and the multuple choice answers
            patient_qs = [j[0]+'answers: '+j[1] for j in longhealth_qs[patient_i]]

            # randomly sample some data from here to pad out the answers
            l_sample = min(len(longhealth_docs[patient_i]),BATCH_SIZE_VAL)
            longhealth_docs_i = random.sample(longhealth_docs[patient_i],l_sample)

            patient_docs = longhealth_docs_i + longhealth_infos[patient_i]

            l_docs = len(longhealth_docs_i)
            patient_idx = np.arange(l_docs,l_docs+len(longhealth_infos[patient_i]))
            
            patient_doc_mtx = model.embed_passages(patient_docs)
            qa_stack = model.embed_questions(patient_qs)

            indices, scores = get_topk_indices(qa_stack, patient_doc_mtx, k=100)  
 
            test_recall_10.append(recall_at_k(indices, patient_idx, k=10)) 
            test_recall_50.append(recall_at_k(indices, patient_idx, k=50))
            test_recall_100.append(recall_at_k(indices, patient_idx, k=100))

            test_acc_10.append(accuracy_at_k(indices, patient_idx, k=10)) 
            test_acc_50.append(accuracy_at_k(indices, patient_idx, k=50))
            test_acc_100.append(accuracy_at_k(indices, patient_idx, k=100))

            # delete these to clear up memory
            del indices, scores, qa_stack, patient_doc_mtx

        results['training_loss'].append(np.mean(training_loss))
        results['test_recall_10'].append(np.mean(test_recall_10))
        results['test_recall_50'].append(np.mean(test_recall_50))
        results['test_recall_100'].append(np.mean(test_recall_100))
        results['test_acc_10'].append(np.mean(test_acc_10))
        results['test_acc_50'].append(np.mean(test_acc_50))
        results['test_acc_100'].append(np.mean(test_acc_100))
        results['eval_recall_10'].append(np.mean(eval_recall_10))
        results['eval_recall_50'].append(np.mean(eval_recall_50))
        results['eval_recall_100'].append(np.mean(eval_recall_100))
        results['eval_acc_10'].append(np.mean(eval_acc_10))
        results['eval_acc_50'].append(np.mean(eval_acc_50))
        results['eval_acc_100'].append(np.mean(eval_acc_100))
        results['epoch'].append(epoch+1)

        # save the current state of the model 
        torch.save(model.state_dict(), f"models/dpr_training_best.pth")
        pd.DataFrame(results).to_csv(f'results/dpr_training.csv',index=False)
        


if __name__=="__main__":
    train_dpr()