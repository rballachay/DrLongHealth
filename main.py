from models.dpr import DPRModel,inbatch_negative_sampling, get_topk_indices,contrastive_loss_criterion
from src.utils import break_text_into_passages, collate_longhealth
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_patient_doc_matrix(patient, model, max_length=512):
    patient_texts = '\n'.join(patient["texts"].values())
    patient_document = break_text_into_passages(patient_texts,max_length)
    passages = []
    for passage in patient_document:
        embedded = model.embed_passages(passage)
        passages.append(embedded)
    return torch.cat(passages), patient_document

def train_dpr():
    """
    Funtion that trains our DPR for retrieval of relevant passages
    """
    MAX_LENGTH = 16
    BATCH_SIZE = 8

    longhealth_docs, longhealth_qs, longhealth_infos = collate_longhealth() #use default location

    model = DPRModel(MAX_LENGTH)

    model.train()
    optimizer = model.optimizer
    for patient_i in range(len(longhealth_docs)):
        patient_doc  = longhealth_docs[patient_i]
        patient_qs = longhealth_qs[patient_i]
        patient_info = longhealth_infos[patient_i]

        #random.shuffle(patient_qs)
        for i in range(0,len(patient_qs)//BATCH_SIZE):
            pat_qs_tup = patient_qs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            patient_qs_i = [j[0] for j in pat_qs_tup]
            patient_as_i = [j[1] for j in pat_qs_tup]

            patient_inf_i = patient_info[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            
            patient_doc_mtx = model.embed_passages(patient_inf_i)
            qa_stack = model.embed_questions(patient_qs_i, patient_as_i)

            # Implement in-batch negative sampling
            S = inbatch_negative_sampling(qa_stack, patient_doc_mtx)

            loss = contrastive_loss_criterion(S)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)


if __name__=="__main__":
    train_dpr()