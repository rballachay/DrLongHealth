from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch
from .utils import get_topk_indices

class DPRModel(nn.Module):
    def __init__(self, max_length=512, device='cpu'):
        super(DPRModel, self).__init__()
        self.max_length = max_length
        self.device = device
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)

    def embed_passages(self, passages: 'list[str]'):
        batch = self.tokenizer(passages,padding=True, max_length=self.max_length, truncation=True, return_tensors='pt').to(self.device)
        results = self.bert(**batch)
        return results.last_hidden_state[:,0,:].to(self.device)

    def embed_questions(self, titles, bodies=None):
        if bodies:
            questions = [(qt,qb) for qt,qb in zip(titles,bodies)]
        else:
            questions=titles
        batch = self.tokenizer(questions,padding=True, max_length=self.max_length, truncation=True, return_tensors='pt').to(self.device)
        results = self.bert(**batch)
        return results.last_hidden_state[:,0,:].to(self.device)
        

def get_relevant_passages(model:nn.Module, answer_docs:list[str], question:str, n_passages:int):
    n_passages = min(n_passages,len(answer_docs))
    patient_doc_mtx = model.embed_passages(answer_docs).to(model.device)
    qa_stack = model.embed_questions([question]).to(model.device)
    indices, scores = get_topk_indices(qa_stack, patient_doc_mtx, k=n_passages)  
    indices = indices.flatten().tolist()
    return ' ... '.join([answer_docs[i] for i in indices])

"""
def get_relevant_passages(model:nn.Module, answer_docs:list[str], question:str, n_passages:int):
    '''
    If the answer docs are longer than a certain length, this takes up too much memory.
    break in into chunks and iterate over these chunks instead
    '''
    indices = []
    scores = []
    for answer_docs_i in [answer_docs[i:i+200] for i in range(0, len(answer_docs), 200)]:
        n_passages = min(n_passages,len(answer_docs_i))
        patient_doc_mtx = model.embed_passages(answer_docs_i).to(model.device)
        qa_stack = model.embed_questions([question]).to(model.device)
        indices_i, scores_i = get_topk_indices(qa_stack, patient_doc_mtx, k=n_passages)  
        indices_i = indices_i.flatten().tolist()
        scores_i = scores_i.flatten().tolist()

        indices.extend(indices_i)
        scores.extend(scores_i)
    
    sorted_items = sorted(zip(indices, scores), key=lambda x: x[1],reverse=True)

    # Extract the max items from the sorted list
    indices_max = [item[0] for item in sorted_items[:n_passages]] 

    return ' ... '.join([answer_docs[i] for i in indices_max])
"""