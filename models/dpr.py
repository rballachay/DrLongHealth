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
    patient_doc_mtx = model.embed_passages(answer_docs).to(model.device)
    qa_stack = model.embed_questions([question]).to(model.device)
    indices, scores = get_topk_indices(qa_stack, patient_doc_mtx, k=n_passages)  
    indices = indices.flatten().tolist()

    return ' ... '.join([answer_docs[i] for i in indices])