from torch import nn
from transformers import AutoTokenizer, AutoModel, BatchEncoding
import torch
import torch.nn.functional as F

class DPRModel(nn.Module):
    def __init__(self, max_length=512):
        super(DPRModel, self).__init__()
        self.max_length = max_length
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)

    def embed_passages(self, passages: 'list[str]'):
        batch = self.tokenizer(passages,padding=True, max_length=self.max_length, truncation=True, return_tensors='pt')
        results = self.bert(**batch)
        return results.last_hidden_state[:,0,:]

    def embed_questions(self, titles, bodies):
        questions = [(qt,qb) for qt,qb in zip(titles,bodies)]
        batch = self.tokenizer(questions,padding=True, max_length=self.max_length, truncation=True, return_tensors='pt')
        results = self.bert(**batch)
        return results.last_hidden_state[:,0,:]


def inbatch_negative_sampling(Q: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    # row - wise dot product
    return torch.matmul(Q, P.T)  

def contrastive_loss_criterion(S: torch.Tensor, labels: torch.Tensor = None):
    softmax_scores = F.log_softmax(S, dim=1)

    if labels is  None:
        labels = torch.range(0,S.shape[0]-1,dtype=torch.long)

    loss = F.nll_loss(
            softmax_scores,
            torch.tensor(labels),
            reduction="mean",
        )
    return loss


def get_topk_indices(Q, P, k: int = None):
    """in the original usage of this function, the first index was the batch, so
    here we have replaced that with the a,b,c,d of the question
    """
    Q_prime = Q.unsqueeze(1).repeat(1, P.shape[0], 1).view(-1, Q.size(1))
    P_prime = P.repeat(Q.shape[0], 1)
    dot_product = torch.sum(Q_prime * P_prime, dim=1)
    dot_product = dot_product.reshape((Q.shape[0],P.shape[0]))
    top_k = torch.topk(dot_product, k)
    return top_k.indices, top_k.values

def tokenize_qa_batch(model, questions, answers, max_length=64) -> BatchEncoding:
    q_batch = model.tokenizer(questions,padding=True, max_length=max_length, truncation=True, return_tensors='pt')
    a_batch = model.tokenizer(answers,padding=True, max_length=max_length, truncation=True, return_tensors='pt')
    return q_batch, a_batch