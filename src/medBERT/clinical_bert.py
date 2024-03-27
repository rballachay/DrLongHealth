from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm
from src.utils import break_text_into_passages, format_prompt
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import re 

class ClinicalBertQA(nn.Module):
    def __init__(self):
        super(ClinicalBertQA, self).__init__()
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.pred_layer = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, inputs):
        """this method is borrowed from from here:
        https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_bert.html#BertForQuestionAnswering
        """
        tokens = self.tokenizer(inputs, padding=True, max_length=512, truncation=True, return_tensors='pt').to(device)
        outputs = self.bert(**tokens)
        sequence_output = outputs[0]

        logits = self.pred_layer(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) #+ outputs[2:]
        return outputs # (start_logits, end_logits)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_clinical_bert():
    model = ClinicalBertQA()

    with open("data/LongHealth/data/benchmark_v5.json", "r") as f:
        benchmark = json.load(f)

    for idx, patient in tqdm(benchmark.items(), position=0):
        # patient texts are a dictionary of texts labelled like 
        # text_0, text_1 ...
        
        patient_text_lengths = {t:len(t) for t in patient['texts']}

        patient_doc_matrix, patient_document = get_patient_doc_matrix(patient, model, device)

        for i, question in tqdm(
            enumerate(patient["questions"]),
            position=1,
            leave=False,
            total=len(patient["questions"]),
        ):
            
            # for each of the question/answers, we are going to get a lookup 
            question_str = question['question']
            
            questions = []
            answers = []
            # remember this is multiple QA, going to stack each of them in the zero dimension
            for answer in filter(lambda x: re.match(r"^answer_[a-z]$",x), question.keys()):
                questions.append(question_str)
                answers.append(answer)

            qa_stack = embed_questions(questions, answers, model, device)

            # get the top-5 entries in the original document for each question, then 
            # get all the unique ones. This will help us to formulate the best passages
            # we can extract for the question 
            indices, _ = get_topk_indices(qa_stack, patient_doc_matrix, k=5)
            
            unq_indices = torch.unique(indices.flatten())

            retrieved_passages = '\n'.join([patient_document[i] for i in unq_indices])

            model_prompt = format_prompt(retrieved_passages,question)

            results = model(model_prompt)

            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs
                



            #input_ids = tokenizer_qa.encode(model_prompt, return_tensors="pt", max_length=512, truncation=True)
            #outputs = model_qa.generate(input_ids)
            #response = tokenizer_qa.decode(outputs[0], skip_special_tokens=True)

            #print(response)

def get_patient_doc_matrix(patient, model, device, max_length=256):
    patient_texts = '\n'.join(patient["texts"].values())
    patient_document = break_text_into_passages(patient_texts,max_length)
    passages = []
    for passage in patient_document:
        embedded = embed_passages(passage ,model, device, max_length)
        passages.append(embedded)
    return torch.cat(passages), patient_document

def embed_passages(passages: 'list[str]', model, device='cpu', max_length=256):
    model.eval()
    batch = model.tokenizer(passages,padding=True, max_length=max_length, truncation=True, return_tensors='pt').to(device)
    results = model.bert(**batch)
    return results.last_hidden_state[:,0,:]

def embed_questions(titles, bodies, model, device='cpu', max_length=512):
    model.eval()
    questions = [(qt,qb) for qt,qb in zip(titles,bodies)]
    batch = model.tokenizer(questions,padding=True, max_length=max_length, truncation=True, return_tensors='pt').to(device)
    results = model.bert(**batch)
    return results.last_hidden_state[:,0,:]

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

if __name__=="__main__":
    run_clinical_bert()