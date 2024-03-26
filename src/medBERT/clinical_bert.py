from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm
from src.utils import break_text_into_passages, format_prompt
import torch
import re 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_clinical_bert():

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Load the model
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    #tokenizer_qa = AutoModelForSeq2SeqLM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer_qa = AutoTokenizer.from_pretrained("facebook/bart-large")
    model_qa = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

    #qa_pipeline= pipeline("text-generation", model="yikuan8/Clinical-Longformer", tokenizer="yikuan8/Clinical-Longformer")

    with open("data/LongHealth/data/benchmark_v5.json", "r") as f:
        benchmark = json.load(f)

    for idx, patient in tqdm(benchmark.items(), position=0):
        # patient texts are a dictionary of texts labelled like 
        # text_0, text_1 ...
        patient_doc_matrix, patient_document = get_patient_doc_matrix(patient, model, tokenizer, device)

        patient_results = {}
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

            qa_stack = embed_questions(questions, answers, model, tokenizer, device)

            # get the top-5 entries in the original document for each question, then 
            # get all the unique ones. This will help us to formulate the best passages
            # we can extract for the question 
            indices, _ = get_topk_indices(qa_stack, patient_doc_matrix, k=5)
            
            unq_indices = torch.unique(indices.flatten())

            retrieved_passages = '\n'.join([patient_document[i] for i in unq_indices])

            model_prompt = format_prompt(retrieved_passages,question)
            
            input_ids = tokenizer_qa.encode(model_prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model_qa.generate(input_ids)
            response = tokenizer_qa.decode(outputs[0], skip_special_tokens=True)

            print(response)

def get_patient_doc_matrix(patient, model, tokenizer, device, max_length=256):
    patient_texts = '\n'.join(patient["texts"].values())
    patient_document = break_text_into_passages(patient_texts,max_length)
    passages = []
    for passage in patient_document:
        embedded = embed_passages(passage,model,tokenizer, device, max_length)
        passages.append(embedded)
    return torch.cat(passages), patient_document

def embed_passages(passages: 'list[str]', model, tokenizer, device='cpu', max_length=256):
    model.eval()
    batch = tokenizer(passages,padding=True, max_length=max_length, truncation=True, return_tensors='pt').to(device)
    results = model(**batch)
    return results.last_hidden_state[:,0,:]

def embed_questions(titles, bodies, model, tokenizer, device='cpu', max_length=512):
    model.eval()
    questions = [(qt,qb) for qt,qb in zip(titles,bodies)]
    batch = tokenizer(questions,padding=True, max_length=max_length, truncation=True, return_tensors='pt').to(device)
    results = model(**batch)
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