import subprocess
import os
import json
from tqdm import tqdm
from data.LongHealth.utils import SYSTEM_PROMPT, create_prompt, query_model
from src.utils import create_prompt_custom
from models.dpr import DPRModel, get_relevant_passages, get_relevant_indices
from transformers import AutoTokenizer
from psutil import process_iter
from signal import SIGTERM # or SIGKILL
import time
import torch
from src.utils import break_text_into_passages
import re
import pandas as pd
import random
import time 

DOTIME = True

# these are the models from hugging face we have decided to use
MODEL_LIST = ['lmsys/vicuna-7b-v1.5-16k'] #'mistralai/Mistral-7B-Instruct-v0.2',

URL = "http://localhost:8000/v1/chat/completions"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference(task:int=1, dpr:bool=False, n_passages:int=25, data_path:str="data/LongHealth/data/benchmark_v5.json", outdir:str='results', max_len=16_000, dpr_path:str='models/dpr_training_best.pth'):
    epoch = int(time.time())

    with open(f'{epoch}_time_logs.txt', 'a') as file:
        file.write('Task\tDPR\tQuestion\tModel\tTime Elapsed\n')

    if task not in [1,2,3]:
        raise Exception("Task must be one of {1,2,3}")
    
    outdir = f"{outdir}/task_{task}"
    
    dpr_model = None

    if dpr:
        dpr_model = DPRModel(256, "cuda:1").to("cuda:1")
        dpr_model.load_state_dict(torch.load(dpr_path))
    
    with open(data_path, "r") as f:
        benchmark = json.load(f)

    for model in MODEL_LIST:
        command = ["./data/LongHealth/serve.sh", model]

        # serve the fastchat model
        process = subprocess.Popen(command)

        time.sleep(10)
        print('starting!')

        tokenizer = AutoTokenizer.from_pretrained(model)  

        model = model.split("/")[-1]
        
        path =f"{outdir}/{model}.json"

        if dpr:
            path = path.replace('.json','-DPR-retriever.json')

        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if (not DOTIME) and os.path.exists(path):
            with open(path, "r") as f:
                eval_results = json.load(f)
        else:
            eval_results = {}

        max_len = max_len - len(tokenizer.encode(SYSTEM_PROMPT))


        question_idx = 0
        for idx, patient in tqdm(benchmark.items(), position=0):
            patient_results = {}
            if idx in eval_results.keys():
                continue
            for i, question in tqdm(
                enumerate(patient["questions"]),
                position=1,
                leave=False,
                total=len(patient["questions"]),
            ):
                time_start = time.time()

                if patient_results.get(f"question_{i}"):
                    continue
                
                if task==1:
                    patient_results[f"question_{i}"] = {"correct": question["correct"]}
                    answer_docs, non_answer_docs = get_task_1(patient, question)
                elif task==2:
                    question["answer_f"] = "Question cannot be answered with provided documents"
                    non_answer_docs = sample_distractions(idx, benchmark, n=10)
                    patient_results[f"question_{i}"] =  {"correct": question["correct"]}
                    answer_docs = {
                        text_id: patient["texts"][text_id]
                        for text_id in question["answer_location"]
                    }
                elif task==3:
                    question["answer_f"] = "Question cannot be answered with provided documents"
                    non_answer_docs = sample_distractions(idx, benchmark, n=10)
                    patient_results[f"question_{i}"] = {"correct":"Question cannot be answered with provided documents"}
                    answer_docs = {}
                    

                # we are not doing it 5 times because it doesn't vary enough 
                for j in [1]:
                    # create_prompt will shuffle the documents in the prompt each time
                    if not dpr:
                        prompt, answer_location = create_prompt(
                            answer_docs,
                            non_answer_docs,
                            question,
                            max_len=max_len,
                            tokenizer=tokenizer,
                        )
                    else:
                        mq_answers = ""
                        # remember this is multiple QA, going to convert to a single string 
                        for answer_key in filter(lambda x: re.match(r"^answer_[a-z]$",x), question.keys()):
                            mq_answers += f"{question[answer_key]} "

                        if task==1:
                            passages = break_text_into_passages('\n'.join(list(answer_docs.values())), 256)
                        elif task==2:
                            passages = break_text_into_passages('\n'.join(list(answer_docs.values())+non_answer_docs), 256)
                        elif task==3:
                            passages = break_text_into_passages('\n'.join(non_answer_docs), 256)
                        
                        question_str = question['question']+' answers:'+ mq_answers
                        
                        # only getting the first n relevant passages 
                        relevant_passages = get_relevant_passages(dpr_model, passages, question_str, n_passages=n_passages)
                        prompt = create_prompt_custom(
                            relevant_passages,
                            question,
                        )
                        answer_location = {} # don't really need this anyways

                    response = query_model(prompt, model=model, system_prompt=SYSTEM_PROMPT, url=URL)
                    choice = response.json()["choices"][0]
                    patient_results[f"question_{i}"][f"answer_{j}"] = choice["message"]["content"]
                    patient_results[f"question_{i}"][f"answer_{j}_locations"] = answer_location

                    time_elapsed = time.time()-time_start

                    with open(f'{epoch}_time_logs.txt', 'a') as file:
                        file.write(f'{task}\t{dpr}\t{question_idx}\t{model}\t{time_elapsed:.2f}\n')

                    question_idx+=1

            if not DOTIME:
                eval_results[idx] = patient_results
                with open(path, "w+") as f:
                    json.dump(eval_results, f)


        process.terminate()

def get_task_1(patient, question):
    answer_docs = {
                    text_id: patient["texts"][text_id] for text_id in question["answer_location"]
                }
    non_answer_docs = [
        text
        for text_id, text in patient["texts"].items()
        if text_id not in question["answer_location"]
    ]
    return answer_docs, non_answer_docs

def sample_distractions(patien_id: str, benchmark: dict, n: int = 4):
    """samples `n` texts from the benchmark, that are not from patient with `patient_id`"""

    all_texts = [
        text
        for pid, patients in benchmark.items()
        if pid != patien_id
        for text in patients["texts"].values()
    ]
    sampled_texts = random.sample(all_texts, min(n, len(all_texts)))
    return sampled_texts

def calculate_dpr_sizes(task=1,data_path:str="data/LongHealth/data/benchmark_v5.json", dpr_path:str='models/dpr_training_best.pth'):

    if task not in [1,2,3]:
        raise Exception("Task must be one of {1,2,3}")
    
    with open(data_path, "r") as f:
        benchmark = json.load(f)

    dpr_model = DPRModel(256, "cuda:0").to("cuda:0")
    dpr_model.load_state_dict(torch.load(dpr_path))
    
    results={'patient':[],'question':[],'l_prompt':[],'n_passages':[]}

    for idx, patient in tqdm(benchmark.items(), position=0):
            for i, question in tqdm(
                enumerate(patient["questions"]),
                position=1,
                leave=False,
                total=len(patient["questions"]),
            ):
                if task==1:
                    answer_docs, non_answer_docs = get_task_1(patient, question)
                elif task==2:
                    question["answer_f"] = "Question cannot be answered with provided documents"
                    non_answer_docs = sample_distractions(idx, benchmark, n=10)
                    answer_docs = {
                        text_id: patient["texts"][text_id]
                        for text_id in question["answer_location"]
                    }
                elif task==3:
                    question["answer_f"] = "Question cannot be answered with provided documents"
                    non_answer_docs = sample_distractions(idx, benchmark, n=10)
                    answer_docs = {}

                mq_answers = ""
                # remember this is multiple QA, going to convert to a single string 
                for answer_key in filter(lambda x: re.match(r"^answer_[a-z]$",x), question.keys()):
                    mq_answers += f"{question[answer_key]} "

                if task==1:
                    passages = break_text_into_passages('\n'.join(list(answer_docs.values())), 256)
                elif task==2:
                    passages = break_text_into_passages('\n'.join(list(answer_docs.values())+non_answer_docs), 256)
                elif task==3:
                    passages = break_text_into_passages('\n'.join(non_answer_docs), 256)

                question_str = question['question']+' answers:'+ mq_answers

                for n_passages in [10,25,10000]:
                    # only getting the first n relevant passages 
                    relevant_passages = get_relevant_passages(dpr_model, passages, question_str, n_passages=n_passages)

                    prompt = create_prompt_custom(
                                relevant_passages,
                                question,
                            )

                    results['patient'].append(idx)
                    results['question'].append(question['No'])
                    results['n_passages'].append(n_passages)
                    results['l_prompt'].append(len(prompt))

    pd.DataFrame(results).to_csv(f'results/task_{task}/length_prompts.csv')


def calculate_dpr_locs(task=1,data_path:str="data/LongHealth/data/benchmark_v5.json", dpr_path:str='models/dpr_training_best.pth'):
    
    with open(data_path, "r") as f:
        benchmark = json.load(f)

    dpr_model = DPRModel(256, "cuda:0").to("cuda:0")
    dpr_model.load_state_dict(torch.load(dpr_path))
    
    results={'patient':[],'question':[],
             'index':[], 'l_embedding':[], 'start_loc':[],
             'length_doc':[]}

    for idx, patient in tqdm(benchmark.items(), position=0):
            for i, question in tqdm(
                enumerate(patient["questions"]),
                position=1,
                leave=False,
                total=len(patient["questions"]),
            ):
                
                non_answer_docs = [
                    text
                    for text_id, text in patient["texts"].items()
                ]
                
                mq_answers = ""
                # remember this is multiple QA, going to convert to a single string 
                for answer_key in filter(lambda x: re.match(r"^answer_[a-z]$",x), question.keys()):
                    mq_answers += f"{question[answer_key]} "

                passages = break_text_into_passages('\n'.join(list(non_answer_docs)), 256)

                question_str = question['question']+' answers:'+ mq_answers

                relevant_indices = get_relevant_indices(dpr_model, passages, question_str, n_passages=5)

                for pass_num, index_num in enumerate(relevant_indices):
                    results['patient'].append(idx)
                    results['question'].append(question['No'])
                    results['index'].append(pass_num)
                    results['l_embedding'].append(len(passages[index_num]))
                    results['start_loc'].append(sum([len(i) for i in passages[:index_num]]))
                    results['length_doc'].append(''.join(non_answer_docs))

    pd.DataFrame(results).to_csv(f'results/task_{task}/prompt_locations.csv')

def KILL():
    for proc in process_iter():
        for conns in proc.connections(kind='inet'):
            if conns.laddr.port == 8000:
                proc.send_signal(SIGTERM) # or SIGKILL
            elif conns.laddr.port == 21001:
                proc.send_signal(SIGTERM) # or SIGKILL
            elif conns.laddr.port == 31000:
                proc.send_signal(SIGTERM) # or SIGKILL


if __name__=="__main__":
    #KILL()
    try:
        #calculate_dpr_locs()
        #calculate_dpr_sizes(task=2)
        run_inference(task=1, dpr=False, n_passages=10)
        #run_inference(task=3, dpr=True, n_passages=25)
    except Exception as e:
        print(f"Excepted! Killing all processes running on ports\n{e}")
    finally:
        #KILL()a
       pass
