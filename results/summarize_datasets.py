import json
import numpy as np
import pandas as pd


def summarize_emrQA_dataset(medication_data='data/emrQA/clean/medication-SQUAD.json',
                            relation_data='data/emrQA/clean/relation-SQUAD.json'):
    
    with open(medication_data, "r") as f:
        medication_data = json.load(f)

    with open(relation_data, "r") as f:
        relation_data = json.load(f)

    # 
    results = {'Dataset':[],'Question Length':[], 'Number of Questions':[],
               'Passage Length':[]}
    question_answer = []
    question_prompt = []
    contexts = []
    n_questions = {}
    for record in medication_data['data']:
        for paragraph in record['paragraphs']:
            context = paragraph['context']
            contexts.append(context)
            for question in paragraph['qas']:
                if question['answers'][0]['text'] not in question_answer:
                    question_answer.append(question['answers'][0]['text'])
                    question_prompt.append(question['question'])
                    n_questions[question['answers'][0]['text']]=1
                else:
                    n_questions[question['answers'][0]['text']]+=1


    results['Dataset'].append('medication')
    results['Question Length'].append(np.mean([len(q.split()) for q in question_prompt]))
    results['Passage Length'].append(np.mean([len(q.split()) for q in question_answer]))
    #results['Document Length'].append(np.mean([len(q.split()) for q in contexts]))
    #results['Questions per Excerpt'].append(np.mean([*n_questions.values()]))
    results['Number of Questions'].append(len(question_prompt))

    question_answer = []
    question_prompt = []
    contexts = []
    n_questions = {}
    for record in relation_data['data']:
        for paragraph in record['paragraphs']:
            context = paragraph['context']
            contexts.append(context)
            for question in paragraph['qas']:
                if question['answers'][0]['text'] not in question_answer:
                    question_answer.append(question['answers'][0]['text'])
                    question_prompt.append(question['question'])
                    n_questions[question['answers'][0]['text']]=1
                else:
                    n_questions[question['answers'][0]['text']]+=1

    results['Dataset'].append('relation')
    results['Question Length'].append(np.mean([len(q.split()) for q in question_prompt]))
    results['Passage Length'].append(np.mean([len(q.split()) for q in question_answer]))
    #results['Document Length'].append(np.mean([len(q.split()) for q in contexts]))
    #results['Questions per Excerpt'].append(np.mean([*n_questions.values()]))
    results['Number of Questions'].append(len(question_prompt))


    print(pd.DataFrame(results).to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.1f}".format,))


def summarize_longhealth_dataset(longhealth_data='data/LongHealth/data/benchmark_v5.json'):

    with open(longhealth_data, "r") as f:
        benchmark = json.load(f)

    results = {'Num Questions':[],
               'Question Length':[],'Passage Length':[], 
               'Document Length':[]}

    questions = []
    passages = []
    
    for patient_idx, patient in benchmark.items():
        patient_texts = ''.join(patient["texts"].values())


        for question in patient["questions"]:
            question_str = question['question']
            questions.append(question_str)

            info_str = ""
            for text_id, keys in question["answer_location"].items():
                text = patient['texts'][text_id]
                start_vals = keys["start"]
                end_vals = keys["end"]

                for start, end in zip(start_vals, end_vals):
                    start_idx = int(start * len(text))
                    end_idx = int(end * len(text))

                    if start_idx==end_idx:
                        continue

                    info_str+=f"{text[start_idx:end_idx]} "
            
            passages.append(info_str)

            print(question_str)
            print(info_str)
            print("\n\n")

            
        #results["Patient"].append(int(patient_idx.split('_')[1]))
    results['Question Length'].append(np.mean([len(i.split()) for i in questions]))
    results['Passage Length'].append(np.mean([len(i.split()) for i in passages]))
    results['Num Questions'].append(len(passages))
    results['Document Length'].append(len(patient_texts.split()))

    print(pd.DataFrame(results).to_latex(index=False,
                formatters={"name": str.upper},
                float_format="{:.1f}".format,))



if __name__=='__main__':
    #summarize_emrQA_dataset()
    summarize_longhealth_dataset()