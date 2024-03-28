import nltk
import pandas as pd
import json
import re
import random
nltk.download('punkt')

SYSTEM_PROMPT = """
You are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge letters from hospitals. When you receive one or more of these letters, you will be expected to carefully review the contents and accurately answer multiple-choice questions related to these documents. 

Your answers should be:
1. Accurate: Make sure your answers are based on the information provided in the letters.
2. Concise: Provide brief and direct answers without unnecessary elaboration.
3. Contextual: Consider the context and specifics of each question to provide the most relevant information.

Remember, your job is to streamline the physician's decision-making process by providing them with accurate and relevant information from discharge summaries. Efficiency and reliability are key.
"""


PROMPT_TEMPLATE = """
--------------BEGIN DOCUMENTS--------------

{documents}

--------------END DOCUMENTS--------------

{question_text}
{options}

Please answer using the following format:
1. Begin your answer with the phrase "The correct answer is".
2. State the letter of the correct option (e.g., A, B, C, D, E).
3. Follow the letter with a colon and the exact text of the option you chose.
4. Make sure your answer is a single, concise sentence.

For example, if the correct answer to a question is option C, and the text for C is 'Acute Bronchitis', your answer should be: 
'The correct answer is C: Acute bronchitis.'
"""


def break_text_into_passages(text, max_passage_length=128):
    """Break a text into passages of a certain length, each containing
    complete sentences. note that we don't want the max passage length to 
    be longer than the max token length passed to our model as there will be 
    a lot of truncation, and information lost
    """

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    fragmented_passages = []
    current_passage = ""
    for sentence in sentences:
        # If adding the current sentence to the current passage exceeds the maximum length,
        # start a new passage
        if len(current_passage) + len(sentence) <= max_passage_length:
            current_passage += sentence + " "
        else:
            # Append the current passage to the list of fragmented passages
            fragmented_passages.append(current_passage.strip())
            # Start a new passage with the current sentence
            current_passage = sentence + " "
    
    # Append the last remaining passage
    if current_passage:
        fragmented_passages.append(current_passage.strip())
    
    return fragmented_passages


def format_prompt(retrieved_passages, question, option_labels='abcde'):
    """format the question using the DPR method to get relevant passages
    from the document prior to passing to our QA model.
    """
    question_text = question["question"]

    options = "\n".join(
        [label.upper() + ": " + question[f"answer_{label}"] for label in option_labels]
    )

    prompt = PROMPT_TEMPLATE.format(
        documents=retrieved_passages, question_text=question_text, options=options
    )
    return prompt


def collate_longhealth(data_path:str="data/LongHealth/data/benchmark_v5.json", answer_path:str="data/LongHealth/data/answer_locations.csv"):
    """iterate over longhealth and convert from a json form into a form that will be useful in DPR
    """
    patient_documents = []
    patient_questions = []
    patient_info_data = []

    with open(data_path, "r") as f:
        benchmark = json.load(f)

    for patient_idx, patient in benchmark.items():
        patient_texts = '\n'.join(patient["texts"].values())
        patient_document = break_text_into_passages(patient_texts, 256) # chosen as half of our max length
        
        # add the entire broken-down patient document here
        patient_documents.append(patient_document)

        patient_qa = []
        patient_info = []

        for question in patient["questions"]:
            question_str = question['question']

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

            mq_answers = ""
            # remember this is multiple QA, going to convert to a single string 
            for answer_key in filter(lambda x: re.match(r"^answer_[a-z]$",x), question.keys()):
                mq_answers += f"{question[answer_key]} "

            #mq_answers = question['correct']

            patient_qa.append((question_str,mq_answers))
            patient_info.append(info_str)
        patient_questions.append(patient_qa)
        patient_info_data.append(patient_info)

    return patient_documents, patient_questions, patient_info_data

            

def collate_emrQA(medication_data:str='data/emrQA/clean/medication-SQUAD.json',relation_data='data/emrQA/clean/relation-SQUAD.json', train_frac:float=0.85):
    """convert medication and relation data into form that will be useful for DPR training
    """

    with open(medication_data, "r") as f:
        medication_json = json.load(f)

    with open(relation_data, "r") as f:
        relation_json = json.load(f)

    
    questions = []
    answers = []
    for data_source in [medication_json, relation_json]:
        for record in data_source['data']:
            for paragraph in record['paragraphs']:
                answer_start = -1
                for question in paragraph['qas']:
                    if question['answers'][0]['answer_start']!=answer_start:
                        questions.append(question['question'])
                        answers.append(question['answers'][0]['text'])
                        answer_start=question['answers'][0]['answer_start']

    # shuffle the data
    combined = list(zip(questions, answers))
    random.shuffle(combined)
    questions, answers = zip(*combined)

    # split the data into training and validation 
    train_end = int(train_frac*len(questions))

    questions_train = questions[:train_end]
    answers_train = answers[:train_end]

    questions_val = questions[train_end:]
    answers_val = answers[train_end:]

    return questions_train, answers_train, questions_val, answers_val

    
                

