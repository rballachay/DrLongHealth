import nltk
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


def break_text_into_passages(text, max_passage_length=256):
    """Break a text into passages of a certain length, each containing
    complete sentences.
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