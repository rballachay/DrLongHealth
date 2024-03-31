from pathlib import Path
import json
import re 

def remove_pattern(string):
    pattern = r" The correct answer is [A-E]:"
    string = re.search(pattern, string)

    if string:
        return string.group(0)[-2]
    else:
        return 'F'

def plot_task(data_path='results/task_1', ground_truth='data/LongHealth/data/benchmark_v5.json'):
    """
    task 1 is the problem of asking questions where the problem 
    may be positively answered according to the data in the document.
    """

    with open(ground_truth, "r") as f:
        benchmark = json.load(f)

    for path in Path(data_path).glob('*.json'):
        with open(str(path), 'r') as js:
            data = json.load(js)

        true_total = 0
        total_count = 0
        for patient, results in data.items():
            for i,(question, response) in enumerate(results.items()):
                question = benchmark[patient]['questions'][i]
                
                for answer_key in filter(lambda x: re.match(r"^answer_[a-z]$",x), question.keys()):
                    if question[answer_key]==question['correct']:
                        break

                correct_answer = answer_key[-1].capitalize()
                match = remove_pattern(response['answer_1'])==correct_answer

                true_total+=match
                total_count+=1

        print(100*true_total/total_count)
                

if __name__=="__main__":
    plot_task('results/task_3')