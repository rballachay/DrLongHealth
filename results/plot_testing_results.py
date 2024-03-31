from pathlib import Path
import json
import re 
import pandas as pd
import seaborn as sns

def remove_pattern(string):
    pattern = r" The correct answer is [A-E]:"
    string = re.search(pattern, string)

    if string:
        return string.group(0)[-2]
    else:
        return 'F'
    
def check_passage_num(string):
    pattern = r"-\d{2}pass$"
    string = re.search(pattern, string)
    if string:
        return int(string.group(0)[1:3])
    else:
        return "Full Doc"

def parse_model_name(string):
    bits = string.split('-')
    return'-'.join(bits[:3])

def parse_tasks_jsons(data_path_root='results', ground_truth='data/LongHealth/data/benchmark_v5.json'):
    """
    task 1 is the problem of asking questions where the problem 
    may be positively answered according to the data in the document.
    """

    with open(ground_truth, "r") as f:
        benchmark = json.load(f)

    results_df = {'accuracy':[],'model':[],'task':[],'# passages':[]}

    for task in [1,2,3]:
        data_path=f"{data_path_root}/task_{task}"
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


            results_df['model'].append(parse_model_name(str(path.stem)))
            results_df['accuracy'].append(100*true_total/total_count)
            results_df['# passages'].append(check_passage_num(str(path.stem)))
            results_df['task'].append(task)
    
    return pd.DataFrame(results_df)

            
def plot_results(results_df):
    results_df = results_df.rename(
        columns={"model": "Chat Model", 
                 "# passages": "Num. passages",
                 "task":"Task Number",
                 "accuracy":"Accuracy (%)"
                 })
        
    plot = sns.catplot(
    data=results_df, x="Chat Model", y="Accuracy (%)", col="Task Number",
    kind="bar",hue="Num. passages",palette="hls"
    )
    return plot.fig

def plot_prompt_lengths(data_path_root='results'):
    results_final = []
    for task in [1,2]:
        data_path=f"{data_path_root}/task_{task}/length_prompts.csv"
        results = pd.read_csv(data_path)

        results['task'] = task
        results_final.append(results)

    results_final = pd.concat(results_final)
    results_final['n_passages'] = results_final['n_passages'].replace({10000:"Full Document"})
    results_final = results_final.rename(
        columns={"n_passages": "Number of passages", 
                 "l_prompt": "Length Prompt (# chars)",
                 "task":"Task Number"
                 })

    plot = sns.catplot(
    data=results_final, x="Number of passages", y="Length Prompt (# chars)", 
    col="Task Number", kind="bar",palette="hls"
    )
    return plot.fig
                
if __name__=="__main__":
    sns.set_theme()
    results_df = parse_tasks_jsons('results')
    fig = plot_results(results_df)
    fig.savefig('results/accuracy_by_model.png')

    fig = plot_prompt_lengths()
    fig.savefig('results/prompt_lengths.png')
