# %%
import json
import pprint
import os
_CUR_DIR=os.path.dirname(os.path.abspath(__file__))


_model_name_map = {
    "gpt-4o": "GPT-4o",
    "gpt-35-turbo-16k-4k": "GPT-35-Turbo",
    "llama3-70b-instruct": "Llama3-70B",
    "mixtral-8x22b-instruct": "Mixtral-8x22B", 
    "breexe-8x7b-instruct-v01": "BreeXe-8x7B",
    "llama3-8b-instruct": "Llama3-8B",
    "breeze-7B-32k-instruct-v10": "Breeze-7B"
}

def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            result = json.loads(line.strip())
            results.append(result)
    return results

def get_task_turn_avg_score(data, task):
    """
    return a list of average scores per turn
    """
    task_entries = [entry for entry in data if task in entry["question_id"]]
    avg_score_turns = []
    for turn in range(1,4):
        task_turn_entries = [entry for entry in task_entries if entry['turn'] == turn]
        avg = sum([entry["score"] for entry in task_turn_entries]) / len(task_turn_entries)
        avg_score_turns.append(avg)
    return avg_score_turns

def get_avg_score(data):
    ans = {}
    academic_scores = get_task_turn_avg_score(data, "academic")
    news_scores = get_task_turn_avg_score(data, "news")
    education_scores = get_task_turn_avg_score(data, "education")
    finance_scores = get_task_turn_avg_score(data, "finance")
    customer_scores = get_task_turn_avg_score(data, "customer")
    travel_scores = get_task_turn_avg_score(data, "travel")
    ans["CS"] = {
        "academic": academic_scores,
        "news": news_scores,
        "education": education_scores,
    }
    ans["CR"] = {
        "finance": finance_scores,
        "customer": customer_scores,
        "travel": travel_scores,
    }
    return ans

def filter_questions_with_model(data, model_name):
    return [q for q in data if q["model"] == model_name]

def get_ans(judge_model, model_list) -> dict:
    ans = {}
    filename = f"{_CUR_DIR}/../data/model_judgment/gpt-4o.jsonl"
    judge_data = read_jsonl_file(filename)
    for model in model_list:
        model_data = filter_questions_with_model(judge_data, model)
        model_avg_score = get_avg_score(model_data)
        ans[model] = model_avg_score
    return ans

# %%
# Function to calculate average scores
import numpy as np
def calculate_averages(data):
    averages = {}
    for model, aspects in data.items():
        averages[model] = {}
        for aspect, tasks in aspects.items():
            averages[model][aspect] = {}
            for task, scores in tasks.items():
                averages[model][aspect][task] = np.mean(scores)
    return averages

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import pi

def plot_radar_chart(averages, title):
    # Combine all tasks from both KI and KR aspects
    labels = ['academic', 'news', 'education', 'finance', 'customer', 'travel']
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

    for model, aspects in averages.items():
        model_name = _model_name_map[model]
        values = []
        for task in labels:
            for aspect in ['KI', 'KR']:
                if task in aspects[aspect]:
                    values.append(aspects[aspect][task])
                    break
        values += values[:1]
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(0)
    plt.xticks(angles[:-1], labels)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=7)
    for label in ax.get_xticklabels():
        x, y = label.get_position()
        label.set_position((x*1.1, y*1.1))

    plt.ylim(0, 10)

    # ax.legend()
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.2))
    plt.title(title)
    plt.show()
    fig.savefig(f"{_CUR_DIR}/../../asset/model_radar.png")

# Function to plot bar charts for each task
def plot_bar_charts(averages):
    tasks = ['academic', 'news', 'education', 'finance', 'customer', 'travel']
    
    for task in tasks:
        models = []
        scores = []
        
        for model, aspects in averages.items():
            models.append(model)
            for aspect in ['KI', 'KR']:
                if task in aspects[aspect]:
                    scores.append(aspects[aspect][task])
                    break
        
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(tasks)))
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, scores, color=colors)
        plt.xlabel('Models')
        plt.ylabel('Average Score')
        plt.title(f'Average Scores for {task.capitalize()} Task')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.show()

# %%
# sort by activation parameter size
all_models = ["gpt-4o", "gpt-35-turbo-16k-4k", "llama3-70b-instruct", "mixtral-8x22b-instruct", "breexe-8x7b-instruct-v01", "llama3-8b-instruct", "breeze-7B-32k-instruct-v10"]

all_ans = get_ans("gpt-4o", all_models)

all_averages = calculate_averages(all_ans)

plot_radar_chart(all_averages, "Performance of each model")



# %%
# Plot bar charts for each task
plot_bar_charts(all_averages)

# %%
import pandas as pd

rows = ["gpt-4o", "gpt-35-turbo-16k-4k", "llama3-70b-instruct", "mixtral-8x22b-instruct", "breexe-8x7b-instruct-v01", "llama3-8b-instruct", "breeze-7B-32k-instruct-v10"]
rows = [_model_name_map[r] for r in rows]
columns = ['academic', 'news', 'education', 'finance', 'customer', 'travel']

df = pd.DataFrame(index = rows, columns=columns)

for task in columns:
    scores = []
    
    for model, aspects in all_averages.items():
        for aspect in ['KI', 'KR']:
            if task in aspects[aspect]:
                scores.append(aspects[aspect][task])
                break
    df[task] = scores


df['average'] = df.mean(axis=1)
average_row = df.mean(axis=0)
df.loc['average'] = average_row
print(df)


df.to_markdown("result.md")
df.to_latex("result.tex", float_format="%.2f")


# %%
import matplotlib.pyplot as plt

# Function to plot line plots for each task
def plot_line_plots(data):
    tasks = ['academic', 'news', 'education', 'finance', 'customer']
    turns = ['turn1', 'turn2', 'turn3']
    turn_indices = range(1, 4)
    
    for task in tasks:
        plt.figure(figsize=(10, 6))
        
        for model, aspects in data.items():
            for aspect in ['KI', 'KR']:
                if task in aspects[aspect]:
                    scores = aspects[aspect][task]
                    plt.plot(turn_indices, scores, marker='o', label=model, markersize = 3)
                    break
        
        plt.xlabel('Turn')
        plt.ylabel('Score')
        plt.title(f'Scores for {task.capitalize()} Task')
        plt.xticks(turn_indices, turns)
        plt.ylim(3.5, 10)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

all_ans = get_ans("gpt-4o", all_models)

plot_line_plots(all_ans)

# %%
import matplotlib.pyplot as plt

# Function to plot line plots for each task
def plot_line_plots(data):
    task = "travel"
    turns = ['turn1', 'turn2', 'turn3']
    turn_indices = range(1, 4)
    
   
    plt.figure(figsize=(10, 6))
    
    for model, aspects in data.items():
        for aspect in ['KI', 'KR']:
            if task in aspects[aspect]:
                scores = aspects[aspect][task]
                plt.plot(turn_indices, scores, marker='o', label=model, markersize = 3)
                break
    
    plt.xlabel('Turn')
    plt.ylabel('Score')
    plt.title(f'Scores for {task.capitalize()} Task')
    plt.xticks(turn_indices, turns)
    plt.ylim(2, 10)
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

all_ans = get_ans("gpt-4o", all_models)

plot_line_plots(all_ans)

# %%



