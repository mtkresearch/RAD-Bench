# %%
import json
import pprint
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import pi
import numpy as np
import pandas as pd
from cycler import cycler

_CUR_DIR=os.path.dirname(os.path.abspath(__file__))

_model_name_map = {
    "gpt-4o": "GPT-4o",
    "gpt-35-turbo-16k-4k": "GPT-3.5-Turbo",
    "llama3-70b-instruct": "Llama3-70B",
    "mixtral-8x22b-instruct": "Mixtral-8x22B", 
    "breexe-8x7b-instruct-v01": "BreeXe-8x7B",
    "llama3-8b-instruct": "Llama3-8B",
    "breeze-7B-32k-instruct-v10": "Breeze-7B"
}

_key_to_labels = {
    "news": "News TLDR",
    "academic": "Academic",
    "education": "Education",
    "customer": "Customer\nSupport",
    "finance": "Finance",
    "travel": "Travel\nPlanning"
}

# %%
def plot_radar_chart(averages, title, 
                     title_size=20, 
                     tick_label_size=24, 
                     legend_font_size=20):
    # Combine all tasks from both KI and KR aspects
    labels = ['academic', 'news', 'education', 'finance', 'customer', 'travel']
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    print(len(angles), angles)

    fig, ax = plt.subplots(figsize=(12, 10), 
                           subplot_kw=dict(polar=True))


    # New color palette (subset of Tableau 20)
    colors = [
        '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f',
        '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'
    ]
    # Ensure we have enough colors
    if len(averages) > len(colors):
        colors = colors * (len(averages) // len(colors) + 1)
        
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


    for model, tasks in averages.items():
        # model_name = _model_name_map[model]
        model_name = model
        values = [tasks[l] for l in labels]
        
        values += values[:1]
        print(values)
        ax.plot(angles, values, label=model_name, linewidth=3)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(0)
    tick_labels = [_key_to_labels[l] for l in labels]
    ax.set_xticks(angles[:-1], tick_labels, size=tick_label_size)
    ax.set_yticks([3, 4, 5, 6, 7, 8, 9], ["3", "4", "5", "6", "7", "8", "9"], color="grey", size=tick_label_size)
    # for label in ax.get_xticklabels():
    #     x, y = label.get_position()
    #     label.set_position((x*1.1, y*1.1))
        # Move tick labels further away from the circle
    ax.tick_params(axis='x', pad=40)  # Increase padding for x-axis (radial) labels
    
    # Adjust position of individual labels if needed
    for label, angle in zip(ax.get_xticklabels(), angles):
        x, y = label.get_position()
        lab = ax.text(x, y, label.get_text(), transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va())
        lab.set_position((x, y))
        label.set_visible(False)

    ax.set_ylim(3.2, 9.5)

    plt.subplots_adjust(bottom=0.25)

    # plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.2), fontsize=legend_font_size)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=3, fontsize=legend_font_size)
    # plt.title(title, fontsize=title_size)
    # plt.show()
    fig.savefig(f"{_CUR_DIR}/../../asset/model_radar.png")
#%%
df_final = pd.read_csv("evaluated_models.csv")
del df_final['average']
df_final = df_final.set_index("model_name").transpose()
print(df_final)
#%%
plot_radar_chart(df_final.to_dict(), "")
# %%
