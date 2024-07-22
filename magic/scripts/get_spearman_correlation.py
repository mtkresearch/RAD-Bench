#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import seaborn as sns

# As of 240721
lmsys_results = {
    "GPT-4o": {"hard-all": 1288, "hard-en": 1280, "long-query": 1307},
    "GPT-4-Turbo": {"hard-all": 1254, "hard-en": 1247, "long-query": 1268},
    "GPT-35-Turbo": {"hard-all": 1124, "hard-en": 1117, "long-query": 1125},
    "Gemma-2-27B-it": {"hard-all": 1199, "hard-en": 1185, "long-query": 1227},
    "Gemma-2-9B-it": {"hard-all": 1169, "hard-en": 1155, "long-query": 1198},
    "Llama3-70B": {"hard-all": 1197, "hard-en": 1212, "long-query": 1185},
    "Llama3-8B": {"hard-all": 1137, "hard-en": 1144, "long-query": 1134},
    "Mixtral-8x22B": {"hard-all": 1147, "hard-en": 1142, "long-query": 1154},
    "Deepseek-v2": {"hard-all": 1241, "hard-en": 1233, "long-query": 1255},
    "Yi-Large": {"hard-all": 1209, "hard-en": 1198, "long-query": 1224},
}

_result_csv_path = "evaluated_models.csv"

result_df = pd.read_csv(_result_csv_path)

result_df['rs_avg'] = result_df[['academic', 'news', 'education']].mean(axis=1)
result_df['rr_avg'] = result_df[['finance', 'customer', 'travel']].mean(axis=1)
result_df = result_df.set_index("model_name")
print(result_df)

use_models = list(set(lmsys_results.keys()) & set(result_df.index))
print(use_models)

# %%
df_lmsys = pd.DataFrame.from_dict(lmsys_results).transpose()
print(df_lmsys)
# %%

df_final = pd.concat([result_df.loc[use_models], df_lmsys.loc[use_models]], axis=1)
print(df_final)

#%%

# x_keys = ["average", "rs_avg", "rr_avg"]
# y_keys = ["hard-all", "hard-en", "long-query"]

# n_rows = len(y_keys)
# n_cols = len(x_keys)

    
# Assuming df_final is already defined and contains the data
x_key = "hard-all"
y_key = "average"
x_vals = df_final[x_key]
y_vals = df_final[y_key]

# Calculate Spearman correlation
correlation, p_value = stats.spearmanr(x_vals, y_vals)

print(f"Spearman correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Create a scatter plot with improvements
plt.figure(figsize=(12, 8))
plt.scatter(x_vals, y_vals, color='royalblue', s=100, edgecolors='k', alpha=0.7)

# Annotate each point with the model name, with offset to avoid overlap
for i, model in enumerate(df_final.index):
    plt.annotate(model, (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(5,-5), ha='center', fontsize=10, color='darkred')

plt.xlabel(x_key, fontsize=14)
plt.ylabel(y_key, fontsize=14)
plt.title(f'Scatter plot: {x_key} vs {y_key}', fontsize=16)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.show()


#%%
x_key = "average"
y_key = "hard-all"
x_label = "RAD-Bench, score"
y_label = "Chatbot Arena Rating\n(Hard prompts), score"
title = 'Scatter Plot: Hard-All vs Average'
# fontfamily = "Times New Roman"
fontfamily = "Sans-serif" # "Serif" # "Helvetica"
fontsize = 28
fontweight="normal"

# Set the style for a clean, professional look
plt.style.use('seaborn-v0_8')
sns.set_palette("deep")
plt.rcParams['font.family'] = fontfamily  # You can change 'Arial' to your preferred font

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the scatter points
sns.scatterplot(x=x_key, y=y_key, data=df_final, s=120, ax=ax)

# Add labels for each point
for idx, row in df_final.iterrows():
    ax.annotate(idx, (row[x_key], row[y_key]), 
                xytext=(5, 5), textcoords='offset points', 
                fontsize=24, fontweight=fontweight, fontfamily=fontfamily)

# Adjust tick label font size
ax.tick_params(axis='both', which='major', labelsize=20)

# Customize the plot
# ax.set_title(title, fontsize=16, fontweight=fontweight, fontfamily=fontfamily)
ax.set_xlabel(x_label, fontsize=fontsize, fontfamily=fontfamily)
ax.set_ylabel(y_label, fontsize=fontsize, fontfamily=fontfamily)

# Add gridlines
ax.grid(True, linestyle='--', alpha=0.99)

# Adjust the plot range
ax.set_ylim(1110, 1310)
ax.set_xlim(6.2, 9.2)

# Add a subtle background color
ax.set_facecolor('#f0f0f0')

# Calculate and display correlation
corr, _ = stats.pearsonr(df_final[x_key], df_final[y_key])
ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes, 
        fontsize=fontsize, verticalalignment='top', fontweight=fontweight, fontfamily=fontfamily)

# Adjust layout and display
plt.tight_layout()
plt.show()
fig.savefig("../../asset/spearman_correlation.png", dpi=150)

# %%
# Define the keys for x and y axes
y_keys = ["average", "rs_avg", "rr_avg"]
x_keys = ["hard-all", "hard-en", "long-query"]

# Set up the subplots
fig, axes = plt.subplots(len(y_keys), len(x_keys), figsize=(30, 26))
fig.suptitle("Correlation Plots for Various Metrics", fontsize=fontsize)

# Generate all combinations of x_keys and y_keys
combinations = list(product(x_keys, y_keys))

# Plot each combination
for i, ((x_key, y_key), ax) in enumerate(zip(combinations, axes.flatten())):
    x = df_final[x_key]
    y = df_final[y_key]
    
    # Calculate Spearman correlation
    corr, p_value = stats.spearmanr(x, y)
    
    # Plot scatter
    ax.scatter(x, y)
    
    # Add labels for each point
    for j, model in enumerate(df_final.index):
        ax.annotate(model, (x[j], y[j]), xytext=(5, 5), textcoords='offset points')
    
    # Set labels and title
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f'{x_key} vs {y_key}')
    
    # Add correlation and p-value info
    ax.text(0.05, 0.95, f'Corr: {corr:.2f}\np-value: {p_value:.2f}', 
            transform=ax.transAxes, verticalalignment='top')

# Adjust layout
plt.tight_layout()
plt.show()
# %%
