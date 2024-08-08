#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import seaborn as sns
from scipy.stats import spearmanr

_EXCLUDE_MODELS = ["Gemma-2-27B", "Gemma-2-9B"]

def cal_spearman(xvals, yvals, y_lower = None, y_upper = None):
    if y_lower is None or y_upper is None:
        return spearmanr(xvals, yvals)[0], None, None
    
    
    # Number of bootstrap samples
    bootstraps = 1000
    correlations = np.zeros(bootstraps)

    np.random.seed(0)  # For reproducibility

    for i in range(bootstraps):
        # Generate samples within the confidence intervals for y
        y_sample = np.random.uniform(y_lower, y_upper)
        
        # Calculate Spearman correlation for this bootstrap sample
        correlations[i] = spearmanr(xvals, y_sample)[0]

    # Calculate the average correlation and confidence interval
    mean_correlation = np.mean(correlations)
    lower_ci = np.percentile(correlations, 2.5)
    upper_ci = np.percentile(correlations, 97.5)
    
    return mean_correlation, lower_ci, upper_ci

# # As of 240721
# lmsys_results = {
#     "GPT-4o": {"hard-all": 1288, "hard-en": 1280, "long-query": 1307},
#     "GPT-4-Turbo": {"hard-all": 1254, "hard-en": 1247, "long-query": 1268},
#     "GPT-35-Turbo": {"hard-all": 1124, "hard-en": 1117, "long-query": 1125},
#     "Gemma-2-27B": {"hard-all": 1199, "hard-en": 1185, "long-query": 1227},
#     "Gemma-2-9B": {"hard-all": 1169, "hard-en": 1155, "long-query": 1198},
#     "Llama3-70B": {"hard-all": 1197, "hard-en": 1212, "long-query": 1185},
#     "Llama3-8B": {"hard-all": 1137, "hard-en": 1144, "long-query": 1134},
#     "Mixtral-8x22B": {"hard-all": 1147, "hard-en": 1142, "long-query": 1154},
#     "Deepseek-v2": {"hard-all": 1241, "hard-en": 1233, "long-query": 1255},
#     "Yi-Large": {"hard-all": 1209, "hard-en": 1198, "long-query": 1224},
# }
lmsys_results = pd.read_csv("lmsys_results_240807.csv")
# lmsys_results = lmsys_results.set_index("model_name").to_dict()
print(lmsys_results)
lmsys_results = lmsys_results.set_index("model_name").transpose().to_dict()
print(lmsys_results)

#%%
_result_csv_path = "evaluated_models.csv"

result_df = pd.read_csv(_result_csv_path)

result_df['rs_avg'] = result_df[['academic', 'news', 'education']].mean(axis=1)
result_df['rr_avg'] = result_df[['finance', 'customer', 'travel']].mean(axis=1)
result_df = result_df.set_index("model_name")
print(result_df)

use_models = list(set(lmsys_results.keys()) & set(result_df.index))
for em in _EXCLUDE_MODELS:
    use_models.remove(em)
print(use_models)

# %%
df_lmsys = pd.DataFrame.from_dict(lmsys_results).transpose()
print(df_lmsys)

lmsys_boards = df_lmsys.keys()
print(lmsys_boards)

df_lmsys_processed = df_lmsys.copy()
for b in lmsys_boards:
    vals = df_lmsys[b].apply(lambda x: float(x.split("(")[0]))
    up_bounds = df_lmsys[b].apply(lambda x: float(x.split("(")[-1].split(',')[0]))
    low_bounds = df_lmsys[b].apply(lambda x: float(x.split(",")[-1].split(')')[0]))
    # print(vals, up_bounds, low_bounds)
    
    df_lmsys_processed[b] = vals
    df_lmsys_processed[f"{b}_up"] = up_bounds
    df_lmsys_processed[f"{b}_low"] = low_bounds

print(df_lmsys_processed)
# print(df_lmsys['all'].apply(lambda x: int(x.split("(")[0])))
# print(df_lmsys['all'].apply(lambda x: int(x.split("(")[-1].split(',')[0])))
# %%

df_final = pd.concat([result_df.loc[use_models], df_lmsys_processed.loc[use_models]], axis=1)
print(df_final)
#%%

# print(df_final["all"].to_numpy())
#%%

# x_keys = ["average", "rs_avg", "rr_avg"]
# y_keys = ["hard-all", "hard-en", "long-query"]

# n_rows = len(y_keys)
# n_cols = len(x_keys)

    
# Assuming df_final is already defined and contains the data
y_key = "hard-all"
x_key = "average"
x_vals = df_final[x_key]
y_vals = df_final[y_key]
y_upper = df_final[f"{y_key}_up"]
y_lower = df_final[f"{y_key}_low"]

# Calculate Spearman correlation
# correlation, p_value = stats.spearmanr(x_vals, y_vals)
correlation, ci_up, ci_low = cal_spearman(x_vals, y_vals, y_upper, y_lower)

print(f"Spearman correlation coefficient: {correlation:.4f} (+{ci_up}, -{ci_low})")
# print(f"P-value: {p_value:.4f}")

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
def plot_corr_plot(chat_arena_category="hard-all"):
    x_key = "average"
    y_key = chat_arena_category
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
    # Add error bars
    # Add asymmetric error bars
    yerr = [df_final[f'{y_key}_low'], df_final[f'{y_key}_up']]
    ax.errorbar(df_final[x_key], df_final[y_key], yerr=yerr, fmt='none', 
                ecolor='gray', elinewidth=2, capsize=4)

    # Add labels for each point
    for idx, row in df_final.iterrows():
        xytext = (5,5)
        if idx in {"Llama3.1-70B"}:
            xytext = (-150, -25)
            # xytext = (-150, 5)
        elif idx in {"Llama3.1-405B"}:
            xytext = (5,-25)
        
        ax.annotate(idx, (row[x_key], row[y_key]), 
                    xytext=xytext, textcoords='offset points', 
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
    ax.set_ylim(1090, 1310)
    ax.set_xlim(6.2, 9.2)

    # Add a subtle background color
    ax.set_facecolor('#f0f0f0')

    # Calculate and display correlation
    # corr, ci_up, ci_low = cal_spearman(df_final[x_key], df_final[y_key], df_final[f"{y_key}_low"], df_final[f"{y_key}_up"])
    # ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes, 
    #         fontsize=fontsize, verticalalignment='top', fontweight=fontweight, fontfamily=fontfamily)
    corr, _ = stats.pearsonr(df_final[x_key], df_final[y_key])
    ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes, 
            fontsize=fontsize, verticalalignment='top', fontweight=fontweight, fontfamily=fontfamily)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    if chat_arena_category == "hard-all":
        fig.savefig(f"../../asset/spearman_correlation.png", dpi=150)
    fig.savefig(f"../../asset/spearman_correlation_{chat_arena_category}.png", dpi=150)

plot_corr_plot("all")
plot_corr_plot("multi-turn")
plot_corr_plot("hard-all")
plot_corr_plot("hard-en")
plot_corr_plot("long-query")

# %%
# Define the keys for x and y axes
c_keys = ["average", "rs_avg", "rr_avg"]
r_keys = ["all", "multi-turn", "hard-all", "hard-en", "long-query"]

# Set up the subplots
fig, axes = plt.subplots(len(r_keys), len(c_keys), figsize=(10 * len(c_keys), 2 + 8 * len(r_keys)))
fig.suptitle("Correlation Plots for Various Metrics", fontsize=fontsize)

# # Generate all combinations of x_keys and y_keys
combinations = list(product(r_keys, c_keys))

# # Plot each combination
for i, ((r_key, c_key), ax) in enumerate(zip(combinations, axes.flatten())):
# for r, y_key in enumerate(y_keys):
#     for c, x_key in enumerate(x_keys):
        # ax = axes[r][c]
    y = df_final[r_key]
    x = df_final[c_key]
    y_lower = df_final[f"{r_key}_low"]
    y_upper = df_final[f"{r_key}_up"]
    # Calculate Spearman correlation
    corr, p_value = stats.spearmanr(x, y)
    # corr, ci_up, ci_low = cal_spearman(x, y, y_lower, y_upper)
    
    # Plot scatter
    ax.scatter(x, y)
    
    # Add labels for each point
    for j, model in enumerate(df_final.index):
        if model == "Llama3.1-70B":
            ax.annotate(model, (x[j], y[j]), xytext=(-10, 5), textcoords='offset points')
        else:
            ax.annotate(model, (x[j], y[j]), xytext=(5, 5), textcoords='offset points')
    
    # Set labels and title
    ax.set_xlabel(c_key)
    ax.set_ylabel(r_key)
    ax.set_title(f'{r_key} vs {c_key}')
    
    # Add correlation and p-value info
    # ax.text(0.05, 0.95, f'Corr: {corr:.2f} (+{ci_up:.2f}, -{ci_low:.2f})', 
    #         transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.95, f'Corr: {corr:.2f}\np-value: {p_value:.2f}', 
            transform=ax.transAxes, verticalalignment='top')

# Adjust layout
plt.tight_layout()
plt.show()
# %%
