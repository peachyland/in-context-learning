from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"

df = read_run_dir(run_dir)
print(df)
task = 'linear_regression'
run_id = "20de6bf1-41b3-4206-a0be-091677856df1"  # if you train more models, replace with the run_id from the table above

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False

if recompute_metrics:
    get_run_metrics(run_path)  # these are normally precomputed at the end of training


def valid_row(r):
    return r.task == task and r.run_id == run_id #and r.run_id2 == run_id2

metrics = collect_results(run_dir, df, valid_row=valid_row)

models = relevant_model_names[task]
basic_plot(metrics["standard"], models=models)
output_file_path = "666_2"

# plt.figure(figsize=(12, 6))  # Adjusting the width and height of the figure

plt.yticks(fontsize=8)
plt.ylabel('squared error', fontsize=8)  
plt.xticks(fontsize=8)
plt.xlabel('in-context examples', fontsize=6)  
for line in plt.gca().lines:
    line.set_linewidth(1)
plt.legend(fontsize=6, loc=3)  
plt.tight_layout()  
plt.savefig(output_file_path) 








