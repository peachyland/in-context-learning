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

def list_subfolders(directory):
    """
    List all subfolders in the specified directory.
    
    :param directory: Path to the directory to search for subfolders.
    :return: A list containing the names of the subfolders.
    """
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

# Example usage:
# subfolders = list_subfolders('/path/to/directory')
# print(subfolders)
sub_tasks = list_subfolders("/egr/research-dselab/renjie3/renjie/LLM/multi_head/in-context-learning/models/linear_regression")

start_id = 63
end_id = 72
filter_jobid = []

target_jobid = [f"jobid{job_id}" for job_id in range(start_id, end_id+1)]

for task_id in sub_tasks:
    if task_id.split('_')[0] in target_jobid:
        filter_jobid.append(task_id)

print(filter_jobid)

# import pdb; pdb.set_trace()

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"
task = 'linear_regression'

for _filtered_runid in filter_jobid:
    
    run_id = _filtered_runid  # if you train more models, replace with the run_id from the table above

    def valid_row(r):
        return r.task == task and r.run_id == run_id #and r.run_id2 == run_id2

    df = read_run_dir(run_dir, single_task=True, job_id=run_id)
    print(df)

    run_path = os.path.join(run_dir, task, run_id)
    recompute_metrics = False

    if recompute_metrics:
        get_run_metrics(run_path)  # these are normally precomputed at the end of training

    metrics = collect_results(run_dir, df, valid_row=valid_row)

    models = relevant_model_names[task]
    prefix_split = run_id.split('_')
    n_head = prefix_split[3].replace('head', '')
    if n_head == '8':
        models[0] = "Transformer-1-layer"
    else:
        models[0] = f"Transformer-1-layer-{n_head}-head"
    basic_plot(metrics["standard"], models=models)
    output_file_path = f"./plot_results/{run_id}"

    # plt.figure(figsize=(12, 6))  # Adjusting the width and height of the figure

    plt.yticks(fontsize=8)
    plt.ylabel('squared error', fontsize=8)  
    plt.xticks(fontsize=8)
    plt.xlabel('in-context examples', fontsize=6)  
    for line in plt.gca().lines:
        line.set_linewidth(1)
    plt.legend(fontsize=6, loc=3)  
    plt.tight_layout()  
    plt.savefig(f"{output_file_path}.png") 








