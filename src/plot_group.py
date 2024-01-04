import argparse

parser = argparse.ArgumentParser(description="Example script to demonstrate argparse usage.")

parser.add_argument("--mode", type=str, default='one_by_one')
parser.add_argument("--plot_from_pickle", action='store_true', default=False, help="A boolean flag, defaults to False")
parser.add_argument("--job_by_list", action='store_true', default=False, help="A boolean flag, defaults to False")
parser.add_argument("--legend_by_list", action='store_true', default=False, help="A boolean flag, defaults to False")
parser.add_argument("--use_log", action='store_true', default=False, help="A boolean flag, defaults to False")
parser.add_argument("--output", type=str, default='')
# parser.add_argument("--integers", nargs='+', type=int, help="A list of integers")

args = parser.parse_args()

from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
import pickle
import numpy as np

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

start_id = 53
end_id = 62
filter_jobid = []
legend_names = ['0.01', '0.05', '0.1', '0.5', '1']

# target_jobid = [f"jobid{job_id}" for job_id in range(start_id, end_id+1)]
# emb6 ['1', '2', '3', '6'] [37, 36, 38, 34]
# emb64 ['1', '8', '16', '64'] [39, 40, 51, 41]
# emb128 ['1', '8', '16', '64', '128'] [42, 43, 108, 44, 45]
# emb256 ['1', '8', '16', '64', '128', '256'] [110, 46, 112, 47, 48, 49]
# fixed head each emb 8 ['1', '8', '16', '32'], [113, 40, 108, 109]
# head 1 ['0.01', '0.05', '0.1', '0.5', '1'], [114, 115, 116, 119, 120]
# head 16 ['0.01', '0.05', '0.1', '0.5', '1'], [121, 122, 123, 124, 125]
if args.job_by_list:
    target_jobid = [f"jobid{job_id}" for job_id in [114, 115, 116, 119, 120]] # emb64: [39, 40, 51, 41]
else:
    target_jobid = [f"jobid{job_id}" for job_id in range(start_id, end_id+1)]

# for task_id in sub_tasks:
#     if task_id.split('_')[0] in target_jobid:
#         filter_jobid.append(task_id)

for job_id in target_jobid:
    for task_id in sub_tasks:
        # print(f"jobid{job_id}_", task_id)
        if f"{job_id}_" in task_id:
            filter_jobid.append(task_id)
            break

print(filter_jobid)

# import pdb; pdb.set_trace()

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"
task = 'linear_regression'
blue_metrics = {}
blue_models = []

for local_id, _filtered_runid in enumerate(filter_jobid):

    if args.plot_from_pickle:
        continue
    
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

    # import pdb ; pdb.set_trace()

    if args.mode == "one_by_one":

        basic_plot(metrics["standard"], models=models, use_log=args.use_log)
        output_file_path = f"./plot_results/{run_id}"

        # plt.figure(figsize=(8, 6))  # Adjusting the width and height of the figure

        plt.yticks(fontsize=16)
        plt.ylabel('squared error', fontsize=16)
        plt.xticks(fontsize=16)
        plt.xlabel('in-context examples', fontsize=12)  
        for line in plt.gca().lines:
            line.set_linewidth(2)
        plt.legend(fontsize=12, loc=3)  
        plt.tight_layout()  
        plt.savefig(f"{output_file_path}.png") 

    elif args.mode in ["plot_blue", "plot_blue_ave"]:
        if args.legend_by_list:
            blue_models.append(legend_names[local_id])
            blue_metrics[legend_names[local_id]] = metrics["standard"][models[0]]
        else:
            blue_models.append(str(local_id))
            blue_metrics[str(local_id)] = metrics["standard"][models[0]]

        # import pdb ; pdb.set_trace()
        # dict_keys(['mean', 'std', 'bootstrap_low', 'bootstrap_high'])

    else:
        raise("Wrong mode")

if args.mode == "plot_blue":
    # blue_models = ['1', '8', '16', '64']
    basic_plot(blue_metrics, models=blue_models, use_log=args.use_log)
    if args.output == '':
        output_file_path = f"./plot_results/job{start_id}to{end_id}"
    else:
        output_file_path = f"./plot_results/{args.output}"

    plt.yticks(fontsize=16)
    plt.ylabel('squared error', fontsize=22)
    plt.xticks(fontsize=16)
    plt.xlabel('in-context examples', fontsize=22)
    for line in plt.gca().lines:
        line.set_linewidth(2)
    plt.legend(fontsize=16, loc=3)
    plt.tight_layout()
    plt.savefig(f"{output_file_path}.png")

    # plt.yticks(fontsize=20)
    # plt.ylabel('squared error', fontsize=34)
    # plt.xticks(fontsize=20)
    # plt.xlabel('in-context examples', fontsize=34)
    # for line in plt.gca().lines:
    #     line.set_linewidth(2)
    # plt.legend(fontsize=20, loc=3)
    # plt.tight_layout()
    # plt.savefig(f"{output_file_path}.png")

elif args.mode == "plot_blue_ave":

    if args.plot_from_pickle:
        with open(f"./plot_results/job53to62.pkl", 'rb') as f:
            blue_metrics1 = pickle.load(f)

        with open(f"./plot_results/job63to72.pkl", 'rb') as f:
            blue_metrics2 = pickle.load(f)

        blue_models = ['ave_emb6', 'ave_noreadin']
        ave_metrics = {}
        m_keys = ['mean', 'std', 'bootstrap_low', 'bootstrap_high']

        temp = {}
        for key in m_keys:
            sum = 0
            for i in range(10):
                sum += np.array(blue_metrics1[str(i)][key])
            temp[key] = sum / 10
        ave_metrics['ave_emb6'] = temp

        temp = {}
        for key in m_keys:
            sum = 0
            for i in range(10):
                sum += np.array(blue_metrics2[str(i)][key])
            temp[key] = sum / 10
        ave_metrics['ave_noreadin'] = temp

            # import pdb ; pdb.set_trace()
    
    else:
        with open(f"./plot_results/job{start_id}to{end_id}.pkl", 'wb') as f:
            pickle.dump(blue_metrics, f)

    basic_plot(ave_metrics, models=blue_models, use_log=args.use_log)
    output_file_path = f"./plot_results/job{53}to{72}_ave"

    # plt.figure(figsize=(8, 6))  # Adjusting the width and height of the figure

    plt.yticks(fontsize=24)
    plt.ylabel('squared error', fontsize=24)
    plt.xticks(fontsize=24)
    plt.xlabel('in-context examples', fontsize=24)
    for line in plt.gca().lines:
        line.set_linewidth(2)
    plt.legend(fontsize=24, loc=3)
    plt.tight_layout()
    plt.savefig(f"{output_file_path}.png")

