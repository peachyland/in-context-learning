import argparse

parser = argparse.ArgumentParser(description="Example script to demonstrate argparse usage.")

parser.add_argument("--mode", type=str, default='one_by_one')
parser.add_argument("--plot_from_pickle", action='store_true', default=False, help="A boolean flag, defaults to False")
parser.add_argument("--manual_pkl", action='store_true', default=False, help="A boolean flag, defaults to False")
parser.add_argument("--job_by_list", action='store_true', default=False, help="A boolean flag, defaults to False")
parser.add_argument("--legend_by_list", action='store_true', default=False, help="A boolean flag, defaults to False")
parser.add_argument("--use_log", action='store_true', default=False, help="A boolean flag, defaults to False")
parser.add_argument("--output", type=str, default='')
# parser.add_argument("--manual_jobid", type=int, default=None)
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

# target_jobid = [f"jobid{job_id}" for job_id in range(start_id, end_id+1)]
# emb6 ['1', '2', '3', '6'] [37, 36, 38, 34]
# emb64 ['1', '8', '16', '64'] ['1', '8', '16', '64'] [39, 40, 51, 41]
# emb128 ['1', '8', '16', '64', '128'] [42, 43, 108, 44, 45]
# emb256 ['1', '8', '16', '64', '128', '256'] [110, 46, 112, 47, 48, 49]
# fixed head each emb 8 ['1', '8', '16', '32'], [113, 40, 108, 109]
# head 1 ['0.01', '0.05', '0.1', '0.5', '1'], [114, 115, 116, 119, 120] [139, 140, 141, 142, 143]
# head 16 ['0.01', '0.05', '0.1', '0.5', '1'], [121, 122, 123, 124, 125] [133, 132, 131, 130, 129]
# ['$\eta=0$', '$\eta=\theta_0$', '$\eta=-\theta_0$', '$\eta \perp \theta_0$'] 142
# local 126(variance=1) 127 128 51 144(variance=0.01) 145 146 52


# noiseLR
# no read in ['1', '2', '3', '6'] [198, 199, 200, 201]
# emb6 ['1', '2', '3', '6'] [194, 195, 196, 197]
# emb64 ['1', '8', '16', '64'] [202, 203, 204, 205]
# emb128 ['1', '8', '16', '64', '128'] [206, 207, 208, 210, 211]
# emb256 ['1', '8', '16', '64', '128', '256'] [212, 213, 214, 215, 216, 217]
# fixed head each emb 8 ['1', '8', '16', '32'], 
# head 1 ['0.01', '0.05', '0.1', '0.5', '1'], 
# head 16 ['0.01', '0.05', '0.1', '0.5', '1'], 


# correlated
# no read in ['1', '2', '3', '6'] [279, 278, 280, 281]
# emb6 ['1', '2', '3', '6'] [282, 283, 284, 285]
# emb64 ['1', '8', '16', '64'] [286, 287, 288, 289]
# emb128 ['1', '8', '16', '64', '128'] [290, 291, 292, 293, 294]
# emb256 ['1', '8', '16', '64', '128', '256'] [295, 296, 297, 298, 299, 300]
# fixed head each emb 8 ['1', '8', '16', '32'], 
# head 1 ['0.01', '0.05', '0.1', '0.5', '1'], 
# head 16 ['0.01', '0.05', '0.1', '0.5', '1'], 

legend_names = ['1', '8', '16', '64', '128', '256']

if args.job_by_list:
    target_jobid = [f"jobid{job_id}" for job_id in [295, 296, 297, 298, 299, 300]] # emb64: [39, 40, 51, 41]
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
ICL_results = []

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

    print(metrics["standard"][models[0]]['mean'][-1])
    # import pdb ; pdb.set_trace()
    ICL_results.append(metrics["standard"][models[0]]['mean'][-1])

    if args.mode == "one_by_one":

        basic_plot(metrics["standard"], models=models, use_log=args.use_log)
        output_file_path = f"./plot_results_new/{run_id}"

        # plt.figure(figsize=(8, 6))  # Adjusting the width and height of the figure

        plt.yticks(fontsize=16)
        plt.ylabel('squared error', fontsize=16)
        plt.xticks(fontsize=16)
        plt.xlabel('in-context examples', fontsize=12)  
        for line in plt.gca().lines:
            line.set_linewidth(2)
        plt.legend(fontsize=12, loc=3, title="Number of head")  
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

    method = ['eta0', 'parallel', 'reverse_parallel', 'perp']
    idx = 0

    if args.manual_pkl:
        # with open(f"./plot_results_new/head1_simga0.01_{method[idx]}.pkl", 'wb') as f:
        #     pickle.dump(blue_metrics, f)
        #     print(f"Saved at ./plot_results_new/head1_simga0.5_{method[idx]}.pkl")
        # exit()
    # # import pdb ; pdb.set_trace()
    #     prefix = "head16_simga0.01"
    #     with open(f"./plot_results_new/{prefix}_eta0.pkl", 'rb') as f:
    #         blue_metrics1 = pickle.load(f)
    #     with open(f"./plot_results_new/{prefix}_parallel.pkl", 'rb') as f:
    #         blue_metrics2 = pickle.load(f)
    #     with open(f"./plot_results_new/{prefix}_reverse_parallel.pkl", 'rb') as f:
    #         blue_metrics3 = pickle.load(f)
    #     with open(f"./plot_results_new/{prefix}_perp.pkl", 'rb') as f:
    #         blue_metrics4 = pickle.load(f)
        
        # with open(f"./plot_results_new/local_head1_test_var1.pkl", 'wb') as f:
        #     pickle.dump(blue_metrics, f)
        #     # print(f"Saved at ./plot_results_new/head1_simga0.5_{method[idx]}.pkl")
        # exit()

        # blue_metrics[r'$\eta=0$'] = blue_metrics1['1']
        # blue_metrics[r'$\eta=\theta_0$'] = blue_metrics2['1']
        # blue_metrics[r'$\eta=-\theta_0$'] = blue_metrics3['1']
        # blue_metrics[r'$\eta \perp \theta_0$'] = blue_metrics4['1']

        # blue_models = [r'$\eta=0$', r'$\eta=\theta_0$', r'$\eta=-\theta_0$', r'$\eta \perp \theta_0$']

        prefix = "local_head1_test_var"
        with open(f"./plot_results_new/{prefix}0.01.pkl", 'rb') as f:
            blue_metrics1 = pickle.load(f)
        with open(f"./plot_results_new/{prefix}0.1.pkl", 'rb') as f:
            blue_metrics2 = pickle.load(f)
        with open(f"./plot_results_new/{prefix}1.pkl", 'rb') as f:
            blue_metrics3 = pickle.load(f)
        
        blue_metrics[r'0.01'] = blue_metrics1['1']
        blue_metrics[r'0.1'] = blue_metrics2['1']
        blue_metrics[r'1'] = blue_metrics3['1']

        blue_models = ['0.01', '0.1', '1']

    basic_plot(blue_metrics, models=blue_models, use_log=args.use_log)
    if args.output == '':
        output_file_path = f"./plot_results_new/job{start_id}to{end_id}"
    else:
        output_file_path = f"./plot_results_new/{args.output}"

    # plt.yticks(fontsize=16)
    # plt.ylabel('squared error', fontsize=22)
    # plt.xticks(fontsize=16)
    # plt.xlabel('in-context examples', fontsize=22)
    # for line in plt.gca().lines:
    #     line.set_linewidth(2)
    # plt.legend(fontsize=16, loc=3)
    # plt.tight_layout()
    # plt.savefig(f"{output_file_path}.png")

    plt.yticks(fontsize=20)
    plt.ylabel('squared error', fontsize=26)
    plt.xticks(fontsize=20)
    plt.xlabel('in-context examples', fontsize=26)
    for line in plt.gca().lines:
        line.set_linewidth(2)
    # plt.legend(fontsize=18, loc=1, title=r"$\theta=\theta_0 + \alpha\eta$", title_fontsize=18)
    plt.legend(fontsize=18, loc=3, title="head", title_fontsize=18)
    # plt.legend(fontsize=18, loc=3, title="$p$", title_fontsize=18)
        # plt.legend(fontsize=18, loc=3, title=r"$\sigma_x$", title_fontsize=18)
    plt.tight_layout()
    if args.manual_pkl:
        plt.savefig(f"./plot_results_new/{prefix}.png")
    else:
        plt.savefig(f"./{output_file_path}.png")

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
        with open(f"./plot_results_new/job53to62.pkl", 'rb') as f:
            blue_metrics1 = pickle.load(f)

        with open(f"./plot_results_new/job63to72.pkl", 'rb') as f:
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
        with open(f"./plot_results_new/job{start_id}to{end_id}.pkl", 'wb') as f:
            pickle.dump(blue_metrics, f)

    basic_plot(ave_metrics, models=blue_models, use_log=args.use_log)
    output_file_path = f"./plot_results_new/job{53}to{72}_ave"

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

print(ICL_results)
