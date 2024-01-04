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

# for task in sub_tasks:


sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"
task = 'linear_regression'
run_id = "jobid40_embd64_layer1_head8_read_inTrue_debad986-60a7-41dd-a06c-013e4691667c"  # if you train more models, replace with the run_id from the table above 
# jobid114_embd64_layer1_head1_read_inTrue_theta0True_sigma0.01_fca2b4ff-5eab-43aa-83ae-04acb3fd3e42


df = read_run_dir(run_dir, single_task=True, job_id=run_id)
print(df)

# jobid30_embd64_layer1_head1_read_inFalse_34312e97-f075-4b99-8c75-1283e4dbd150
# jobid31_embd64_layer1_head2_read_inFalse_b62d442a-c261-4544-a8e9-e9803b45a4c0
# jobid32_embd64_layer1_head3_read_inFalse_9f6b4a62-824c-4339-b4a3-adeb34a47277
# jobid33_embd64_layer1_head6_read_inFalse_cc78c189-ff9d-4568-99cc-fce17d40e122

# jobid34_embd6_layer1_head6_read_inTrue_fbfd213d-282a-4682-a98e-b40bc9e022e4
# jobid35_embd6_layer1_head2_read_inTrue_5d459d48-97b2-4810-b91a-069030c0ae1f
# jobid37_embd6_layer1_head1_read_inTrue_712152dc-cf41-4f26-acee-91a45c666794
# jobid38_embd6_layer1_head3_read_inTrue_bb319f3e-a8c1-403b-8d03-607cbaa53101

# jobid39_embd64_layer1_head1_read_inTrue_024866d3-180d-47ee-98bf-04f852d2024c
# jobid40_embd64_layer1_head8_read_inTrue_debad986-60a7-41dd-a06c-013e4691667c
# jobid51_embd64_layer1_head16_read_inTrue_ee7e16c5-d0f9-4404-9f58-4f0e453d91e7
# jobid41_embd64_layer1_head64_read_inTrue_756fcae5-0310-42e3-b1cb-c47a72386551

# jobid42_embd128_layer1_head1_read_inTrue_51036fef-fb44-49f1-8013-aeebb33d8168
# jobid43_embd128_layer1_head8_read_inTrue_e9fa9b32-b311-471d-a10c-faedaf1ba5f6
# jobid44_embd128_layer1_head64_read_inTrue_c4c57dd6-5c81-461a-b6eb-b5527c074f9a
# jobid45_embd128_layer1_head128_read_inTrue_cb007eeb-afb5-4839-9943-c1cf9332ca4c

# jobid46_embd256_layer1_head8_read_inTrue_71266dc7-d66a-4e24-bda8-eab73761dd6b
# jobid47_embd256_layer1_head64_read_inTrue_91f9eb90-689d-427f-b4a1-81b37162fa57
# jobid48_embd256_layer1_head128_read_inTrue_c860d51d-9cd9-4eb9-b680-210972bf3733
# jobid49_embd256_layer1_head256_read_inTrue_02b53af4-adda-45e5-902e-d55d574d2b2e

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False

if recompute_metrics:
    print("recompute_metrics")
    get_run_metrics(run_path)  # these are normally precomputed at the end of training


def valid_row(r):
    return r.task == task and r.run_id == run_id #and r.run_id2 == run_id2

metrics = collect_results(run_dir, df, valid_row=valid_row)

# import pdb ; pdb.set_trace()

models = relevant_model_names[task]
prefix_split = run_id.split('_')
n_head = prefix_split[3].replace('head', '')
if n_head == '8':
    models[0] = "Transformer-1-layer"
else:
    models[0] = f"Transformer-1-layer-{n_head}-head"
# print(models)
# import pdb ; pdb.set_trace()
basic_plot(metrics["standard"], models=models, use_log=False)
output_file_path = f"./plot_results/{run_id}"

# plt.figure(figsize=(12, 6))  # Adjusting the width and height of the figure

plt.yticks(fontsize=16)
plt.ylabel('squared error', fontsize=16)  
plt.xticks(fontsize=16)
plt.xlabel('in-context examples', fontsize=12)  
for line in plt.gca().lines:
    line.set_linewidth(2)
plt.legend(fontsize=12, loc=3)
plt.tight_layout()  
plt.savefig(f"{output_file_path}.png") 








