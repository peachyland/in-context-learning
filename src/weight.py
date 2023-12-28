import torch

run_id = "jobid72_embd6_layer1_head1_read_inFalse_9ee850ef-ef88-4922-8798-49145be39b7e"
a=torch.load(f'../models/linear_regression/{run_id}/state.pt')

tmp1 = a['model_state_dict']['_backbone.h.0.attn.c_attn.weight'][:,:12]
torch.set_printoptions(precision=4, linewidth=200)
print((tmp1[:,:6]@tmp1[:,6:].T))
# import torch

# a = torch.load('../models/linear_regression/3500a3f0-028e-4a55-b382-143fe7462168/state.pt')
# torch.set_printoptions(precision=4, linewidth=200)
# tmp1 = a['model_state_dict']['_read_in.bias']
# print((tmp1))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample matrix - replace this with your matrix
matrix = (tmp1[:,:6]@tmp1[:,6:].T).cpu().numpy()

# Create the heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(matrix, annot=True, cmap='viridis', square=True)

# run_id = "jobid65_embd6_layer1_head1_read_inFalse_6558cf88-275b-4c61-81da-060498a7ee39"
output_file_path = f"./plot_results/{run_id}"

# Add title and labels as needed
# plt.title('Heatmap of the Matrix')
plt.xlabel('Column Index', fontsize=21)
plt.ylabel('Row Index', fontsize=21)
plt.xticks(fontsize=15)  # Set x-tick font size
plt.yticks(fontsize=15)  # Set y-tick font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.savefig(f"{output_file_path}_weight.png")

# import pdb ; pdb.set_trace()
