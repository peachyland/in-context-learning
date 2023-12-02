import torch

a=torch.load('../models/linear_regression/3500a3f0-028e-4a55-b382-143fe7462168/state.pt')

tmp1 = a['model_state_dict']['_backbone.h.0.attn.c_attn.weight'][:,:12]
torch.set_printoptions(precision=4, linewidth=200)
print((tmp1[:,:6]@tmp1[:,6:].T))
# import torch

# a = torch.load('../models/linear_regression/3500a3f0-028e-4a55-b382-143fe7462168/state.pt')
# torch.set_printoptions(precision=4, linewidth=200)
# tmp1 = a['model_state_dict']['_read_in.bias']
# print((tmp1))
