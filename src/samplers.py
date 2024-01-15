import math

import torch
import numpy as np


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims, device='cuda:0')
            # xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims, device='cuda:0')
            # xs_b = torch.randn(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator, device='cuda:0')
                # xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale.to(xs_b.device)
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
    
# class GaussianSampler(DataSampler):
#     def __init__(self, n_dims, bias=None, scale=None):
#         super().__init__(n_dims)
#         self.bias = bias
#         self.scale = scale

#         print("####################################")
#         print("#")
#         print(f"# [Warning!] Danger! You are using only neighbor examples. Please be careful.")
#         print("#")
#         print("####################################")

#     def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
#         local_variance = 1
#         print(local_variance)
#         xs_q = torch.randn(b_size, 1, self.n_dims, device='cuda:0')
#         if n_points == 1:
#             xs_b = xs_q
#         else:
#             xs_icl = xs_q + local_variance * torch.randn(b_size, n_points-1, self.n_dims, device='cuda:0')
#             xs_b = torch.cat([xs_icl, xs_q], dim=1)
#         #     import pdb ; pdb.set_trace()
#         # import pdb ; pdb.set_trace()
        
#         if self.scale is not None:
#             xs_b = xs_b @ self.scale.to(xs_b.device)
#         if self.bias is not None:
#             xs_b += self.bias
#         if n_dims_truncated is not None:
#             xs_b[:, :, n_dims_truncated:] = 0
#         return xs_b


# class GaussianSampler(DataSampler):
#     def __init__(self, n_dims, bias=None, scale=None):
#         super().__init__(n_dims)
#         self.bias = bias
#         self.scale = scale
#         dimension = self.n_dims
#         Lambda = torch.load("correlated_lambda.pt")
#         # Scaling factor c
#         c = 1
#         # Covariance matrix is c * Lambda
#         covariance_matrix = c * Lambda
#         self.lambda_diag_dis = torch.distributions.MultivariateNormal(torch.zeros(dimension), covariance_matrix)

#         print("####################################")
#         print("#")
#         print(f"# [Warning!] Danger! You are using correlated. Please be careful.")
#         print("#")
#         print("####################################")

#     def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
#         if seeds is None:
#             # xs_b = torch.randn(b_size, n_points, self.n_dims, device='cuda:0')
#             # Number of samples you want to generate
#             num_samples = b_size * n_points

#             # # The dimension of the vector x (can be adjusted as needed)
#             # dimension = self.n_dims  # for example
#             # # Generate the random diagonal matrix Lambda with entries from an exponential distribution with scale = 1
#             # # Lambda = torch.diag(torch.from_numpy(np.random.exponential(scale=1, size=dimension)).float())
#             # # import pdb ; pdb.set_trace()
#             # Lambda = torch.load("correlated_lambda.pt")
#             # # Scaling factor c
#             # c = 1
#             # # Covariance matrix is c * Lambda
#             # covariance_matrix = c * Lambda

#             # Generate samples from the multivariate normal distribution using PyTorch
#             xs_b = self.lambda_diag_dis.sample((num_samples,)).to('cuda:0').reshape((b_size, n_points, self.n_dims))
#             # xs_b = torch.distributions.MultivariateNormal(torch.zeros(dimension), covariance_matrix).sample((num_samples,)).to('cuda:0').reshape((b_size, n_points, self.n_dims))
#         else:
#             xs_b = torch.zeros(b_size, n_points, self.n_dims, device='cuda:0')
#             # xs_b = torch.randn(b_size, n_points, self.n_dims)
#             generator = torch.Generator()
#             assert len(seeds) == b_size
#             for i, seed in enumerate(seeds):
#                 generator.manual_seed(seed)
#                 xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator, device='cuda:0')
#                 # xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
#         if self.scale is not None:
#             xs_b = xs_b @ self.scale.to(xs_b.device)
#         if self.bias is not None:
#             xs_b += self.bias
#         if n_dims_truncated is not None:
#             xs_b[:, :, n_dims_truncated:] = 0
#         return xs_b
