import torch
import numpy as np

# Number of samples you want to generate
num_samples = 1000

# The dimension of the vector x (can be adjusted as needed)
dimension = 10  # for example
# Generate the random diagonal matrix Lambda with entries from an exponential distribution with scale = 1
Lambda = torch.diag(torch.from_numpy(np.random.exponential(scale=1, size=dimension)).float())
# Scaling factor c
c = 1
# Covariance matrix is c * Lambda
covariance_matrix = c * Lambda

# Generate samples from the multivariate normal distribution using PyTorch
x_samples = torch.distributions.MultivariateNormal(torch.zeros(dimension), covariance_matrix).sample((num_samples,))

# Display the first 5 samples
print(x_samples[:5])
print(x_samples.shape)
