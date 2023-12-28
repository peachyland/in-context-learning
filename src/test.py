import torch

# Example for a 2D vector
vec_2d = torch.tensor([2.0, 3.0])
perp_vec_2d = torch.tensor([-vec_2d[1], vec_2d[0]])

# Example for a 3D vector
vec_3d = torch.tensor([1.0, 2.0, 3.0])
basis_vector = torch.tensor([1.0, 0.0, 0.0])
perp_vec_3d = torch.cross(vec_3d, basis_vector)

print("Perpendicular vector in 2D:", perp_vec_2d)
print("Perpendicular vector in 3D:", perp_vec_3d)
