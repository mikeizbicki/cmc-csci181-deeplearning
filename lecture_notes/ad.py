print('hello world')

import torch

# 0 order tensors = numbers
x = torch.tensor(0.0)
y = torch.tensor(2.0)
z = x + y
print('z=',z.item())

# 1st order tensors = vectors
x = torch.tensor([1,2,3])

# 2nd order tensor = matrix
m = torch.tensor(
    [[1,2,3]
    ,[4,5,6]])

#m2 = torch.tensor(
#    [[1,2,3,4]
#    ,[4,5,6]])

# 3rd order tensors = cubes
c = torch.tensor([[[3],[3]]])

# two new features of torch
# 1. works on GPUs
# 2. supports automatic diff.
# tensorflow:
# 1. also has TPU
# 2. better deployment deveops
