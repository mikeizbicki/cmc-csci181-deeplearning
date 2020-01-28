import torch

x=torch.ones(2,3)

print('x=',x)

print('sum=',torch.einsum('im->',[x]))
print('trans=',torch.einsum('ij->ji',[x]))

y = torch.ones(2)
print('l2=',torch.einsum('i,i->',[y,y]))

print('complex=',torch.einsum(
    'ij, ij, ij -> ij',
    [x,x,x]
    ))
