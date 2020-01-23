import torch

def f(x):
    return x**2 + 4*x + 2

def df(x):
    return 2*x + 4

# minimum at x=-2
# analytic formula
# closed-form formula
# for the minimum of f

x = torch.tensor(
    0.0,
    requires_grad=True
    )
y = torch.tensor(
    1.0,
    requires_grad=True
    )


# d/dx f(x) == d/dx z
z = f(x)
z.backward() # computes the derivative

x.grad # this is df(x)

print('f(x)=',f(x))
#print('df(x)=',df(x))
print('x.grad=',x.grad)
print('y.grad=',y.grad)

# gradient descent
x0 = torch.tensor(7.0,requires_grad=True)
z0 = f(x0)
z0.backward()

alpha = 0.1 # step size, learning rate
x1 = x0 - alpha * x0.grad # key formula
x1 = torch.tensor(x1,requires_grad=True)
z1 = f(x1)
z1.backward()

x2 = x1 - alpha * x1.grad

print('x0=',x0)
print('x1=',x1)
print('x2=',x2)

# loop version of gradient descent
x = torch.tensor(7.0,requires_grad=True)
for i in range(50):
    print('i=',i,'x=',x)
    z = f(x)
    z.backward()
    x = x - alpha * x.grad
    x = torch.tensor(x,requires_grad=True)


# higher order tensors
x = torch.tensor([[[[[7.0,5.6]]]]])

x = torch.ones(3,4,5)
# 3rd order = R^m*n*o
# m = 3, n=4, o=5
print('x=',x)

#x = torch.zeros(3,4,5)
#x = torch.empty(3,4,5)
print('x=',x)

z = f(x)
print('z=',z)
z.backward()
x.grad
