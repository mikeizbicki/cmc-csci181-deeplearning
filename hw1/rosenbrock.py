'''
The rosenbrock test function is a common "banana-shaped" function to test how well optimization routines work.
See: https://en.wikipedia.org/wiki/Rosenbrock_function
'''
import torch

def rosenbrock(x,y):
    a = 2 
    b = 4 
    return (a-x)**2 + b*(y-x**2)**2

def rosenbrock_mod(x):
    a = 2 
    b = 4 
    return (a-x[0])**2 + b*(x[1]-x[0]**2)**2

# add your code here
alpha = 0.01
x = torch.tensor([0.0,0.0],requires_grad=True)
for i in range(5000):
    print('i=',i,'x=',x)
    z = rosenbrock_mod(x)
    z.backward()
    x = x - alpha * x.grad
    x = torch.tensor(x,requires_grad=True)
