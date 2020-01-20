'''
The rosenbrock test function is a common "banana-shaped" function to test how well optimization routines work.
See: https://en.wikipedia.org/wiki/Rosenbrock_function
'''
import torch

def rosenbrock(x,y):
    a = 2 
    b = 4 
    return (a-x)**2 + b*(y-x**2)**2

# add your code here
