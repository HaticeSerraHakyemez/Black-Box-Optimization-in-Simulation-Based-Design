import numpy as np

# this is just a starting example

def f(x, y):
    return np.sqrt(x - 2*y)

x_values = np.linspace(0, 10, 100)  
y_values = np.linspace(0, 5, 50) 

X, Y = np.meshgrid(x_values, y_values)

Z = f(X, Y)

print(Z)
