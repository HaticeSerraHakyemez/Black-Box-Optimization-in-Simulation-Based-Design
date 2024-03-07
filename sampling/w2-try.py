import numpy as np

def eggholder_function(x, y):
    term1 = -(y + 47) * np.sin(np.sqrt(abs(x/2 + (y + 47))))
    term2 = -x * np.sin(np.sqrt(abs(x - (y + 47))))
    return term1 + term2

x_min, x_max = -512, 512
y_min, y_max = -512, 512

x_values = np.linspace(x_min, x_max, 1000)
y_values = np.linspace(y_min, y_max, 1000)
X, Y = np.meshgrid(x_values, y_values)

Z = eggholder_function(X, Y)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyDOE import lhs

num_samples = 50

lhs_samples = lhs(2, samples=num_samples, criterion='maximin')

lhs_samples_scaled = np.zeros_like(lhs_samples)
lhs_samples_scaled[:, 0] = lhs_samples[:, 0] * (x_max - x_min) + x_min
lhs_samples_scaled[:, 1] = lhs_samples[:, 1] * (y_max - y_min) + y_min

lhs_function_values = Z[np.round(lhs_samples[:, 1] * (Z.shape[0] - 1)).astype(int),
                        np.round(lhs_samples[:, 0] * (Z.shape[1] - 1)).astype(int)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

ax.scatter(lhs_samples_scaled[:, 0], lhs_samples_scaled[:, 1], lhs_function_values, color='red', s=50)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Eggholder Function')

plt.show()
