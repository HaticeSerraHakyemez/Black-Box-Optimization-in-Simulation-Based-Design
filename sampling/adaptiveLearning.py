import numpy as np
from utils.exFunctions import eggholder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_min, x_max = -512, 512
y_min, y_max = -512, 512

def adaptive_sampling(func, num_samples, search_space):
    samples = []
    for _ in range(num_samples):
        sample = np.random.uniform(search_space[0], search_space[1], size=(2,))
        samples.append(sample)
        samples.sort(key=lambda x: func(x))
    return np.array(samples)

num_samples = 5
search_space = [(x_min, y_min), (x_max, y_max)]

adaptive_samples = adaptive_sampling(eggholder, num_samples, search_space)

x_values = np.linspace(x_min, x_max, 1000)
y_values = np.linspace(y_min, y_max, 1000)
X, Y = np.meshgrid(x_values, y_values)
Z = eggholder([X, Y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
ax.scatter(adaptive_samples[:, 0], adaptive_samples[:, 1], eggholder(adaptive_samples.T),
           color='red', s=50, label='Sample Points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Eggholder Function')

plt.legend()
plt.show()
