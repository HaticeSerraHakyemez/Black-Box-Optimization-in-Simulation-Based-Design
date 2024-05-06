import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the Eggholder function
def eggholder_function(x, y):
    term1 = -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47))))
    term2 = -x * np.sin(np.sqrt(abs(x - (y + 47))))
    return term1 + term2

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Sample initial points using Latin hypercube sampling
lhs = qmc.LatinHypercube(d=2)
initial_samples = lhs.random(n=5)
initial_points = qmc.scale(initial_samples, [-512, -512], [512, 512])
initial_values = np.array([eggholder_function(x, y) for x, y in initial_points])

# Number of iterations for adaptive sampling loop
num_iterations = 10

# Define the objective function for hyperopt
def objective(params):
    x, y = params['x'], params['y']
    return eggholder_function(x, y)

# Define the search space for hyperopt
space = {'x': hp.uniform('x', -512, 512),
         'y': hp.uniform('y', -512, 512)}

# Generate a mesh for the background Eggholder function
x = np.linspace(-512, 512, 100)
y = np.linspace(-512, 512, 100)
X, Y = np.meshgrid(x, y)
Z = eggholder_function(X, Y)

# Trials object to store intermediate results
trials = Trials()

# Run adaptive sampling loop
for iteration in range(num_iterations):
    # Generate new candidate samples using Latin hypercube sampling
    new_samples_lhs = lhs.random(n=20)
    new_points = qmc.scale(new_samples_lhs, [-512, -512], [512, 512])

    # Calculate distance score based on Euclidean distances
    distances_to_existing = np.array([[min([euclidean_distance(new_point, sample) for sample in initial_points]) for new_point in new_points]])
    max_distance = np.max(distances_to_existing)
    distance_scores = 1 - distances_to_existing / max_distance

    # Predict uncertainty score based on RF variance
    # Since we don't have the RF model here, let's assume random scores
    uncertainty_scores = np.random.rand(len(new_points))

    # Calculate joint score
    joint_scores = 0.5*distance_scores + 0.5*uncertainty_scores

    # Find the index of the point with the maximum joint score
    index_of_max_joint_score = np.argmin(joint_scores)

    # The point with the maximum joint score
    chosen_new_point = new_points[index_of_max_joint_score]

    # Update the model with the new point
    chosen_new_value = np.array([eggholder_function(chosen_new_point[0], chosen_new_point[1])])

    # Update training data
    initial_points = np.vstack([initial_points, chosen_new_point])
    initial_values = np.append(initial_values, chosen_new_value)

    # Optionally, you can sort the values and keep track of the minimum function value observed so far

# Run TPE optimization
best_params = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

# Extract samples and values from trials
all_samples = [trial['misc']['vals'] for trial in trials.trials]
all_values = [trial['result']['loss'] for trial in trials.trials]

# Print number of iterations
print(f"Number of iterations: {len(all_samples)}")

# Print all samples and values
print("All samples and corresponding values:")
for i in range(len(all_samples)):
    print(f"Iteration {i+1}: Sample: {all_samples[i]}, Value: {all_values[i]}")

# Extract best parameters
best_x = best_params['x']
best_y = best_params['y']
best_value = eggholder_function(best_x, best_y)

# Print results
print("\nBest Parameters:")
print(f"x: {best_x}, y: {best_y}")
print(f"Best Value (Eggholder Function): {best_value}")

# Calculate RMSE from the best value
rmse = mean_squared_error(initial_values, np.full_like(initial_values, best_value))**0.5
print(f"RMSE: {rmse}")

# Plot initial and final samples
plt.figure(figsize=(12, 10))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.scatter(initial_points[:,0], initial_points[:, 1], c='red', label='Initial Samples')
for sample in all_samples:
    plt.scatter(sample['x'], sample['y'], c='blue', marker='x', alpha=0.5)
plt.scatter(best_x, best_y, c='green', label='Best Sample (TPE)')
plt.legend()
plt.title('Initial Samples, All Samples, and Best Sample (TPE) on Eggholder Function')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
