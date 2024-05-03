import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import utils.generalDefinitions as func_def
from skopt import gp_minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression
    

class ActiveLearning:
    def __init__(self, initial_points, initial_values, model, sampling_strategy, num_iterations=1000, tolerance=0.1):
        self.initial_points = initial_points
        self.initial_values = initial_values
        self.model = model
        self.sampling_strategy = sampling_strategy
        self.num_iterations = num_iterations
        self.tolerance = tolerance

    def run_active_learning(self, num_samples, ranges, k):
        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values

        iter_to_train = 0

        for iteration in range(self.num_iterations):
            
            iter_to_train+= 1
            
            # Generate new candidate samples using sampling strategy
            new_points = self.sampling_strategy.sample(num_samples, ranges)

            # Calculate distance score based on distances
            distances_to_existing = np.array([[min([func_def.euclidean_distance([new_point, sample]) for sample in samples]) for new_point in new_points]])
            max_distance = np.max(distances_to_existing)
            distance_scores = 1 - distances_to_existing / max_distance

            # Predict uncertainty score based on model variance
            variance = np.var([tree.predict(new_points) for tree in self.model.get_model().estimators_], axis=0)
            uncertainty_scores = 1 - variance / np.max(variance)

            # Calculate joint scores
            joint_scores = 0.5 * distance_scores + 0.5 * uncertainty_scores

            # Find the index of the point with the maximum joint score
            index_of_max_joint_score = np.argmin(joint_scores)

            # The point with the maximum joint score
            chosen_new_point = new_points[index_of_max_joint_score]

            # Update the model with the new point
            chosen_new_value = np.array([self.sampling_strategy.evaluate_function(chosen_new_point)])
            samples = np.vstack([samples, chosen_new_point])
            sample_values = np.append(sample_values, chosen_new_value)

            # Update training data
            new_points_all = np.vstack([samples, chosen_new_point])
            new_values_all = np.append(sample_values, chosen_new_value)

            # Sort the values in descending order
            new_values_all.sort()

            # Keep track of the minimum function value observed so far
            min_function_values.append(np.min(sample_values))

            if(iter_to_train == k):    

                # Re-fit the model including the new points
                iter_to_train = 0
                self.model.train_model(samples, sample_values)

            # Calculate improvement
            rmse = (mean_squared_error(sample_values, self.model.get_model().predict(samples))) ** 0.5
            improvement.append(rmse)

            print(f"Iteration {iteration + 1}, RMSE: {rmse}")

            # Check for convergence (change in function value less than the tolerance)
            if len(improvement) > 1 and abs(improvement[-2] - improvement[-1]) <= self.tolerance:
                print(f"Convergence reached at iteration {iteration + 1}.")
                break

        return min_function_values, improvement, new_points_all
    
    
class GradientDescentOptimizer:
    def __init__(self, function):
        self.function = function

    def approximate_gradient(self, point, h=1e-5):
        f = self.function
        gradients = []
        for i in range(len(point)):
            h_vector = np.zeros_like(point)
            h_vector[i] = h
            grad_i = (f(*(point + h_vector)) - f(*(point - h_vector))) / (2 * h)
            gradients.append(grad_i)
        return np.array(gradients)

    def gradient_descent_step(self, point, learning_rate=0.001):
        gradient = self.approximate_gradient(point)
        new_point = point - learning_rate * gradient
        return new_point

    def optimize(self, initial_point, learning_rate=0.001, num_iterations=100, convergence_threshold=1e-6):
        current_point = np.array(initial_point)
        for i in range(num_iterations):
            next_point = self.gradient_descent_step(current_point, learning_rate)
            if np.linalg.norm(next_point - current_point) < convergence_threshold:
                break
            current_point = next_point
        return current_point
    
    
class RandomForestModel:
    def __init__(self, n_estimators=10, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf_model = None

    def train_model(self, initial_points, initial_values):
        self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        self.rf_model.fit(initial_points, initial_values)

    def get_model(self):
        return self.rf_model
    

class LinearRegressionModel:
    def __init__(self):
        self.lr = None

    def fit(self, inputs, output):
        self.lr = LinearRegression()
        self.lr.fit(inputs, output)

    def predict(self, inputs):
        return self.lr.predict(inputs)
    
    
class GaussianProcessRegressionModel:
    def __init__(self, kernel=None, alpha=1e-5):
        self.kernel = kernel
        self.alpha = alpha
        self.gp = None

    def fit(self, inputs, output):
        if self.kernel is None:
            self.kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        self.gp.fit(inputs, output)

    def predict(self, inputs):
        return self.gp.predict(inputs, return_std=True)


class BayesianOptimizer:
    def __init__(self, function, space, acq_func="EI", n_calls=100, n_random_starts=5, noise=0.1**2, random_state=1729):
        self.function = function
        self.space = space
        self.acq_func = acq_func
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.noise = noise
        self.random_state = random_state

    def optimize(self):
        result = gp_minimize(self.function,
            self.space,
            acq_func=self.acq_func,
            n_calls=self.n_calls,
            n_random_starts=self.n_random_starts,
            noise=self.noise,
            random_state=self.random_state)
        return result