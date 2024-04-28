import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import utils.generalDefinitions as func_def
    
# LR, GR and BO will be added

class ActiveLearning:
    def __init__(self, initial_points, initial_values, model, sampling_strategy, num_iterations=1000, tolerance=0.1):
        self.initial_points = initial_points
        self.initial_values = initial_values
        self.model = model
        self.sampling_strategy = sampling_strategy
        self.num_iterations = num_iterations
        self.tolerance = tolerance

    def run_active_learning(self):
        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values
        for iteration in range(self.num_iterations):
            # Generate new candidate samples using sampling strategy
            new_points = self.sampling_strategy.sample(samples)

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

            # Re-fit the model including the new points
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

    def approximate_gradient(self, x, y, h=1e-5):
        f = self.function
        grad_x = (f(x + h, y) - f(x - h, y)) / (2 * h)
        grad_y = (f(x, y + h) - f(x, y - h)) / (2 * h)
        return np.array([grad_x, grad_y])

    def gradient_descent_step(self, x, y, learning_rate=0.001):
        gradient = self.approximate_gradient(x, y)
        new_x = x - learning_rate * gradient[0]
        new_y = y - learning_rate * gradient[1]
        return new_x, new_y

    def optimize(self, x_init, y_init, learning_rate=0.001, num_iterations=100, convergence_threshold=1e-6):
        x_current, y_current = x_init, y_init
        for i in range(num_iterations):
            x_next, y_next = self.gradient_descent_step(x_current, y_current, learning_rate)
            if np.sqrt((x_next - x_current)**2 + (y_next - y_current)**2) < convergence_threshold:
                break
            x_current, y_current = x_next, y_next
        return x_current, y_current
    
    
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