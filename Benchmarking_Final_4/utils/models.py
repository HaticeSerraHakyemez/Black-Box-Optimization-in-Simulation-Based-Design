import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import utils.generalDefinitions as func_def
from skopt import gp_minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression


class ActiveLearning:
    def __init__(self, initial_points, initial_values, model, sampling_strategy, num_iterations=1000, tolerance=0.15):
        self.initial_points = initial_points
        self.initial_values = initial_values
        self.model = model
        self.sampling_strategy = sampling_strategy
        self.num_iterations = num_iterations
        self.tolerance = tolerance

    # Weight adjustment function based on iteration and performance
    def adjust_weights(self, weights, iteration, total_iterations, prev_error, curr_error):
        weight_factor = (iteration / total_iterations)
        error_improvement = prev_error - curr_error
        if error_improvement > 0:
            weight_factor += 0.1
        w_distance, w_uncertainty, w_prediction = weights
        w_prediction = min(w_prediction + weight_factor * 0.05, 0.7)
        remaining_weight = 1 - w_prediction
        w_distance = remaining_weight / 2
        w_uncertainty = remaining_weight / 2
        return (w_distance, w_uncertainty, w_prediction)    

    def run_active_learning(self, num_samples, ranges, k):

        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values

        prev_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
        weights = (0.33, 0.33, 0.34)
        val_errors = []
        increasing_error_count = 0  # To monitor increasing validation errors

        iter_to_train = 0

        for iteration in range(self.num_iterations):

            if(iteration>100):
                break

            curr_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
            if curr_val_error > prev_val_error:
                increasing_error_count += 1
            else:
                increasing_error_count = 0

            if increasing_error_count >= 10:  # Stop if validation error increases for three consecutive iterations
                pass
                print("Stopping early due to increasing validation error.")
                break

            weights = self.adjust_weights(weights, iteration, self.num_iterations, prev_val_error, curr_val_error)
            #weights = (0.5, 0, 0.7)
            prev_val_error = curr_val_error
            val_errors.append(curr_val_error)
            
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

            # Predict values for the candidate samples
            predicted_values = self.model.get_model().predict(new_points)

            # Normalize the predicted values between 0 and 1 using Min-Max normalization
            min_predicted_value = np.min(predicted_values)
            max_predicted_value = np.max(predicted_values)
            predicted_score = (predicted_values - min_predicted_value) / (max_predicted_value - min_predicted_value)

            # Calculate joint scores
            #0 ÇARPANINI UNUTMA
            joint_scores = weights[0]*distance_scores + weights[1]*uncertainty_scores + weights[2]*predicted_score

            # Find the index of the point with the maximum joint score
            index_of_max_joint_score = np.argmin(joint_scores)

            # The point with the maximum joint score
            chosen_new_point = new_points[index_of_max_joint_score]

            # Update the model with the new point
            chosen_new_value = np.array([self.sampling_strategy.evaluate_function(chosen_new_point)])
            

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

            #print(f"Iteration {iteration + 1}, RMSE: {rmse}")
            #print('weights:',weights)

            # Check for convergence (change in function value less than the tolerance)
            if len(improvement) > 1 and abs(improvement[-2] - improvement[-1]) <= self.tolerance:
                #print(f"Convergence reached at iteration {iteration + 1}.")
                break
            samples = np.vstack([samples, chosen_new_point])
            sample_values = np.append(sample_values, chosen_new_value)
        iteration = iteration + 1

        return min_function_values, improvement, new_points_all, iteration
    
    def run_active_learning_constant_iterations(self, num_samples, ranges, k):

        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values

        prev_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
        weights = (0.33, 0.33, 0.34)
        val_errors = []
        increasing_error_count = 0  # To monitor increasing validation errors

        iter_to_train = 0

        for iteration in range(self.num_iterations):

            if(iteration>99):
                break

            curr_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
            if curr_val_error > prev_val_error:
                increasing_error_count += 1
            else:
                increasing_error_count = 0

            if increasing_error_count >= 10:  # Stop if validation error increases for three consecutive iterations
                pass
                print("Stopping early due to increasing validation error.")
                break

            weights = self.adjust_weights(weights, iteration, self.num_iterations, prev_val_error, curr_val_error)
            #weights = (0.5, 0, 0.7)
            prev_val_error = curr_val_error
            val_errors.append(curr_val_error)
            
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

            # Predict values for the candidate samples
            predicted_values = self.model.get_model().predict(new_points)

            # Normalize the predicted values between 0 and 1 using Min-Max normalization
            min_predicted_value = np.min(predicted_values)
            max_predicted_value = np.max(predicted_values)
            predicted_score = (predicted_values - min_predicted_value) / (max_predicted_value - min_predicted_value)

            # Calculate joint scores
            #0 ÇARPANINI UNUTMA
            joint_scores = weights[0]*distance_scores + weights[1]*uncertainty_scores + weights[2]*predicted_score

            # Find the index of the point with the maximum joint score
            index_of_max_joint_score = np.argmin(joint_scores)

            # The point with the maximum joint score
            chosen_new_point = new_points[index_of_max_joint_score]

            # Update the model with the new point
            chosen_new_value = np.array([self.sampling_strategy.evaluate_function(chosen_new_point)])
            

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

            #print(f"Iteration {iteration + 1}, RMSE: {rmse}")
            #print('weights:',weights)

            # Check for convergence (change in function value less than the tolerance)
            #if len(improvement) > 1 and abs(improvement[-2] - improvement[-1]) <= self.tolerance:
             #   print(f"Convergence reached at iteration {iteration + 1}.")
              #  break
            samples = np.vstack([samples, chosen_new_point])
            sample_values = np.append(sample_values, chosen_new_value)
        iteration = iteration + 1

        return min_function_values, improvement, new_points_all, iteration
    
    def run_active_learning_constant_iterations_50(self, num_samples, ranges, k):

        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values

        prev_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
        weights = (0.33, 0.33, 0.34)
        val_errors = []
        increasing_error_count = 0  # To monitor increasing validation errors

        iter_to_train = 0

        for iteration in range(self.num_iterations):

            if(iteration>49):
                break

            curr_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
            if curr_val_error > prev_val_error:
                increasing_error_count += 1
            else:
                increasing_error_count = 0

            if increasing_error_count >= 10:  # Stop if validation error increases for three consecutive iterations
                pass
                print("Stopping early due to increasing validation error.")
                break

            weights = self.adjust_weights(weights, iteration, self.num_iterations, prev_val_error, curr_val_error)
            #weights = (0.5, 0, 0.7)
            prev_val_error = curr_val_error
            val_errors.append(curr_val_error)
            
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

            # Predict values for the candidate samples
            predicted_values = self.model.get_model().predict(new_points)

            # Normalize the predicted values between 0 and 1 using Min-Max normalization
            min_predicted_value = np.min(predicted_values)
            max_predicted_value = np.max(predicted_values)
            predicted_score = (predicted_values - min_predicted_value) / (max_predicted_value - min_predicted_value)

            # Calculate joint scores
            #0 ÇARPANINI UNUTMA
            joint_scores = weights[0]*distance_scores + weights[1]*uncertainty_scores + weights[2]*predicted_score

            # Find the index of the point with the maximum joint score
            index_of_max_joint_score = np.argmin(joint_scores)

            # The point with the maximum joint score
            chosen_new_point = new_points[index_of_max_joint_score]

            # Update the model with the new point
            chosen_new_value = np.array([self.sampling_strategy.evaluate_function(chosen_new_point)])
            

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

            #print(f"Iteration {iteration + 1}, RMSE: {rmse}")
            #print('weights:',weights)

            # Check for convergence (change in function value less than the tolerance)
            #if len(improvement) > 1 and abs(improvement[-2] - improvement[-1]) <= self.tolerance:
             #   print(f"Convergence reached at iteration {iteration + 1}.")
              #  break
            samples = np.vstack([samples, chosen_new_point])
            sample_values = np.append(sample_values, chosen_new_value)
        iteration = iteration + 1

        return min_function_values, improvement, new_points_all, iteration
    
    def run_active_learning_constant_iterations_25(self, num_samples, ranges, k):

        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values

        prev_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
        weights = (0.33, 0.33, 0.34)
        val_errors = []
        increasing_error_count = 0  # To monitor increasing validation errors

        iter_to_train = 0

        for iteration in range(self.num_iterations):

            if(iteration>24):
                break

            curr_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
            if curr_val_error > prev_val_error:
                increasing_error_count += 1
            else:
                increasing_error_count = 0

            if increasing_error_count >= 10:  # Stop if validation error increases for three consecutive iterations
                pass
                print("Stopping early due to increasing validation error.")
                break

            weights = self.adjust_weights(weights, iteration, self.num_iterations, prev_val_error, curr_val_error)
            #weights = (0.5, 0, 0.7)
            prev_val_error = curr_val_error
            val_errors.append(curr_val_error)
            
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

            # Predict values for the candidate samples
            predicted_values = self.model.get_model().predict(new_points)

            # Normalize the predicted values between 0 and 1 using Min-Max normalization
            min_predicted_value = np.min(predicted_values)
            max_predicted_value = np.max(predicted_values)
            predicted_score = (predicted_values - min_predicted_value) / (max_predicted_value - min_predicted_value)

            # Calculate joint scores
            #0 ÇARPANINI UNUTMA
            joint_scores = weights[0]*distance_scores + weights[1]*uncertainty_scores + weights[2]*predicted_score

            # Find the index of the point with the maximum joint score
            index_of_max_joint_score = np.argmin(joint_scores)

            # The point with the maximum joint score
            chosen_new_point = new_points[index_of_max_joint_score]

            # Update the model with the new point
            chosen_new_value = np.array([self.sampling_strategy.evaluate_function(chosen_new_point)])
            

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

            #print(f"Iteration {iteration + 1}, RMSE: {rmse}")
            #print('weights:',weights)

            # Check for convergence (change in function value less than the tolerance)
            #if len(improvement) > 1 and abs(improvement[-2] - improvement[-1]) <= self.tolerance:
             #   print(f"Convergence reached at iteration {iteration + 1}.")
              #  break
            samples = np.vstack([samples, chosen_new_point])
            sample_values = np.append(sample_values, chosen_new_value)
        iteration = iteration + 1

        return min_function_values, improvement, new_points_all, iteration
    
    def run_active_learning_constant_iterations_15(self, num_samples, ranges, k):

        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values

        prev_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
        weights = (0.33, 0.33, 0.34)
        val_errors = []
        increasing_error_count = 0  # To monitor increasing validation errors

        iter_to_train = 0

        for iteration in range(self.num_iterations):

            if(iteration>14):
                break

            curr_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
            if curr_val_error > prev_val_error:
                increasing_error_count += 1
            else:
                increasing_error_count = 0

            if increasing_error_count >= 10:  # Stop if validation error increases for three consecutive iterations
                pass
                print("Stopping early due to increasing validation error.")
                break

            weights = self.adjust_weights(weights, iteration, self.num_iterations, prev_val_error, curr_val_error)
            #weights = (0.5, 0, 0.7)
            prev_val_error = curr_val_error
            val_errors.append(curr_val_error)
            
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

            # Predict values for the candidate samples
            predicted_values = self.model.get_model().predict(new_points)

            # Normalize the predicted values between 0 and 1 using Min-Max normalization
            min_predicted_value = np.min(predicted_values)
            max_predicted_value = np.max(predicted_values)
            predicted_score = (predicted_values - min_predicted_value) / (max_predicted_value - min_predicted_value)

            # Calculate joint scores
            #0 ÇARPANINI UNUTMA
            joint_scores = weights[0]*distance_scores + weights[1]*uncertainty_scores + weights[2]*predicted_score

            # Find the index of the point with the maximum joint score
            index_of_max_joint_score = np.argmin(joint_scores)

            # The point with the maximum joint score
            chosen_new_point = new_points[index_of_max_joint_score]

            # Update the model with the new point
            chosen_new_value = np.array([self.sampling_strategy.evaluate_function(chosen_new_point)])
            

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

            #print(f"Iteration {iteration + 1}, RMSE: {rmse}")
            #print('weights:',weights)

            # Check for convergence (change in function value less than the tolerance)
            #if len(improvement) > 1 and abs(improvement[-2] - improvement[-1]) <= self.tolerance:
             #   print(f"Convergence reached at iteration {iteration + 1}.")
              #  break
            samples = np.vstack([samples, chosen_new_point])
            sample_values = np.append(sample_values, chosen_new_value)
        iteration = iteration + 1

        return min_function_values, improvement, new_points_all, iteration
    
    def run_active_learning_no_prediction(self, num_samples, ranges, k):

        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values

        prev_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
        weights = (0.33, 0.33, 0.34)
        val_errors = []
        increasing_error_count = 0  # To monitor increasing validation errors

        iter_to_train = 0

        for iteration in range(self.num_iterations):

            curr_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
            if curr_val_error > prev_val_error:
                increasing_error_count += 1
            else:
                increasing_error_count = 0

            if increasing_error_count >= 10:  # Stop if validation error increases for three consecutive iterations
                pass
                print("Stopping early due to increasing validation error.")
                break

            weights = self.adjust_weights(weights, iteration, self.num_iterations, prev_val_error, curr_val_error)
            #weights = (0.5, 0, 0.7)
            prev_val_error = curr_val_error
            val_errors.append(curr_val_error)
            
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

            # Predict values for the candidate samples
            predicted_values = self.model.get_model().predict(new_points)

            # Normalize the predicted values between 0 and 1 using Min-Max normalization
            min_predicted_value = np.min(predicted_values)
            max_predicted_value = np.max(predicted_values)
            predicted_score = (predicted_values - min_predicted_value) / (max_predicted_value - min_predicted_value)

            # Calculate joint scores
            #0 ÇARPANINI UNUTMA
            joint_scores = weights[0]*distance_scores + weights[1]*uncertainty_scores + 0*weights[2]*predicted_score

            # Find the index of the point with the maximum joint score
            index_of_max_joint_score = np.argmin(joint_scores)

            # The point with the maximum joint score
            chosen_new_point = new_points[index_of_max_joint_score]

            # Update the model with the new point
            chosen_new_value = np.array([self.sampling_strategy.evaluate_function(chosen_new_point)])
            

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
            print('weights:',weights)

            # Check for convergence (change in function value less than the tolerance)
            if len(improvement) > 1 and abs(improvement[-2] - improvement[-1]) <= self.tolerance:
                print(f"Convergence reached at iteration {iteration + 1}.")
                break
            samples = np.vstack([samples, chosen_new_point])
            sample_values = np.append(sample_values, chosen_new_value)
        iteration = iteration + 1

        return min_function_values, improvement, new_points_all, iteration
    
    def run_active_learning_no_weight_adjustment_low_pred(self, num_samples, ranges, k):

        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values

        prev_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
        weights = (0.33, 0.33, 0.34)
        val_errors = []
        increasing_error_count = 0  # To monitor increasing validation errors

        iter_to_train = 0

        for iteration in range(self.num_iterations):

            if(iteration>24):
                break

            curr_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
            if curr_val_error > prev_val_error:
                increasing_error_count += 1
            else:
                increasing_error_count = 0

            if increasing_error_count >= 10:  # Stop if validation error increases for three consecutive iterations
                pass
                print("Stopping early due to increasing validation error.")
                break

            #weights = self.adjust_weights(weights, iteration, self.num_iterations, prev_val_error, curr_val_error)
            weights = (0.45, 0.4, 0.15)
            prev_val_error = curr_val_error
            val_errors.append(curr_val_error)
            
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

            # Predict values for the candidate samples
            predicted_values = self.model.get_model().predict(new_points)

            # Normalize the predicted values between 0 and 1 using Min-Max normalization
            min_predicted_value = np.min(predicted_values)
            max_predicted_value = np.max(predicted_values)
            predicted_score = (predicted_values - min_predicted_value) / (max_predicted_value - min_predicted_value)

            # Calculate joint scores
            #0 ÇARPANINI UNUTMA
            joint_scores = weights[0]*distance_scores + weights[1]*uncertainty_scores + weights[2]*predicted_score

            # Find the index of the point with the maximum joint score
            index_of_max_joint_score = np.argmin(joint_scores)

            # The point with the maximum joint score
            chosen_new_point = new_points[index_of_max_joint_score]

            # Update the model with the new point
            chosen_new_value = np.array([self.sampling_strategy.evaluate_function(chosen_new_point)])
            

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

            #print(f"Iteration {iteration + 1}, RMSE: {rmse}")
            #print('weights:',weights)

            # Check for convergence (change in function value less than the tolerance)
            if len(improvement) > 1 and abs(improvement[-2] - improvement[-1]) <= self.tolerance:
                #print(f"Convergence reached at iteration {iteration + 1}.")
                break
            samples = np.vstack([samples, chosen_new_point])
            sample_values = np.append(sample_values, chosen_new_value)
        iteration = iteration + 1

        return min_function_values, improvement, new_points_all, iteration
    
    def run_active_learning_no_weight_adjustment_high_pred(self, num_samples, ranges, k):

        # Initialize lists to track performance metrics
        min_function_values = [np.min(self.initial_values)]  # Track min function value for each iteration
        improvement = []
        new_points_all = []  # Store all the new points generated

        # Run active learning loop
        samples = self.initial_points
        sample_values = self.initial_values

        prev_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
        weights = (0.33, 0.33, 0.34)
        val_errors = []
        increasing_error_count = 0  # To monitor increasing validation errors

        iter_to_train = 0

        for iteration in range(self.num_iterations):

            if(iteration>24):
                break

            curr_val_error = mean_squared_error(sample_values, self.model.get_model().predict(samples))
            if curr_val_error > prev_val_error:
                increasing_error_count += 1
            else:
                increasing_error_count = 0

            if increasing_error_count >= 10:  # Stop if validation error increases for three consecutive iterations
                pass
                print("Stopping early due to increasing validation error.")
                break

            #weights = self.adjust_weights(weights, iteration, self.num_iterations, prev_val_error, curr_val_error)
            weights = (0.1, 0.1, 0.8)
            prev_val_error = curr_val_error
            val_errors.append(curr_val_error)
            
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

            # Predict values for the candidate samples
            predicted_values = self.model.get_model().predict(new_points)

            # Normalize the predicted values between 0 and 1 using Min-Max normalization
            min_predicted_value = np.min(predicted_values)
            max_predicted_value = np.max(predicted_values)
            predicted_score = (predicted_values - min_predicted_value) / (max_predicted_value - min_predicted_value)

            # Calculate joint scores
            #0 ÇARPANINI UNUTMA
            joint_scores = weights[0]*distance_scores + weights[1]*uncertainty_scores + weights[2]*predicted_score

            # Find the index of the point with the maximum joint score
            index_of_max_joint_score = np.argmin(joint_scores)

            # The point with the maximum joint score
            chosen_new_point = new_points[index_of_max_joint_score]

            # Update the model with the new point
            chosen_new_value = np.array([self.sampling_strategy.evaluate_function(chosen_new_point)])
            

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

            #print(f"Iteration {iteration + 1}, RMSE: {rmse}")
            #print('weights:',weights)

            # Check for convergence (change in function value less than the tolerance)
            if len(improvement) > 1 and abs(improvement[-2] - improvement[-1]) <= self.tolerance:
                #print(f"Convergence reached at iteration {iteration + 1}.")
                break
            samples = np.vstack([samples, chosen_new_point])
            sample_values = np.append(sample_values, chosen_new_value)
        iteration = iteration + 1

        return min_function_values, improvement, new_points_all, iteration
    
    
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
    def __init__(self, n_estimators=100, random_state=None):
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