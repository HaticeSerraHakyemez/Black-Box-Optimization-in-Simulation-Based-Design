import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def euclidean_distance(p):
    return np.sqrt(np.sum((p[0] - p[1]) ** 2))


class SamplingVisualization:
    def __init__(self, initial_points, new_points_all, min_function_values, improvement):
        self.initial_points = initial_points
        self.new_points_all = new_points_all
        self.min_function_values = min_function_values
        self.improvement = improvement

    def plot_visuals(self, input, Z, function_name):
        if len(input) == 1:  # 1D function
            plt.figure(figsize=(12, 6))
            plt.plot(input[0], Z, label=function_name)
            plt.plot(self.initial_points[:, 0], np.zeros_like(self.initial_points[:, 0]), 'o', c='red', label='Initial Samples')
            plt.plot(self.new_points_all[5:, 0], np.zeros_like(self.new_points_all[5:, 0]), 'o', c='blue', label='Adaptive Samples')

            # Save samples to DataFrame
            samples_df = pd.DataFrame({"X": self.new_points_all[:, 0]})


        elif len(input) == 2:  # 2D function
            plt.figure(figsize=(12, 10))
            plt.contourf(input[0], input[1], Z, levels=50, cmap='viridis')
            plt.colorbar()
            plt.scatter(self.initial_points[:, 0], self.initial_points[:, 1], c='red', label='Initial Samples')
            plt.scatter(self.new_points_all[5:, 0], self.new_points_all[5:, 1], c='blue', label='Adaptive Samples')

            # Save samples to DataFrame
            samples_df = pd.DataFrame({"X": self.new_points_all[:, 0], "Y": self.new_points_all[:, 1]})


        else:
            print(f"The function '{function_name}' has a dimension higher than 2 and cannot be visualized.")
            return
        

        plt.legend()
        plt.title(f'Initial and Adaptive Samples on {function_name} Function')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        print(samples_df)


    def plot_results(self):
        # plot convergence
        plt.figure()
        plt.plot(range(len(self.improvement)), self.improvement, marker='o')
        plt.title('Convergence Plot')
        plt.xlabel('Iteration')
        plt.ylabel('Root Mean Squared Error')
        plt.grid(True)
        plt.show()

        # plot_iteration_vs_function_value
        plt.figure(figsize=(10, 5))
        plt.plot(self.min_function_values, marker='o', linestyle='-', color='blue')
        plt.title('Iteration vs Minimum Function Value Observed')
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Function Value')
        plt.grid(True)
        plt.show()

        # print_final_result
        print(f"Best observed function value after active learning: {min(self.min_function_values)}")