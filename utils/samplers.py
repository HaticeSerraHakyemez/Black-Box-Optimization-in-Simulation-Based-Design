import numpy as np
from scipy.stats import qmc

class Sampler:
    def __init__(self, function):
        self.function = function
     
    @staticmethod
    def generate_mesh(x_range=(-512, 512), y_range=(-512, 512), num_points=100, function=None):
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        if function:
            Z = function([X, Y])
            return X, Y, Z
        else:
            return X, Y

    def scale_samples(self, samples, x_min, y_min, x_max, y_max):
        scaled_samples = (samples * np.array([x_max - x_min, y_max - y_min])) + np.array([x_min, y_min])
        return scaled_samples

    def evaluate_function(self, samples):
        return self.function(samples[:, ])
   

class RandomSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, x_min, y_min, x_max, y_max, sampler=np.random.rand):
        samples = sampler(num_samples, 2)
        samples_scaled = self.scale_samples(samples, x_min, y_min, x_max, y_max)
        function_values = self.evaluate_function(samples_scaled)
        return samples_scaled, function_values


class MonteCarloSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, x_min, y_min, x_max, y_max):
        x_samples = np.random.uniform(x_min, x_max, num_samples)
        y_samples = np.random.uniform(y_min, y_max, num_samples)
        samples = np.column_stack((x_samples, y_samples))
        function_values = self.evaluate_function(samples)
        return samples, function_values


class SobolSequenceSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, x_min, y_min, x_max, y_max):
        sampler = qmc.Sobol(d=2, scramble=False)
        samples = sampler.random_base2(m=int(np.log2(num_samples)))
        samples_scaled = qmc.scale(samples, (x_min, y_min), (x_max, y_max))
        function_values = self.evaluate_function(samples_scaled)
        return samples_scaled, function_values
    

class HaltonSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, x_min, y_min, x_max, y_max):
        sampler = qmc.Halton(d=2, scramble=False)
        samples = sampler.random(n=num_samples)
        samples_scaled = self.scale_samples(samples, x_min, y_min, x_max, y_max)
        function_values = self.evaluate_function(samples_scaled)
        return samples_scaled, function_values
    

class LatinHypercubeSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, x_min, y_min, x_max, y_max):
        sampler = qmc.LatinHypercube(d=2)
        samples = sampler.random(n=num_samples)
        scaled_samples = self.scale_samples(samples, x_min, y_min, x_max, y_max)
        return scaled_samples
    
    
class AdaptiveSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def initial_samples(self, num_samples=20, x_min=-512, y_min=-512, x_max=512, y_max=512):
        samples = LatinHypercubeSampling.sample(num_samples, x_min, y_min, x_max, y_max)
        return samples

    def sample(self, initial_samples, iterations=5, samples_per_iter=10):
        x_min, y_min = np.min(initial_samples, axis=0)
        x_max, y_max = np.max(initial_samples, axis=0)
        current_samples = initial_samples
        for _ in range(iterations):
            function_values = self.evaluate_function(current_samples)
            # Indices of points with highest absolute values (proxy for interest)
            top_indices = np.argsort(np.abs(function_values))[-10:]
            top_samples = current_samples[top_indices]

            # Generate new samples around these points
            new_samples = np.array([np.random.uniform(low=sample-50, high=sample+50, size=(samples_per_iter, 2)) for sample in top_samples])
            new_samples = new_samples.reshape(-1, 2)
            new_samples = np.clip(new_samples, [x_min, y_min], [x_max, y_max])

            # Add new samples to the current set
            current_samples = np.vstack((current_samples, new_samples))

        return current_samples
    