import numpy as np
from scipy.stats import qmc
#from sklearn.model_selection import GridSearchCV
import numpy as np

class Sampler:
    def __init__(self, function):
        self.function = function

    @staticmethod
    def generate_mesh(ranges, num_points=100, function=None):
        mesh_args = np.meshgrid(*[np.linspace(range[0], range[1], num_points) for range in ranges])
        if function:
            Z = function(mesh_args)
            return (mesh_args, Z)
        else:
            return mesh_args

    def scale_samples(self, samples, *ranges):
        scaled_samples = (samples * np.array([range[1] - range[0] for range in ranges])) + np.array([range[0] for range in ranges])
        return scaled_samples

    def evaluate_function(self, samples):
        return self.function(samples)
    
   
class GridSearch(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, ranges):
        return 0


class LatinHypercubeSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, ranges):
        dimension = len(ranges)
        sampler = qmc.LatinHypercube(d=dimension)
        samples = sampler.random(n=num_samples)
        scaled_samples = self.scale_samples(samples, *ranges)
        return scaled_samples
    

class RandomSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, ranges, sampler=np.random.rand):
        samples = sampler(num_samples, len(ranges))
        samples_scaled = self.scale_samples(samples, *ranges)
        return samples_scaled
    


class MonteCarloSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, ranges):
        samples = np.column_stack([np.random.uniform(*range, num_samples) for range in ranges])
        return samples



class SobolSequenceSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, ranges):
        dimension = len(ranges)
        sampler = qmc.Sobol(d=dimension, scramble=False)
        samples = sampler.random_base2(m=int(np.log2(num_samples)))
        samples_scaled = qmc.scale(samples, *ranges)
        return samples_scaled
    

class HaltonSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, ranges):
        dimension = len(ranges)
        sampler = qmc.Halton(d=dimension, scramble=False)
        samples = sampler.random(n=num_samples)
        samples_scaled = self.scale_samples(samples, *ranges)
        return samples_scaled
    

class LatinHypercubeSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def sample(self, num_samples, ranges):
        dimension = len(ranges)
        sampler = qmc.LatinHypercube(d=dimension)
        samples = sampler.random(n=num_samples)
        scaled_samples = self.scale_samples(samples, *ranges)
        return scaled_samples
    
    
class AdaptiveSampling(Sampler):
    def __init__(self, function):
        super().__init__(function)

    def initial_samples(self, num_samples=20, *ranges):
        sampler = LatinHypercubeSampling(self.function)
        samples = sampler.sample(num_samples, *ranges)
        return samples

    def sample(self, initial_samples, iterations=5, samples_per_iter=10):
        min_values = np.min(initial_samples, axis=0)
        max_values = np.max(initial_samples, axis=0)
        current_samples = initial_samples
        for _ in range(iterations):
            function_values = self.evaluate_function(current_samples)
            # Indices of points with highest absolute values (proxy for interest)
            top_indices = np.argsort(np.abs(function_values))[-10:]
            top_samples = current_samples[top_indices]

            # Generate new samples around these points
            new_samples = []
            for sample in top_samples:
                new_samples.append(np.random.uniform(low=sample-50, high=sample+50, size=(samples_per_iter, len(sample))))
            new_samples = np.array(new_samples).reshape(-1, len(sample))
            new_samples = np.clip(new_samples, min_values, max_values)

            # Add new samples to the current set
            current_samples = np.vstack((current_samples, new_samples))

        return current_samples
    