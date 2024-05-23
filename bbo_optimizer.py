import numpy as np
import utils.functions_simple as ex_func 
import utils.models as models

def optimize(function_name, sampling_method, predictor, ):
    function_param = getattr(ex_func, function_name.lower().replace(" ", "_"))
    range_param = ex_func.function_ranges[function_name]

    # Generate initial samples
    sampler = sampling_method(function_param)
    initial_points = sampler.sample(num_samples=5, ranges=range_param)
    initial_values = np.array([function_param(i) for i in initial_points])

    # Train the initial model using random forests
    rf_model = predictor(n_estimators=200, random_state=1729)
    rf_model.train_model(initial_points, initial_values)

    # Perform active learning
    active_learning = models.ActiveLearning(initial_points, initial_values, rf_model, sampler, 50)
    result = active_learning.run_active_learning(2000, range_param, 3)

    print(f"Best observed function value after active learning: {min(result['min_function_values'])}")