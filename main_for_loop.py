import numpy as np
import utils.exampleFunctions as ex_func
import utils.generalDefinitions as gen_def
import utils.models as models
import utils.samplers as samplers

# Fonksiyonların isimlerini al
function_names = ex_func.function_ranges.keys()

# Her bir fonksiyon için işlemleri gerçekleştir
for function_name in function_names:
    try:
        print(f"Working on function: {function_name}")

        function_param = getattr(ex_func, function_name.lower().replace(" ", "_"))

        # Aralığı al
        range_param = ex_func.function_ranges[function_name]

        # Generate a mesh for the background function
        input, output = samplers.Sampler.generate_mesh(range_param, num_points=100, function=function_param)

        # Generate initial samples
        sampler = samplers.LatinHypercubeSampling(function_param)
        initial_points = sampler.sample(num_samples=5, ranges=range_param)
        initial_values = np.array([function_param(i) for i in initial_points])

        # Train the initial model using random forests
        rf_model = models.RandomForestModel(n_estimators=20, random_state=1729)
        rf_model.train_model(initial_points, initial_values)
        trained_rf_model = rf_model.get_model()

        # Perform active learning
        active_learning = models.ActiveLearning(initial_points, initial_values, rf_model, sampler)
        min_function_values, improvement, new_points_all = active_learning.run_active_learning(200, range_param, 3)

        # Plot the results
        sampling_vis = gen_def.SamplingVisualization(initial_points, np.array(new_points_all), min_function_values,
                                                     improvement)
        sampling_vis.plot_visuals(input, output, function_name)
        sampling_vis.plot_results()
    except Exception as e:
        print('İMDAT', (e))
