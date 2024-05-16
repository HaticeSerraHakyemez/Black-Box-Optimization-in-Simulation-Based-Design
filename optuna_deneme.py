import numpy as np
import utils.exampleFunctions as ex_func 
import utils.generalDefinitions as gen_def
import utils.models as models
import utils.samplers as samplers
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

function_name = "Eggholder"

function_param = getattr(ex_func, function_name.lower().replace(" ", "_"))


def objective(trial):
    ranges = ex_func.function_ranges[function_name]
    params = [trial.suggest_uniform(f'x{i}', min_range, max_range) for i, (min_range, max_range) in enumerate(ranges)]

    param_sets = []
    
    for _ in range(20):
        params = [trial.suggest_uniform(f'x{i}', min_range, max_range) for i, (min_range, max_range) in enumerate(ranges)]
        param_sets.append(params)
    
    rf_regressor = RandomForestRegressor(n_estimators=200)
    score = cross_val_score(rf_regressor, param_sets, [function_param(params) for params in param_sets], cv=5).mean()
    
    return score


study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)

print('Best parameters:', study.best_params)
print('Best value:', study.best_value)

function_param([study.best_params['x0'],study.best_params['x1']])