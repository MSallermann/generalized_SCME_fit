import logging
from typing import Callable
import numpy as np


logger = logging.getLogger(__name__)


class Fitter:
    def __init__(
        self, objective_function_cb: Callable[[int, dict], float], n_contributions: int
    ):
        self.objective_function_cb = objective_function_cb
        self.n_contributions = n_contributions

    def compute_total_objective_function(self, parameters: dict) -> float:
        result = 0.0

        for idx_cont in range(self.n_contributions):
            p = parameters.copy()
            current_contribution = self.objective_function_cb(idx_cont, p)
            result += current_contribution
            logger.debug(
                f"... Computing contribution {idx_cont} = {current_contribution}"
            )

        logger.debug(f"Current params = {parameters}")
        logger.debug(f"total objective function = {result}")

        return result

    def fit_scipy(self, initial_parameters: dict, **kwargs):
        from scipy.optimize import minimize

        logger.info(f"Start fitting with initial parameters {initial_parameters}")
        logger.info(f"Initial objective function {self.compute_total_objective_function(initial_parameters)}")

        # Scipy expects a function with n real-valued parameters f( x )
        # but our objective function takes a dictionary of parameters
        # this is fine, we just define our objective function locally
        def list_to_dict(x, keys):
            if len(x) != len(keys):
                raise Exception(f"{len(x) = }, {len(keys) = }, {keys = }")

            res = {k: v for k, v in zip(keys, x)}

            return res

        def dict_to_list(params: dict):
            return np.array(list(params.values()))

        def objective_function_scipy(x):
            params_cur = list_to_dict(x, keys=initial_parameters.keys())
            return self.compute_total_objective_function(params_cur)

        result = minimize(
            objective_function_scipy, x0=dict_to_list(initial_parameters), **kwargs
        )
        x = result.x
        opt_params = list_to_dict(x, initial_parameters.keys())


        logger.info(f"Final objective function {self.compute_total_objective_function(opt_params)}")
        logger.info(f"Optimized_parameters {opt_params}")

        return opt_params