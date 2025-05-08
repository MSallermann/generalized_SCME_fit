import logging
from typing import Callable
import numpy as np
from typing import Optional, Sequence, Dict

logger = logging.getLogger(__name__)


class Fitter:
    """
    Fits parameters by minimizing a weighted sum of individual contribution functions.
    """

    def __init__(
        self,
        objective_function_cb: Callable[[int, Dict[str, float]], float],
        n_contributions: int,
        weights: Optional[Sequence[float]] = None,
    ):
        self.obj_cb = objective_function_cb
        self.n_contrib = n_contributions
        self.weights = (
            np.ones(n_contributions) if weights is None else np.array(weights)
        )
        assert self.weights.shape == (n_contributions,)
        self._keys: list[str] = []

    def compute_total(self, params: Dict[str, float]) -> float:
        result = 0
        for i, w in enumerate(self.weights):
            # We make a copy of params here, just in case the objective function modifies it
            p = params.copy()
            result += self.obj_cb(i, p) * w

        return result

    def fit_scipy(self, initial_parameters: Dict[str, float], **kwargs):
        from scipy.optimize import minimize

        logger.info(f"Start fitting with initial parameters {initial_parameters}")
        logger.info(
            f"Initial objective function {self.compute_total(initial_parameters)}"
        )

        # capture key order once
        self._keys = list(initial_parameters.keys())
        x0 = np.array([initial_parameters[k] for k in self._keys])

        # Scipy expects a function with n real-valued parameters f(x)
        # but our objective function takes a dictionary of parameters.
        # This is fine, we just define our objective function locally and
        # put a parameters in a dictionary based on the captured keys
        def f_scipy(x):
            p = dict(zip(self._keys, x))
            return self.compute_total(p)

        res = minimize(f_scipy, x0, **kwargs)
        if not res.success:
            logger.warning("Fit did not converge: %s", res.message)

        opt_params = dict(zip(self._keys, res.x))
        return opt_params  # full OptimizeResult
