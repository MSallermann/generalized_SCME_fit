import logging
from typing import Callable
import numpy as np
from typing import Optional, Sequence, Dict

logger = logging.getLogger(__name__)


class Fitter:
    """
    Fits parameters by minimizing a weighted sum of individual contribution functions.

    The Fitter class allows users to define an objective function callback that computes
    individual contributions to a global objective based on an index and a parameter set.
    It then aggregates these contributions with optional weights, and offers an interface
    to optimize the parameters using SciPy.
    """

    def __init__(
        self,
        objective_function_cb: Callable[[int, Dict[str, float]], float],
        n_contributions: int,
        weights: Optional[Sequence[float]] = None,
    ):
        """
        Initialize a Fitter instance.

        Parameters
        ----------
        objective_function_cb : Callable[[int, Dict[str, float]], float]
            A callback function that, given an integer index and a parameter dict,
            returns a float contribution to the objective.
        n_contributions : int
            The number of contributions the callback will be called for.
        weights : Sequence[float], optional
            An optional sequence of weights for each contribution. If None,
            all contributions are weighted equally.

        Raises
        ------
        AssertionError
            If provided weights do not match the number of contributions.
        """

        self.obj_cb = objective_function_cb
        self.n_contrib = n_contributions
        self.weights = (
            np.ones(n_contributions) if weights is None else np.array(weights)
        )
        assert self.weights.shape == (n_contributions,)
        self._keys: list[str] = []

    def compute_total(self, params: Dict[str, float]) -> float:
        """
        Compute the weighted sum of contributions from the objective callback.

        Parameters
        ----------
        params : Dict[str, float]
            Dictionary mapping parameter names to their current float values.

        Returns
        -------
        float
            The total weighted objective value.

        Notes
        -----
        A copy of the params dict is passed to each callback invocation to
        prevent unintended side effects if the callback mutates its input.
        """

        result = 0
        for i, w in enumerate(self.weights):
            # We make a copy of params here, just in case the objective function modifies it
            p = params.copy()
            result += self.obj_cb(i, p) * w

        return result

    def fit_scipy(self, initial_parameters: Dict[str, float], **kwargs) -> Dict:
        """
        Optimize parameters using SciPy's minimize function.

        Parameters
        ----------
        initial_parameters : Dict[str, float]
            Initial guess for each parameter, as a mapping from name to value.
        **kwargs
            Additional keyword arguments passed directly to scipy.optimize.minimize.

        Returns
        -------
        Dict[str, float]
            Dictionary of optimized parameter values.

        Warnings
        --------
        If the optimizer does not converge, a warning is logged.

        Example
        -------
        >>> def objective_function_cb(idx: int, params: dict):
        ...     if idx == 0:
        ...         return 2.0 * (params["x"] - 2) ** 2
        ...     if idx == 1:
        ...         return 3.0 * (params["y"] + 1) ** 2
        >>> fitter = Fitter(objective_function_cb=objective_function_cb, n_contributions=2)
        >>> initial_params = dict(x=0.0, y=0.0)
        >>> optimal_params = fitter.fit_scipy(initial_parameters=initial_params)
        >>> print(optimal_params)
        {'x': 2.0, 'y': -1.0}
        """

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
