from scme_fitting.fitter import Fitter
import numpy as np
import logging

logging.basicConfig(filename="test_fitter.log", level=logging.DEBUG)


def test_with_square_func():
    def objective_function_cb(idx: int, params: dict):
        if idx == 0:
            return 2.0 * (params["x"] - 2) ** 2
        if idx == 1:
            return 3.0 * (params["y"] + 1) ** 2

    fitter = Fitter(objective_function_cb=objective_function_cb, n_contributions=2)

    initial_params = dict(x=0.0, y=0.0)

    optimal_params = fitter.fit_scipy(initial_parameters=initial_params)

    print(f"{optimal_params = }")

    assert np.isclose(optimal_params["x"], 2.0)
    assert np.isclose(optimal_params["y"], -1.0)


if __name__ == "__main__":
    test_with_square_func()
