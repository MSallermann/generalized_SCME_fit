from pint import Quantity


class Fitter:
    def __init__(self, fit_func, initial_params: dict):
        self.fit_func = fit_func

        self.initial_params = {}
        for k, v in initial_params.items():
            if isinstance(v, Quantity):
                self.initial_params[k] = v.magnitude
            else:
                self.initial_params[k] = v
