import fitting
import pandas as pd
import numpy as np
import jax.numpy as jnp
from util import n_coefficients
import logging
from pathlib import Path

logging.basicConfig(filename="run_fit.log", level=logging.INFO)


default_train_params: fitting.TrainParams = {
    "exponent_sum_max": 4,
    "exponent_max": 5,
    "skip_zero": True,
    "frac_test": 0.2,
    "num_epochs": int(1e6),
    "n_epoch_log": 10000,
    "initial_lr": 1e-1,
    "transition_steps": 1000,
    "decay_rate": 0.99,
    "weight_decay": 1e-4,
    "lambda_weight": 1e-4,
    "beta": 50.0,
    "output_plot_dir": "plots",
    "initial_params": {},
    "r_e": 0.0,
    "theta_e": 0.0,
}

# These are made-up
default_initial_params = {
    "alphaoh": 2.7 / fitting.ureg.angstrom,
    "beta": -0.2 / fitting.ureg.angstrom**2,
    "deoh": 3.0 * fitting.ureg.electron_volt,
    "energy_correction": -0.6 * fitting.ureg.electron_volt,
    "phh1": 20.2 * fitting.ureg.electron_volt,
    "phh2": 0.7 / fitting.ureg.angstrom,
}


### BEEF
def process_csv(path_csv):
    data = pd.read_csv(path_csv)
    n_samples = len(data)

    geometries = np.zeros((n_samples, 3))

    geometries[:, 0] = data["r1"]
    geometries[:, 1] = data["r2"]
    geometries[:, 2] = data["Th"] * np.pi / 180.0

    energies = np.zeros((n_samples))
    energies[:] = data["energy"]

    return jnp.array(geometries), jnp.array(energies)


def fit(path_csv, r_e, theta_e, output_dir):
    output_dir = Path(output_dir)

    cfg = default_train_params.copy()

    initial_params = default_initial_params.copy()

    n_coeffs = n_coefficients(
        exponent_sum_max=cfg["exponent_sum_max"],
        exponent_max=cfg["exponent_max"],
        skip_zero=True,
    )
    initial_params["coefficients"] = np.random.rand(n_coeffs) * fitting.ureg.eV

    cfg["initial_params"] = initial_params
    cfg["r_e"] = r_e
    cfg["theta_e"] = theta_e
    cfg["output_plot_dir"] = output_dir

    geometries, energies = process_csv(path_csv)

    trained = fitting.train(
        geometries_train=geometries,
        energies_train=energies,
        geometries_test=geometries,
        energies_test=energies,
        cfg=cfg,
    )

    # Write results
    fitting.write_params_to_file(
        trained,
        output_dir / "trained_params.hdf5",
        exponent_max=cfg["exponent_max"],
        exponent_sum_max=cfg["exponent_sum_max"],
        skip_zero=cfg["skip_zero"],
    )

    atomic = {k: fitting.convert_to_atomic_units(v) for k, v in trained.items()}
    fitting.write_params_to_file(
        atomic,
        output_dir / "trained_params_atomic.hdf5",
        exponent_max=cfg["exponent_max"],
        exponent_sum_max=cfg["exponent_sum_max"],
        skip_zero=cfg["skip_zero"],
    )


# BEEF
path_csv = "./input/DATA_BEEF_energy.csv"
r_e = 0.964 * fitting.ureg.angstrom
theta_e = 104.3 * fitting.ureg.degree
theta_e = theta_e.to(fitting.ureg.radian)
fit(path_csv, r_e, theta_e, "./output_beef")

# RPBE
path_csv = "./input/DATA_RPBE_energy_reflected.csv"
r_e = 0.972 * fitting.ureg.angstrom
theta_e = 103.8 * fitting.ureg.degree
theta_e = theta_e.to(fitting.ureg.radian)
fit(path_csv, r_e, theta_e, "./output_rpbe")
