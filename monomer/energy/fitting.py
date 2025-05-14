import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import TypedDict, Dict
from pint import UnitRegistry, Quantity
import h5py

# --- Imports from your project utilities ---
from util import (
    write_params_to_file,
    read_params_from_file,
    energy_monomer_base,
    n_coefficients,
)

# Initialize logging and units
ureg = UnitRegistry()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# TypedDict to hold all configuration parameters for training
# ------------------------------------------------------------------------------
class TrainParams(TypedDict):
    exponent_sum_max: int
    exponent_max: int
    skip_zero: bool
    frac_test: float
    num_epochs: int
    n_epoch_log: int
    initial_lr: float
    transition_steps: int
    decay_rate: float
    weight_decay: float
    lambda_weight: float
    beta: float
    output_plot_dir: Path
    initial_params: Dict[str, Quantity]
    r_e: Quantity
    theta_e: Quantity


# ------------------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------------------
def mse_loss(y_pred: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((y_pred - y) ** 2)


def stable_soft_max(residuals: jnp.ndarray, beta: float) -> jnp.ndarray:
    scaled = beta * residuals
    max_scaled = jnp.max(scaled)
    return (1.0 / beta) * (max_scaled + jnp.log(jnp.sum(jnp.exp(scaled - max_scaled))))


def soft_max_residual_loss(
    y_pred: jnp.ndarray, y: jnp.ndarray, beta: float
) -> jnp.ndarray:
    residuals = jnp.abs(y_pred - y)
    return stable_soft_max(residuals, beta)


# ------------------------------------------------------------------------------
# Plotting utilities
# ------------------------------------------------------------------------------
def plot_loss_curve(epochs, train_losses, test_losses, output_dir: Path):
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, test_losses, label="Test Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close(fig)


def plot_scatter(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, output_file: Path, title: str
):
    fig, ax = plt.subplots()
    # scatter plot of true vs predicted
    ax.scatter(np.array(y_true), np.array(y_pred), alpha=0.6)
    # line y = x
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_xlabel("True Energy")
    ax.set_ylabel("Predicted Energy")
    ax.set_title(title)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_scatter_err(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, output_file: Path, title: str
):
    fig, ax = plt.subplots()
    # error scatter plot of true vs predicted
    ax.scatter(np.array(y_true), np.abs(np.array(y_pred - y_true)), alpha=0.6)
    # line y = x
    ax.set_xlabel("True Energy")
    ax.set_ylabel("abs(Predicted Energy - True Energy)")
    ax.set_title(title)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------
# The main training function
# ------------------------------------------------------------------------------
def train(
    geometries_test: jnp.ndarray,
    energies_test: jnp.ndarray,
    geometries_train: jnp.ndarray,
    energies_train: jnp.ndarray,
    cfg: TrainParams,
) -> Dict[str, Quantity]:
    """
    Train the energy monomer model using JAX and Optax.

    Args:
        geometries_test: test geometry array, shape (n_test, 3)
        energies_test: test energies, shape (n_test,)
        geometries_train: training geometry array, shape (n_train, 3)
        energies_train: training energies, shape (n_train,)
        cfg: configuration dict containing hyperparameters and initial params
    Returns:
        A dict of trained parameters, with pint quantities attached.
    """
    # Unpack config
    beta = cfg["beta"]
    num_epochs = cfg["num_epochs"]
    n_epoch_log = cfg["n_epoch_log"]

    # Prepare the energy monomer partial with fixed r_e and theta_e
    energy_fn = functools.partial(
        energy_monomer_base,
        exponent_sum_max=cfg["exponent_sum_max"],
        exponent_max=cfg["exponent_max"],
        skip_zero=cfg["skip_zero"],
        r_e=cfg["r_e"].magnitude,
        theta_e=cfg["theta_e"].magnitude,
    )

    # Initialize JAX params from pint quantities
    params_jax = {k: v.magnitude for k, v in cfg["initial_params"].items()}
    logger.info(f"Initial JAX params: {params_jax}")

    # Learning rate schedule
    lr_schedule = optax.exponential_decay(
        init_value=cfg["initial_lr"],
        transition_steps=cfg["transition_steps"],
        decay_rate=cfg["decay_rate"],
        staircase=False,
    )

    optimizer = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=cfg["weight_decay"],
    )
    opt_state = optimizer.init(params_jax)

    @jax.jit
    def combined_loss(params, x, y):
        y_pred = energy_fn(x[:, 0], x[:, 1], x[:, 2], **params)
        return mse_loss(y_pred, y) + cfg["lambda_weight"] * soft_max_residual_loss(
            y_pred, y, beta
        )

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(combined_loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    train_losses, test_losses, epochs = [], [], []

    for epoch in range(num_epochs):
        params_jax, opt_state, train_loss = train_step(
            params_jax, opt_state, geometries_train, energies_train
        )
        if epoch % n_epoch_log == 0:
            test_loss = combined_loss(params_jax, geometries_test, energies_test)
            epochs.append(epoch)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss}, test_loss={test_loss}"
            )

    # Plot losses
    plot_loss_curve(epochs, train_losses, test_losses, cfg["output_plot_dir"])

    # Compute predictions for scatter plots
    y_pred_train = energy_fn(
        geometries_train[:, 0],
        geometries_train[:, 1],
        geometries_train[:, 2],
        **params_jax,
    )
    y_pred_test = energy_fn(
        geometries_test[:, 0],
        geometries_test[:, 1],
        geometries_test[:, 2],
        **params_jax,
    )

    mean_error_test = np.mean(y_pred_test - energies_test)
    mean_error_train = np.mean(y_pred_train - energies_train)

    logger.info(f"{mean_error_test = }")
    logger.info(f"{mean_error_train = }")

    # Generate scatter plots
    plot_scatter(
        energies_train,
        y_pred_train,
        cfg["output_plot_dir"] / "scatter_energy_train.png",
        "Train: True vs Predicted Energies",
    )
    plot_scatter(
        energies_test,
        y_pred_test,
        cfg["output_plot_dir"] / "scatter_energy_test.png",
        "Test: True vs Predicted Energies",
    )
    plot_scatter_err(
        energies_train,
        y_pred_train,
        cfg["output_plot_dir"] / "scatter_err_train.png",
        "Train: Error True vs Predicted Energies",
    )
    plot_scatter_err(
        energies_test,
        y_pred_test,
        cfg["output_plot_dir"] / "scatter_err_test.png",
        "Test: Error True vs Predicted Energies",
    )

    # Assemble results with units
    trained_params = {
        k: v * cfg["initial_params"][k].units for k, v in params_jax.items()
    }
    # Reattach r_e and theta_e
    trained_params["r_e"] = cfg["r_e"]
    trained_params["theta_e"] = cfg["theta_e"]

    return trained_params


# Convert to atomic units
def convert_to_atomic_units(q: Quantity) -> Quantity:
    for u in [
        ureg.hartree,
        ureg.bohr,
        1.0 / ureg.bohr,
        1.0 / ureg.bohr**2,
        ureg.radian,
    ]:
        if q.is_compatible_with(u):
            return q.to(u)
    logger.warning(f"Could not convert {q} to atomic units")
    return q


# ------------------------------------------------------------------------------
# Standalone execution: mirrors original script behavior
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(filename="fitting.log", level=logging.INFO)

    # File paths
    FILE_PATH = Path(__file__).parent
    INPUT_FILE = FILE_PATH / "input/fitted_energies.hdf5"
    INITIAL_PARAMS_FILE = FILE_PATH / "input/params.hdf5"
    OUTPUT_FILE = FILE_PATH / "output/params.hdf5"
    OUTPUT_FILE_ATOMIC = FILE_PATH / "output/params_atomic_units.hdf5"
    PLOT_DIR = Path("./plots")

    # Load data
    with h5py.File(INPUT_FILE, "r") as f:
        geom_test = np.array(f["energy"]["geometries"]["test"])
        geom_train = np.array(f["energy"]["geometries"]["train"])
        # convert degrees to radians in third column
        geom_test[:, 2] *= np.pi / 180.0
        geom_train[:, 2] *= np.pi / 180.0
        energies_test = jnp.array(f["energy"]["test"]["target"])
        energies_train = jnp.array(f["energy"]["train"]["target"])

    # Initial parameters
    if INITIAL_PARAMS_FILE.exists():
        init_params = read_params_from_file(INITIAL_PARAMS_FILE)
    else:
        n_coeffs = n_coefficients(exponent_sum_max=4, exponent_max=5, skip_zero=True)
        init_params = {"coefficients": np.random.rand(n_coeffs) * ureg.eV}

    # Fixed constants
    R_E = init_params.pop("r_e").to(ureg.angstrom)
    THETA_E = init_params.pop("theta_e").to(ureg.radian)

    # Build config dict
    cfg: TrainParams = {
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
        "output_plot_dir": PLOT_DIR,
        "initial_params": init_params,
        "r_e": R_E,
        "theta_e": THETA_E,
    }

    # Run training
    trained = train(
        geometries_test=jnp.array(geom_test),
        energies_test=energies_test,
        geometries_train=jnp.array(geom_train),
        energies_train=energies_train,
        cfg=cfg,
    )

    # Write results
    write_params_to_file(
        trained,
        OUTPUT_FILE,
        exponent_max=cfg["exponent_max"],
        exponent_sum_max=cfg["exponent_sum_max"],
        skip_zero=cfg["skip_zero"],
    )

    atomic = {k: convert_to_atomic_units(v) for k, v in trained.items()}
    write_params_to_file(
        atomic,
        OUTPUT_FILE_ATOMIC,
        exponent_max=cfg["exponent_max"],
        exponent_sum_max=cfg["exponent_sum_max"],
        skip_zero=cfg["skip_zero"],
    )
