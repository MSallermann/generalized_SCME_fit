import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import logging
from pint import UnitRegistry, Quantity

from util import (
    write_params_to_file,
    read_params_from_file,
    energy_monomer_base,
    n_coefficients,
)

ureg = UnitRegistry()
logging.basicConfig(filename="fitting.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
#                 Begin: SETUP
# ===============================================

FILE_PATH = Path(__file__).parent

INPUT_FILE = FILE_PATH / "input/fitted_energies.hdf5"
INITIAL_PARAMS_FILE = FILE_PATH / "input/params.hdf5"
OUTPUT_FILE = FILE_PATH / "output/params.hdf5"
OUTPUT_FILE_ATOMIC_UNITS = FILE_PATH / "output/params_atomic_units.hdf5"

PLOT_DIR = Path("./plots")
logger.info(f"{INPUT_FILE = }")
logger.info(f"{OUTPUT_FILE = }")
logger.info(f"{PLOT_DIR = }")

EXPONENT_SUM_MAX = 4
EXPONENT_MAX = 5
SKIP_ZERO = True
FRACT_TEST = 0.2

logger.info(f"{EXPONENT_SUM_MAX = }")
logger.info(f"{EXPONENT_MAX = }")
logger.info(f"{SKIP_ZERO = }")
logger.info(f"{FRACT_TEST = }")

NUM_EPOCHS = int(1e6)
N_EPOCH_LOG = 10000
INITIAL_LR = 1e-1
TRANSITION_STEPS = 1000
DECAY_RATE = 0.99
LR_SCHEDULE = optax.exponential_decay(
    init_value=INITIAL_LR,  # initial learning rate
    transition_steps=TRANSITION_STEPS,  # number of steps before decay
    decay_rate=DECAY_RATE,  # decay factor applied every transition_steps
    staircase=False,  # if True, decay in discrete intervals; otherwise continuous
)

logger.info(f"{NUM_EPOCHS = }")
logger.info(f"{INITIAL_LR = }")
logger.info(f"{LR_SCHEDULE = }")
logger.info(f"{DECAY_RATE = }")
logger.info(f"{TRANSITION_STEPS = }")


# ===============================================
#                 End: SETUP
# ===============================================


with h5py.File(INPUT_FILE, "r") as f:
    geometries_test = np.array(f["energy"]["geometries"]["test"])
    geometries_train = np.array(f["energy"]["geometries"]["train"])

    geometries_test[:, 2] *= np.pi / 180.0
    geometries_train[:, 2] *= np.pi / 180.0

    energies_test = jnp.array(f["energy"]["test"]["target"])
    energies_train = jnp.array(f["energy"]["train"]["target"])
    energies_fit_anoop = jnp.array(f["energy"]["train"]["pred"])


def mse_loss(y_pred, y):
    return jnp.mean((y_pred - y) ** 2)


def stable_soft_max(residuals, beta):
    scaled = beta * residuals
    max_scaled = jnp.max(scaled)
    return (1.0 / beta) * (max_scaled + jnp.log(jnp.sum(jnp.exp(scaled - max_scaled))))


def soft_max_residual_loss(y_pred, y, beta=50.0):
    # Compute absolute residuals
    residuals = jnp.abs(y_pred - y)
    # Compute the soft maximum using log-sum-exp
    return stable_soft_max(residuals, beta)


def compute_y(x, params):
    return energy_monomer(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        **params,
    )


def combined_loss(params, x, y, lambda_weight=1e-4, beta=50.0):
    # Compute prediction from your model, e.g., using energy_monomer or any custom model
    y_pred = compute_y(x, params)
    loss_mse = mse_loss(y_pred, y)
    loss_softmax = soft_max_residual_loss(y_pred, y, beta=beta)
    # Combine the losses: adjust lambda_weight to control emphasis on outliers
    return loss_mse + lambda_weight * loss_softmax


if Path(INITIAL_PARAMS_FILE).exists():
    init_params = read_params_from_file(INITIAL_PARAMS_FILE)
else:
    init_params = None

n_coeffs = n_coefficients(
    exponent_sum_max=EXPONENT_SUM_MAX, exponent_max=EXPONENT_MAX, skip_zero=SKIP_ZERO
)
if init_params is None or len(init_params["coefficients"]) != n_coeffs:
    logger.warning("Using random initialization for coefficients")
    init_params = {
        "alphaoh": 2.6 / ureg.angstrom,
        "beta": -0.2 / ureg.angstrom**2,
        "deoh": 3.6 * ureg.eV,
        "energy_correction": -50.0 * ureg.eV,
        "phh1": 20.0 * ureg.eV,
        "phh2": 3.0 / ureg.angstrom,
    }

    init_params["coefficients"] = np.random.uniform(size=(n_coeffs)) * ureg.eV

logger.info("Initial parameters:")
logger.info(init_params)

logger.info(f"{len(init_params['coefficients']) = }")
logger.info(f"{n_coeffs = }")

# randomly select 20% of the geometries as test
mask_test = np.random.uniform(size=(len(geometries_train))) <= FRACT_TEST

x_train = jnp.array(geometries_train[~mask_test])
y_train = jnp.array(energies_train[~mask_test])
x_test = jnp.array(geometries_train[mask_test])
y_test = jnp.array(energies_train[mask_test])


def plot_training(x_train, params_jax, epochs, test_losses, train_losses):
    y_pred_train = compute_y(x_train, params_jax)
    max_diff = np.max(np.abs(y_pred_train - y_train))
    avg_diff = np.mean(np.abs(y_pred_train - y_train))

    logger.info(f"{max_diff = }")
    logger.info(f"{avg_diff = }")

    plt.plot(epochs, test_losses, label="loss (test)")
    plt.plot(epochs, train_losses, label="loss (train)")
    plt.yscale("log")
    plt.legend()
    plt.savefig(PLOT_DIR / "test_losses.png", dpi=300)


# Training loop.
def train(
    energy_function,
    num_epochs,
    init_params,
    x_train,
    y_train,
    x_test,
    y_test,
    n_epoch_log=N_EPOCH_LOG,
):
    params_jax = {k: v.magnitude for k, v in init_params.items()}
    logger.info(f"{params_jax = }")

    # compute the initial loss values before training
    initial_loss_train = combined_loss(params_jax, x=x_train, y=y_train)
    initial_loss_test = combined_loss(params_jax, x=x_test, y=y_test)

    logger.info(f"{initial_loss_train = }")
    logger.info(f"{initial_loss_test = }")

    optimizer = optax.adamw(learning_rate=LR_SCHEDULE, weight_decay=1e-4)
    opt_state = optimizer.init(params_jax)

    # Create a JIT-compiled training step that computes gradients with respect to the `params` dictionary.
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(combined_loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    train_losses = []
    test_losses = []
    epochs = []

    for epoch in range(num_epochs):
        # Use training data here (energies_train) for training.
        params_jax, opt_state, loss = train_step(
            params_jax, opt_state, x_train, y_train
        )
        if epoch % n_epoch_log == 0:
            test_loss = combined_loss(params_jax, x_test, y_test)
            epochs.append(epoch)
            test_losses.append(test_loss)
            train_losses.append(loss)

            y_pred_train = compute_y(x_train, params_jax)
            mean_diff = np.mean(np.abs(y_train - y_pred_train))
            max_diff = np.max(np.abs(y_train - y_pred_train))

            logger.info(
                f"=========== EPOCH {epoch} ===========\n"
                f"    Loss (train): {loss}\n"
                f"    Loss (test): {test_loss}\n"
                f"    mean_diff: {mean_diff}\n"
                f"    max_diff: {max_diff}\n"
                f"    lr: {LR_SCHEDULE(epoch):.1e}\n"
            )

    plot_training(
        x_train=x_train,
        params_jax=params_jax,
        epochs=epochs,
        test_losses=test_losses,
        train_losses=train_losses,
    )

    logger.info("Trained parameters:")
    logger.info(params_jax)

    params_result = {}
    for k, v in params_jax.items():
        params_result[k] = v * init_params[k].units

    logger.info("Result parameters (with units):")
    logger.info(params_result)

    return params_result


# Removed r_e and theta_e from the initial parameters,
# because they should not be changed by the optimization
R_E = init_params.pop("r_e").to(ureg.angstrom)
THETA_E = init_params.pop("theta_e").to(ureg.radian)

energy_monomer = functools.partial(
    energy_monomer_base,
    exponent_sum_max=EXPONENT_SUM_MAX,
    exponent_max=EXPONENT_MAX,
    skip_zero=SKIP_ZERO,
    r_e=R_E.magnitude,  # We create a partial function where r_e and theta_e are fixed
    theta_e=THETA_E.magnitude,  # We create a partial function where r_e and theta_e are fixed
)

# All the other parameters are free game to be adjusted
params_result = train(
    energy_monomer,
    num_epochs=NUM_EPOCHS,
    init_params=init_params,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
)

# Put r_e and theta_e back in the dictionary before writing the results
params_result["r_e"] = R_E
params_result["theta_e"] = THETA_E

write_params_to_file(
    params_result,
    OUTPUT_FILE,
    exponent_max=EXPONENT_MAX,
    exponent_sum_max=EXPONENT_SUM_MAX,
    skip_zero=SKIP_ZERO,
)


def convert_to_atomic_units(q: Quantity):
    for u in [
        ureg.hartree,
        1.0 / ureg.hartree,
        ureg.bohr,
        1.0 / ureg.bohr,
        1.0 / ureg.bohr**2,
        ureg.radian,
    ]:
        if q.is_compatible_with(u):
            return q.to(u)

    logger.warn(f"Did not convert {q} to atomic units")
    return q


params_atomic_units = {k: convert_to_atomic_units(v) for k, v in params_result.items()}
logger.info(f"{params_atomic_units = }")
write_params_to_file(
    params_atomic_units,
    OUTPUT_FILE_ATOMIC_UNITS,
    exponent_max=EXPONENT_MAX,
    exponent_sum_max=EXPONENT_SUM_MAX,
    skip_zero=SKIP_ZERO,
)
