import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

import logging

logging.basicConfig(filename="fitting.log", level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_PATH = Path(__file__).parent

INPUT_FILE = FILE_PATH / "input/fitted_energies.hdf5"
INITIAL_PARAMS_FILE = FILE_PATH / "input/params.hdf5"
OUTPUT_FILE = FILE_PATH / "output/params.hdf5"

PLOT_DIR = Path("./plots")
logger.info(f"{INPUT_FILE = }")
logger.info(f"{OUTPUT_FILE = }")
logger.info(f"{PLOT_DIR = }")

R_E = 0.97109
THETA_E = 104.140
EXPONENT_SUM_MAX = 4
EXPONENT_MAX = 5
SKIP_ZERO = True
FRACT_TEST = 0.2

logger.info(f"{R_E = }")
logger.info(f"{THETA_E = }")
logger.info(f"{EXPONENT_SUM_MAX = }")
logger.info(f"{EXPONENT_MAX = }")
logger.info(f"{SKIP_ZERO = }")
logger.info(f"{FRACT_TEST = }")

NUM_EPOCHS = int(1e6)
N_EPOCH_LOG = 5000
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


def read_params_from_file(file: Path):
    logger.info(f"Reading parameters from {file}")
    with h5py.File(file, "r") as f:
        params = dict(
            alphaoh=jnp.array(f["energy"]["alphaoh"]),
            beta=jnp.array(f["energy"]["beta"]),
            coefficients=jnp.array(f["energy"]["coefficients"]),
            deoh=jnp.array(f["energy"]["deoh"]),
            energy_correction=jnp.array(f["energy"]["energy_correction"]),
            phh1=jnp.array(f["energy"]["phh1"]),
            phh2=jnp.array(f["energy"]["phh2"]),
        )
    return params


def write_params_to_file(params, file, exponent_max, exponent_sum_max, r_e, theta_e):
    with h5py.File(file, "w") as f:
        energy = f.create_group("energy")

        f.attrs["exponent_max"] = exponent_max
        f.attrs["exponent_sum_max"] = exponent_sum_max

        energy_correction = energy.create_dataset(
            "energy_correction",
            data=params["energy_correction"],
        )
        energy_correction.attrs["units"] = "eV"

        deoh = energy.create_dataset("deoh", data=params["deoh"])
        deoh.attrs["units"] = "eV"

        alphaoh = energy.create_dataset("alphaoh", data=params["alphaoh"])
        alphaoh.attrs["units"] = "1/Angstrom"

        phh1 = energy.create_dataset("phh1", data=params["phh1"])
        phh1.attrs["units"] = "eV"

        phh2 = energy.create_dataset("phh2", data=params["phh2"])
        phh2.attrs["units"] = "1/Angstrom"

        beta = energy.create_dataset("beta", data=params["beta"])
        beta.attrs["units"] = "1/Angstrom^2"

        r_e = energy.create_dataset("r_e", data=r_e)
        r_e.attrs["units"] = "Angstrom"

        theta_e = energy.create_dataset("theta_e", data=theta_e)
        theta_e.attrs["units"] = "deg"

        coefficients = energy.create_dataset(
            name="coefficients", data=params["coefficients"]
        )
        coefficients.attrs["units"] = "eV"
        coefficients.attrs["exponent_max"] = exponent_max
        coefficients.attrs["exponent_sum_max"] = exponent_sum_max
        coefficients.attrs["rank"] = 0


def Va(r1, deoh, alphaoh):
    val = deoh * (jnp.exp(-2.0 * alphaoh * r1) - 2.0 * jnp.exp(-alphaoh * r1))
    return val


def Vb(r, A, b):
    return A * jnp.exp(-b * r)


def n_coefficients(exponent_max, exponent_sum_max, skip_zero):
    counter = 0
    for i in range(0, exponent_max):
        for j in range(0, exponent_max):
            for k in range(0, exponent_max):
                if (i + j + k) <= exponent_sum_max:
                    if skip_zero and (i + j + k) == 0:
                        continue
                    counter += 1
    return counter


def energy_monomer_base(
    r1,
    r2,
    theta,
    rhh,
    coefficients,
    alphaoh,
    beta,
    deoh,
    energy_correction,
    phh1,
    phh2,
    r_e,
    theta_e,
    exponent_sum_max,
    exponent_max,
    skip_zero,
):
    va1 = Va(r1 - r_e, deoh, alphaoh)
    va2 = Va(r2 - r_e, deoh, alphaoh)
    vb = Vb(rhh, phh1, phh2)

    value = energy_correction + va1 + va2 + vb

    # Partridge Schwenke definition
    s1 = (r1 - r_e) / r_e
    s2 = (r2 - r_e) / r_e
    s3 = jnp.cos(theta * np.pi / 180) - jnp.cos(theta_e * np.pi / 180)

    counter = 0
    expansion_sum = 0
    for i in range(0, exponent_max):
        for j in range(0, exponent_max):
            for k in range(0, exponent_max):
                if (i + j + k) <= exponent_sum_max:
                    if skip_zero and (i + j + k) == 0:
                        continue

                    expansion_sum += (
                        coefficients[counter]
                        * jnp.power(s1, i)
                        * jnp.power(s2, j)
                        * jnp.power(s3, k)
                    )
                    counter += 1

    exponential_prefactor = jnp.exp(
        -beta * (jnp.power(r1 - r_e, 2) + jnp.power(r2 - r_e, 2))
    )
    expansion_sum *= exponential_prefactor

    tot_val = value + expansion_sum

    return tot_val


energy_monomer = functools.partial(
    energy_monomer_base,
    r_e=R_E,
    theta_e=THETA_E,
    exponent_sum_max=EXPONENT_SUM_MAX,
    exponent_max=EXPONENT_MAX,
    skip_zero=SKIP_ZERO,
)

n_coeffs = n_coefficients(
    exponent_sum_max=EXPONENT_SUM_MAX, exponent_max=EXPONENT_MAX, skip_zero=SKIP_ZERO
)


with h5py.File(INPUT_FILE, "r") as f:
    geometries_test = jnp.array(f["energy"]["geometries"]["test"])
    geometries_train = jnp.array(f["energy"]["geometries"]["train"])

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
        x[:, 3],
        params["coefficients"],
        params["alphaoh"],
        params["beta"],
        params["deoh"],
        params["energy_correction"],
        params["phh1"],
        params["phh2"],
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

logger.info(f"{len(init_params['coefficients']) = }")
logger.info(f"{n_coeffs = }")

if init_params is None or len(init_params["coefficients"]) != n_coeffs:
    logger.warning("Using random initialization for coefficients")
    init_params = {
        "alphaoh": 2.6,
        "beta": -0.2,
        "deoh": 3.6,
        "energy_correction": -50.0,
        "phh1": 20.0,
        "phh2": 3.0,
    }
    init_params["coefficients"] = np.random.uniform(size=(n_coeffs))

# randomly select 20% of the geometries as test
mask_test = np.random.uniform(size=(len(geometries_train))) <= FRACT_TEST

x_train = jnp.array(geometries_train[~mask_test])
y_train = jnp.array(energies_train[~mask_test])
x_test = jnp.array(geometries_train[mask_test])
y_test = jnp.array(energies_train[mask_test])

# compute the initial loss values before training
initial_loss_train = combined_loss(init_params, x=x_train, y=y_train)
initial_loss_test = combined_loss(init_params, x=x_test, y=y_test)
logger.info(f"{initial_loss_train = }")
logger.info(f"{initial_loss_test = }")


optimizer = optax.adamw(learning_rate=LR_SCHEDULE, weight_decay=1e-4)

opt_state = optimizer.init(init_params)


# Create a JIT-compiled training step that computes gradients with respect to the `params` dictionary.
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(combined_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# Training loop.
train_losses = []
test_losses = []
epochs = []


logger.info("Initial parameters:")
logger.info(init_params)

params = init_params
for epoch in range(NUM_EPOCHS):
    # Use training data here (energies_train) for training.
    params, opt_state, loss = train_step(params, opt_state, x_train, y_train)
    if epoch % N_EPOCH_LOG == 0:
        train_loss = combined_loss(params, x_train, y_train)
        test_loss = combined_loss(params, x_test, y_test)
        epochs.append(epoch)
        test_losses.append(test_loss)
        train_losses.append(loss)

        y_pred_train = compute_y(x_train, params)
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

logger.info("Trained parameters:")
logger.info(params)

write_params_to_file(
    params,
    OUTPUT_FILE,
    exponent_max=EXPONENT_MAX,
    exponent_sum_max=EXPONENT_SUM_MAX,
    r_e=R_E,
    theta_e=THETA_E,
)


y_pred_train = compute_y(x_train, params)

max_diff = np.max(np.abs(y_pred_train - y_train))
max_diff_anoop = np.max(np.abs(energies_fit_anoop[~mask_test] - y_train))
avg_diff = np.mean(np.abs(y_pred_train - y_train))
avg_diff_anoop = np.mean(np.abs(energies_fit_anoop[~mask_test] - y_train))

logger.info(f"{max_diff = }")
logger.info(f"{max_diff_anoop = }")
logger.info(f"{avg_diff = }")
logger.info(f"{avg_diff_anoop = }")


plt.plot(epochs, test_losses, label="loss (test)")
plt.plot(epochs, train_losses, label="loss (train)")
plt.yscale("log")
plt.legend()
plt.savefig(PLOT_DIR / "test_losses.png", dpi=300)

plt.close()

plt.plot(y_pred_train, marker="o", ls="None", label="Target")
plt.plot(y_train, marker=".", ls="None", label="Prediction")
plt.plot(energies_fit_anoop[~mask_test], ls="None", marker="x", label="Anoops fit")

plt.legend()
plt.savefig(PLOT_DIR / "scatter.png", dpi=300)

plt.close()
plt.plot(y_train, np.abs(y_pred_train - y_train), ls="None", marker=".", label="sgd")
plt.plot(
    y_train,
    np.abs(energies_fit_anoop[~mask_test] - y_train),
    ls="None",
    marker=".",
    label="anoop",
)
plt.legend()
plt.savefig(PLOT_DIR / "energy_vs_fit", dpi=300)
