from util import energy_monomer_base, read_params_from_file
import functools
import jax
import jax.numpy as jnp
from pint import UnitRegistry

ureg = UnitRegistry()

EXPONENT_SUM_MAX = 4
EXPONENT_MAX = 5
SKIP_ZERO = True


FILE = "output/params_atomic_units.hdf5"
params = read_params_from_file(FILE)
params["theta_e"] = params["theta_e"].to(ureg.rad)

params_jax = {k: v.magnitude for k, v in params.items()}

print(params_jax)

energy_with_params = functools.partial(
    energy_monomer_base,
    **params_jax,
    skip_zero=SKIP_ZERO,
    exponent_sum_max=EXPONENT_SUM_MAX,
    exponent_max=EXPONENT_MAX,
)


@jax.jit
def energy(x):
    r1, r2, theta = x
    rhh = jnp.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * jnp.cos(theta))
    return energy_with_params(r1, r2, theta)


# Compute the gradient of f with respect to its input parameters.
grad_f = jax.grad(energy)

# Optionally jit compile the gradient function for performance.
grad_f_jit = jax.jit(grad_f)

# Initialize the parameters [x, y, z] with some starting guess.
params = jnp.array([1.0, 1.0, 1.8])

print(grad_f_jit(params))

# Define a learning rate (step size)
learning_rate = 0.001

# Number of iterations for gradient descent
num_iterations = 100000

# Gradient descent loop
for i in range(num_iterations):
    # Compute the gradients.
    grads = grad_f_jit(params)
    # Update the parameters by stepping in the negative gradient direction.
    params = params - learning_rate * grads

    # Optionally print the progress every 10 iterations.
    if i % 100 == 0:
        print(f"Iteration {i}: params = {params}, f(params) = {energy(params)}")

# Final optimized parameters
print(f"Optimized parameters: {params}")
