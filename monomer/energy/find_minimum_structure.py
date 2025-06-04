from util import energy_monomer_base, read_params_from_file, write_params_to_file
import functools
import jax
import fitting
import jax.numpy as jnp
import optax


EXPONENT_SUM_MAX = 4
EXPONENT_MAX = 5
SKIP_ZERO = True


def find_shift_energy(params):
    params["theta_e"] = params["theta_e"].to(fitting.ureg.rad)

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
        return energy_with_params(r1, r2, theta)

    # Compute the gradient of f with respect to its input parameters.
    grad_f = jax.grad(energy)

    # Optionally jit compile the gradient function for performance.
    grad_f_jit = jax.jit(grad_f)

    # Initialize the parameters [x, y, z] with some starting guess.
    params = jnp.array([1.0, 1.0, 1.8])

    print(grad_f_jit(params))

    # Define a learning rate (step size)
    learning_rate = 0.1

    # Number of iterations for gradient descent
    num_iterations = 1000

    lr_schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,
        decay_rate=0.98,
        staircase=False,
    )

    # Gradient descent loop
    for i in range(num_iterations):
        # Compute the gradients.
        grads = grad_f_jit(params)
        # Update the parameters by stepping in the negative gradient direction.
        params = params - lr_schedule(i) * grads

        # Optionally print the progress every 10 iterations.
        if i % 100 == 0:
            print(
                f"Iteration {i}: params = {params}, f(params) = {energy(params)}, |grad| = {jnp.linalg.norm(grads)}, lr = {lr_schedule(i)}"
            )

    shift_energy = energy(params)

    # Final optimized parameters
    print(f"Optimized parameters: {params}")
    print(f"Energy: {energy(params)}")

    return shift_energy


if __name__ == "__main__":
    # FILE = "output/params_atomic_units.hdf5"

    # FILE_IN = "output_beef/trained_params_atomic.hdf5"
    # FILE_OUT = "output_beef/trained_params_atomic_shifted.hdf5"

    # FILE_IN = "output_rpbe/trained_params_atomic.hdf5"
    # FILE_OUT = "output_rpbe/trained_params_atomic_shifted.hdf5"

    FILE_IN = "output_blyp/trained_params_atomic.hdf5"
    FILE_OUT = "output_blyp/trained_params_atomic_shifted.hdf5"

    params = read_params_from_file(FILE_IN)

    shift_energy = find_shift_energy(params)

    print(params)

    params["energy_correction"] = (
        params["energy_correction"].magnitude - shift_energy
    ) * fitting.ureg.hartree

    write_params_to_file(
        params,
        FILE_OUT,
        exponent_max=EXPONENT_MAX,
        exponent_sum_max=EXPONENT_SUM_MAX,
        skip_zero=SKIP_ZERO,
    )
