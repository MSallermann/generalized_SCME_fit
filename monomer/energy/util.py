import jax.numpy as jnp
import numpy as np
import h5py
from pathlib import Path
import logging
from pint import UnitRegistry, Quantity
from typing import Tuple, List

ureg = UnitRegistry()
logging.basicConfig(filename="fitting.log", level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pint_quantity_from_dataset(dataset: h5py.Dataset):
    if "units" in dataset.attrs:
        unit_str = str(dataset.attrs["units"])
    else:
        unit_str = "None"

    return np.array(dataset) * ureg.parse_expression(unit_str)


def read_params_from_file(file: Path):
    logger.info(f"Reading parameters from {file}")
    with h5py.File(file, "r") as f:
        params = dict(
            alphaoh=get_pint_quantity_from_dataset(f["energy"]["alphaoh"]),
            beta=get_pint_quantity_from_dataset(f["energy"]["beta"]),
            coefficients=get_pint_quantity_from_dataset(f["energy"]["coefficients"]),
            deoh=get_pint_quantity_from_dataset(f["energy"]["deoh"]),
            energy_correction=get_pint_quantity_from_dataset(
                f["energy"]["energy_correction"]
            ),
            phh1=get_pint_quantity_from_dataset(f["energy"]["phh1"]),
            phh2=get_pint_quantity_from_dataset(f["energy"]["phh2"]),
            r_e=get_pint_quantity_from_dataset(f["energy"]["r_e"]),
            theta_e=get_pint_quantity_from_dataset(f["energy"]["theta_e"]),
        )

    return params


def write_pint_quantity_to_dataset(dataset: h5py.Dataset, key: str, q: Quantity):
    d = dataset.create_dataset(key, data=q.magnitude)
    try:
        d.attrs["units"] = str(q.units)
    except BaseException() as e:
        logger.warning(f"Could not log units for: {key}. \n {e}")


def get_exponent_arrays(
    max_exponent: int, max_sum_exponent: int, skip_zero: bool
) -> Tuple[List[int], List[int], List[int]]:
    """
    Generate lists of exponent indices for three variables (i, j, k) such that their sum
    does not exceed a given maximum.

    Args:
        max_exponent (int): The upper limit (exclusive) for each individual exponent.
        max_sum_exponent (int): The maximum allowed sum of exponents (i + j + k).
        skip_zero (bool): If `True`, skips the i=j=k=0 exponent


    Returns:
        Tuple[List[int], List[int], List[int]]: Three lists containing valid exponent indices for i, j, and k.
    """
    exponents_i: List[int] = []
    exponents_j: List[int] = []
    exponents_k: List[int] = []

    for i in range(0, max_exponent):
        for j in range(0, max_exponent):
            for k in range(0, max_exponent):
                if (i + j + k) <= max_sum_exponent:
                    if i == 0 and j == 0 and k == 0 and skip_zero:
                        continue
                    exponents_i.append(i)
                    exponents_j.append(j)
                    exponents_k.append(k)

    return exponents_i, exponents_j, exponents_k


def write_params_to_file(params, file, exponent_max, exponent_sum_max, skip_zero):
    with h5py.File(file, "w") as f:
        energy = f.create_group("energy")

        energy.attrs["exponent_max"] = exponent_max
        energy.attrs["exponent_sum_max"] = exponent_sum_max

        # Get the exponent arrays for i, j, and k
        exponents_i, exponents_j, exponents_k = get_exponent_arrays(
            exponent_max, exponent_sum_max, skip_zero=skip_zero
        )

        # Create one dataset per exponent array in the output group
        energy.create_dataset(name="exponents_i", data=np.array(exponents_i))
        energy.create_dataset(name="exponents_j", data=np.array(exponents_j))
        energy.create_dataset(name="exponents_k", data=np.array(exponents_k))

        for k, v in params.items():
            write_pint_quantity_to_dataset(energy, k, v)


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
    rhh = jnp.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * jnp.cos(theta))
    va1 = Va(r1 - r_e, deoh, alphaoh)
    va2 = Va(r2 - r_e, deoh, alphaoh)
    vb = Vb(rhh, phh1, phh2)

    value = energy_correction + va1 + va2 + vb

    # Partridge Schwenke definition
    s1 = (r1 - r_e) / r_e
    s2 = (r2 - r_e) / r_e
    s3 = jnp.cos(theta) - jnp.cos(theta_e)

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
