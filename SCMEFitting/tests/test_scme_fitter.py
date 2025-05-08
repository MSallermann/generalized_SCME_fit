from scme_fitting.fitter import Fitter
from scme_fitting.scme_setup import SCMEParams
from scme_fitting.scme_fitter import SCMEObjectiveFunction
import numpy as np
import logging
from pathlib import Path


logging.basicConfig(filename="test_scme_fitter.log", level=logging.INFO)


def create_scme_fit_data(base_path: Path):
    energies = np.loadtxt(base_path / "PES_dimer_c1_PBE.txt")[:, 1]
    paths = list(base_path.glob("*/CONTCAR"))
    sorted_paths = sorted(paths, key=lambda p: float(p.parent.name))
    return sorted_paths, energies


def test_scme_fitting():
    base_path = Path(
        "/home/moritz/SCME/generalized_SCME_interatomic_fit/SCMEFitting/scme_fitting/resources/PBE"
    )

    paths_to_reference_configurations, reference_energies = create_scme_fit_data(base_path)

    parametrization_key = "component_PBE_fullrange_reflect_4_5"
    parametrization_key = "component_PBE_fullrange_reflect_6_9"
    parametrization_key = "component_PBE_fullrange_reflect_8_12"

    parametrization_key = None

    n_contributions = len(reference_energies)

    DEFAULT_PARAMS = SCMEParams()
    ADJUSTABLE_PARAMS = ["te", "td", "Ar", "Br", "Cr", "r_Br"]

    objective_function = SCMEObjectiveFunction(
        default_scme_params=DEFAULT_PARAMS,
        parametrization_key=parametrization_key,
        adjustable_params=ADJUSTABLE_PARAMS,
        paths_to_reference_configuration=paths_to_reference_configurations,
        reference_energies=reference_energies,
    )

    objective_function.dump_test_configurations("test_configurations_scme")

    fitter = Fitter(
        objective_function_cb=objective_function, n_contributions=n_contributions
    )

    initial_params = {k: dict(DEFAULT_PARAMS)[k] for k in ADJUSTABLE_PARAMS}

    optimal_params = fitter.fit_scipy(
        initial_parameters=initial_params, tol=0, options=dict(maxiter=10, disp=True)
    )

    print(f"{initial_params = }")
    print(f"{optimal_params = }")


if __name__ == "__main__":
    test_scme_fitting()
