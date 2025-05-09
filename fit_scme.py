from scme_fitting.fitter import Fitter
from scme_fitting.scme_setup import SCMEParams
from scme_fitting.scme_fitter import SCMEObjectiveFunction
import logging
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from pathlib import Path
import json
import pyscme
import numpy as np

logger = logging.getLogger(__name__)


def find_output_folder_that_does_not_exist(output_folder_base: str):
    i = 0
    output_folder = Path(f"{output_folder_base}_{i}")
    while output_folder.exists():
        i += 1
        output_folder = Path(f"{output_folder_base}_{i}")
    return output_folder


def dump_dict_to_file(file: Path, dictionary: dict):
    with open(file, "w") as f:
        json.dump(dict(dictionary), f, indent=4)


def run_scme_fitting(
    parametrization_key: str,
    adjustable_params: list[str],
    budget: int,
    name: str,
    default_params: SCMEParams,
    paths: list[Path],
    energies: list[float],
    tags: Optional[list[float]] = None,
    weights: Optional[list[float]] = None,
    output_folder: Optional[Path] = None,
):
    if output_folder is None:
        output_folder = find_output_folder_that_does_not_exist(f"./{name}")

    output_folder.mkdir(exist_ok=True)

    print(f"Output folder: {output_folder}")
    logger.info(f"Output folder: {output_folder}")

    objective_function = SCMEObjectiveFunction(
        default_scme_params=default_params,
        parametrization_key=parametrization_key,
        paths_to_reference_configuration=paths,
        reference_energies=energies,
        divide_by_n_atoms=True,
        tags=tags,
    )

    objective_function.dump_test_configurations(output_folder / "reference_configs")

    n_contributions = len(energies)
    fitter = Fitter(
        objective_function_cb=objective_function,
        n_contributions=n_contributions,
        weights=weights,
    )

    initial_params = {k: dict(default_params)[k] for k in adjustable_params}

    optimal_params = fitter.fit_scipy(
        initial_parameters=initial_params,
        tol=0,
        options=dict(maxiter=budget, disp=True),
    )

    meta = dict()
    meta["name"] = name
    meta["parametrization_key"] = parametrization_key
    meta["scme_version"] = {
        "branch": pyscme.version.branch(),
        "commit": pyscme.version.commit(),
        "date": pyscme.version.date(),
    }

    dump_dict_to_file(output_folder / "meta.json", meta)
    dump_dict_to_file(output_folder / "initial_params.json", initial_params)
    dump_dict_to_file(output_folder / "optimal_params.json", optimal_params)
    dump_dict_to_file(output_folder / "default_params.json", dict(default_params))

    np.savetxt(output_folder / "weights.txt", weights)

    energies_scme = {
        "tag": objective_function.tags,
        "initial": np.array(
            [
                objective_function.get_energy(i, initial_params)
                for i in range(n_contributions)
            ]
        ),
        "optimized": np.array(
            [
                objective_function.get_energy(i, optimal_params)
                for i in range(n_contributions)
            ]
        ),
        "n_atoms": np.array([len(a) for a in objective_function.atoms_list]),
    }

    energies_scme_df = pd.DataFrame(energies_scme)
    energies_scme_df.to_csv(output_folder / "energies_scme.csv")

    plt.close()
    ax = plt.gca()
    fig = plt.gcf()

    ax.plot(energies / energies_scme["n_atoms"], marker=".", label="reference")
    ax.plot(
        energies_scme["initial"] / energies_scme["n_atoms"],
        marker=".",
        label="initial parameters",
    )
    ax.plot(
        energies_scme["optimized"] / energies_scme["n_atoms"],
        marker=".",
        label="fitted parameters",
    )
    ax.set_xticks(range(len(energies)))
    ax.set_xticklabels(tags, rotation=90)
    ax.set_ylabel("energy [meV] / n_atoms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_folder / "plot.png", dpi=300)


def create_input_data(functional: str, dimer: bool, clusters: bool, ice: bool):

    def process_csv(path_to_csv: Path):

        path_to_csv = Path(path_to_csv)
        df = pd.read_csv(path_to_csv)
        paths = [path_to_csv.parent.resolve() / p for p in df["file"]]
        tags = list(df["tag"])
        energies = list(df["reference_energy"])

        weights = [1.0/len(energies) for _ in range(len(energies))]

        return paths, tags, energies, weights

    paths = []
    tags = []
    energies = []
    weights = []

    if dimer:
        path_to_csv = f"./{functional}_reference_configs/dimer/energies.csv"
        p, t, e, w = process_csv(path_to_csv)

        paths += p
        tags += t
        energies += e

        # For the dimer we use different weights
        w = [2.0 for i in range(len(e))]  # note this excludes the first configuration

        w[0] = 0
        w[-1] = 0  # we also exclude the last configuration since it's wonky

        weights += w

    if clusters:
        path_to_csv = f"./{functional}_reference_configs/clusters/energies.csv"
        p, t, e, w = process_csv(path_to_csv)

        paths += p
        tags += t
        energies += e
        weights += w

    if ice:
        path_to_csv = f"./{functional}_reference_configs/ice/energies.csv"
        p, t, e, w = process_csv(path_to_csv)

        paths += p
        tags += t
        energies += e
        weights += w

    return paths, tags, energies, weights


if __name__ == "__main__":
    logging.basicConfig(filename="fit_scme.log", level=logging.INFO)

    adjustable_params_disp = ["te", "td", "Ar", "Br", "Cr", "r_Br", "C6", "C8", "C10"]
    BUDGET = 100

    # Only optimize on the dimers
    # paths, tags, energies, weights = create_input_data(
    #     functional="pbe", dimer=True, clusters=False, ice=False
    # )
    # run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     adjustable_params=adjustable_params_disp,
    #     budget=BUDGET,
    #     name="generalized_dimer",
    #     default_params=SCMEParams(),
    #     paths=paths,
    #     tags=tags,
    #     energies=energies,
    #     weights=weights,
    # )

    # Only optimize on the clusters
    # paths, tags, energies, weights = create_input_data(
    #     functional="pbe", dimer=False, clusters=True, ice=False
    # )
    # run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     adjustable_params=adjustable_params_disp,
    #     budget=BUDGET,
    #     name="generalized_cluster",
    #     default_params=SCMEParams(),
    #     paths=paths,
    #     tags=tags,
    #     energies=energies,
    #     weights=weights,
    # )

    # Optimize on the clusters + dimer
    # paths, tags, energies, weights = create_input_data(
    #     functional="pbe", dimer=True, clusters=True, ice=False
    # )
    # run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     adjustable_params=adjustable_params_disp,
    #     budget=BUDGET,
    #     name="generalized_dimer_cluster",
    #     default_params=SCMEParams(),
    #     paths=paths,
    #     tags=tags,
    #     energies=energies,
    #     weights=weights,
    # )

    paths, tags, energies, weights = create_input_data(
        functional="pbe", dimer=False, clusters=False, ice=True
    )
    run_scme_fitting(
        parametrization_key="component_PBE_fullrange_reflect_8_12",
        adjustable_params=adjustable_params_disp,
        budget=10,
        name="generalized_dimer_cluster_ice",
        default_params=SCMEParams(),
        paths=paths,
        tags=tags,
        energies=energies,
        weights=weights,
    )

    # run_scme_fitting(
    #     parametrization_key=None,
    #     dimer_stretch=True,
    #     small_clusters=False,
    #     adjustable_params=adjustable_params,
    #     which="pbe",
    #     name="pbe_rigid_dimers",
    # )

    # run_scme_fitting(
    #     parametrization_key=None,
    #     dimer_stretch=True,
    #     small_clusters=False,
    #     adjustable_params=adjustable_params_disp,
    #     which="pbe",
    #     name="pbe_rigid_dimers_disp",
    # )

    # run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     dimer_stretch=True,
    #     small_clusters=False,
    #     adjustable_params=adjustable_params,
    #     which="pbe",
    #     name="pbe_generalized_8_12_dimers",
    # )

    # run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     dimer_stretch=True,
    #     small_clusters=False,
    #     adjustable_params=adjustable_params_disp,
    #     which="pbe",
    #     name="pbe_generalized_8_12_dimers_disp",
    # )

    # run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     dimer_stretch=True,
    #     small_clusters=True,
    #     adjustable_params=adjustable_params,
    #     which="pbe",
    #     name="pbe_generalized_8_12_dimers_and_clusters",
    # )

    # run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     dimer_stretch=True,
    #     small_clusters=True,
    #     ice=True,
    #     budget=0,
    #     adjustable_params=adjustable_params_disp,
    #     which="pbe",
    #     name="pbe_generalized_8_12_dimers_and_clusters_and_ice_adjust_disp",
    # )
