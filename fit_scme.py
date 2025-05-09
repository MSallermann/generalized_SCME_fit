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
    parametrization_key: str = "component_PBE_fullrange_reflect_8_12",
    which: str = "pbe",
    adjustable_params: list[str] = ["te", "td", "Ar", "Br", "Cr", "r_Br"],
    output_folder: Optional[Path] = None,
    default_params: SCMEParams = SCMEParams(),
    budget: int = 100,
    dimer_stretch: bool = True,
    small_clusters: bool = True,
    ice: bool = False,
    name: Optional[str] = None,
):
    if output_folder is None:
        output_folder = find_output_folder_that_does_not_exist(f"./{name}")

    output_folder.mkdir(exist_ok=True)

    print(f"Output folder: {output_folder}")
    logger.info(f"Output folder: {output_folder}")

    df_list = []
    weights = []

    # for the dimer stretch, we scale the contributions with the O-O distance
    # so that the short separations are weighted less
    if dimer_stretch:
        df_list.append(pd.read_csv(f"./{which}_dimer_stretch.csv"))
        weights += [float(i) ** 1 for i in range(len(df_list[0]))]

    if small_clusters:
        df_list.append(pd.read_csv(f"./{which}_small_clusters.csv"))
        weights += [1.0 for t in df_list[1]["tags"]]

    if ice:
        raise NotImplementedError()

    paths = []
    energies = []
    tags = []
    for df in df_list:
        paths += list(df["paths"])
        energies += list(df["energies"])
        tags += list(df["tags"])

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
    meta["scme_version"] = {
        "branch": pyscme.version.branch(),
        "commit": pyscme.version.commit(),
        "date": pyscme.version.date(),
    }

    dump_dict_to_file(output_folder / "meta.json", meta)
    dump_dict_to_file(output_folder / "initial_params.json", initial_params)
    dump_dict_to_file(output_folder / "optimal_params.json", optimal_params)
    dump_dict_to_file(output_folder / "default_params.json", dict(default_params))

    energies_scme = {
        "tag": objective_function.tags,
        "initial": [
            objective_function.get_energy(i, initial_params)
            for i in range(n_contributions)
        ],
        "optimized": [
            objective_function.get_energy(i, optimal_params)
            for i in range(n_contributions)
        ],
    }

    energies_scme_df = pd.DataFrame(energies_scme)
    energies_scme_df.to_csv(output_folder / "energies_scme.csv")

    plt.close()
    ax = plt.gca()
    fig = plt.gcf()

    ax.plot(energies, marker=".", label="reference")
    ax.plot(
        "initial",
        marker=".",
        label="initial parameters",
    )
    ax.plot(
        "optimized",
        marker=".",
        label="fitted parameters",
    )
    ax.set_xticks(range(len(energies)))
    ax.set_xticklabels(tags, rotation=90)
    ax.set_ylabel("energy [meV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_folder / "plot.png", dpi=300)


if __name__ == "__main__":
    logging.basicConfig(filename="fit_scme.log", level=logging.INFO)

    adjustable_params = ["te", "td", "Ar", "Br", "Cr", "r_Br"]
    adjustable_params_disp = ["te", "td", "Ar", "Br", "Cr", "r_Br", "C6", "C8", "C10"]

    run_scme_fitting(
        parametrization_key=None,
        dimer_stretch=True,
        small_clusters=False,
        adjustable_params=adjustable_params,
        which="pbe",
        name="pbe_rigid_dimers",
    )

    run_scme_fitting(
        parametrization_key=None,
        dimer_stretch=True,
        small_clusters=False,
        adjustable_params=adjustable_params_disp,
        which="pbe",
        name="pbe_rigid_dimers_disp",
    )

    run_scme_fitting(
        parametrization_key="component_PBE_fullrange_reflect_8_12",
        dimer_stretch=True,
        small_clusters=False,
        adjustable_params=adjustable_params,
        which="pbe",
        name="pbe_generalized_8_12_dimers",
    )

    run_scme_fitting(
        parametrization_key="component_PBE_fullrange_reflect_8_12",
        dimer_stretch=True,
        small_clusters=False,
        adjustable_params=adjustable_params_disp,
        which="pbe",
        name="pbe_generalized_8_12_dimers_disp",
    )

    run_scme_fitting(
        parametrization_key="component_PBE_fullrange_reflect_8_12",
        dimer_stretch=True,
        small_clusters=True,
        adjustable_params=adjustable_params,
        which="pbe",
        name="pbe_generalized_8_12_dimers_and_clusters",
    )

    run_scme_fitting(
        parametrization_key="component_PBE_fullrange_reflect_8_12",
        dimer_stretch=True,
        small_clusters=True,
        adjustable_params=adjustable_params_disp,
        which="pbe",
        name="pbe_generalized_8_12_dimers_and_clusters_adjust_disp",
    )
