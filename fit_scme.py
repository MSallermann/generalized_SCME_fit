from scme_fitting.fitter import Fitter
from scme_fitting.scme_setup import SCMEParams
from scme_fitting.scme_objective_function import SCMEObjectiveFunction
from scme_fitting.combined_objective_function import CombinedObjectiveFunction


import logging
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union, Dict, Callable
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
    adjustable_params: Union[list[str], Dict[str, float]],
    default_params: SCMEParams,
    objective_function: Callable,
    budget: int,
):
    fitter = Fitter(
        objective_function=objective_function,
    )

    try:
        initial_params = {k: v for k, v in adjustable_params.items()}
    except:
        initial_params = {k: dict(default_params)[k] for k in adjustable_params}

    optimal_params = fitter.fit_scipy(
        initial_parameters=initial_params,
        tol=0,
        options=dict(maxiter=budget, disp=True),
    )

    return optimal_params, initial_params


def make_plots(
    output_folder: Path,
    energies_scme_df: pd.DataFrame,
):
    tags = energies_scme_df["tag"]
    energy_initial = energies_scme_df["energy_initial"]
    energy_fitted = energies_scme_df["energy_fitted"]
    energy_reference = energies_scme_df["energy_reference"]
    n_atoms = energies_scme_df["n_atoms"]

    plt.close()
    ax = plt.gca()
    fig = plt.gcf()

    ax.plot(
        tags, energy_reference / n_atoms, marker="o", color="black", label="reference"
    )
    ax.plot(tags, energy_fitted / n_atoms, marker="x", label="fitted")
    ax.plot(tags, energy_initial / n_atoms, marker=".", label="initial")
    ax.set_xticks(range(len(energy_reference)))
    ax.set_xticklabels(tags, rotation=90)
    ax.set_ylabel("energy [meV] / n_atoms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_folder / "plot_energy.png", dpi=300)


def write_output(
    name: str,
    parametrization_key: str,
    objective_function: CombinedObjectiveFunction,
    initial_params: Dict,
    optimal_params: Dict,
    default_params: SCMEParams,
    output_folder: Optional[Path] = None,
):
    if output_folder is None:
        output_folder = find_output_folder_that_does_not_exist(f"./{name}")
    else:
        output_folder = find_output_folder_that_does_not_exist(
            output_folder / f"{name}"
        )

    output_folder.mkdir(exist_ok=True)

    print(f"Output folder: {output_folder}")
    logger.info(f"Output folder: {output_folder}")

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

    objective_function_list = objective_function.objective_functions

    for o in objective_function_list:
        o.dump_test_configuration(output_folder / "reference_configs")

    weights_energy = [ob.weight for ob in objective_function_list]
    weights_combination = objective_function.weights
    ob_value = [ob(optimal_params) for ob in objective_function_list]

    weights_total = [w1 * w2 for w1, w2 in zip(weights_energy, weights_combination)]

    energies_scme = {
        "tag": [ob.tag for ob in objective_function_list],
        "energy_initial": [
            ob.get_energy(initial_params) for ob in objective_function_list
        ],
        "energy_fitted": [
            ob.get_energy(optimal_params) for ob in objective_function_list
        ],
        "energy_reference": [ob.reference_energy for ob in objective_function_list],
        "n_atoms": [ob.n_atoms for ob in objective_function_list],
        "weight_energy": [w for w in weights_energy],
        "weight_combination": [w for w in weights_combination],
        "weight": [w for w in weights_total],
        "ob_value": ob_value,
    }

    energies_scme_df = pd.DataFrame(energies_scme)

    energies_scme_df.to_csv(output_folder / "energies_scme.csv")

    make_plots(
        output_folder=output_folder,
        energies_scme_df=energies_scme_df,
    )


def process_csv(path_to_csv: Path):
    path_to_csv = Path(path_to_csv)
    df = pd.read_csv(path_to_csv)
    paths = [path_to_csv.parent.resolve() / p for p in df["file"]]
    tags = list(df["tag"])
    energies = list(df["reference_energy"])
    return paths, tags, energies


def create_obj_funcs_from_csv(
    path_to_csv: Path, default_params: SCMEParams, parametrization_key: str
) -> CombinedObjectiveFunction:
    paths, tags, energies = process_csv(path_to_csv)

    obj_func_list = [
        SCMEObjectiveFunction(
            default_scme_params=default_params,
            parametrization_key=parametrization_key,
            path_to_reference_configuration=xyz_file,
            reference_energy=energy,
            divide_by_n_atoms=True,
            tag=tag,
        )
        for xyz_file, tag, energy in zip(paths, tags, energies)
    ]

    return CombinedObjectiveFunction(obj_func_list)


if __name__ == "__main__":
    logging.basicConfig(filename="fit_scme.log", level=logging.INFO)

    default_params = SCMEParams()
    parametrization_key = "component_PBE_fullrange_reflect_8_12"
    adjustable_params = ["te", "td", "Ar", "Br", "Cr", "r_Br", "C6", "C8", "C10"]
    budget = 10

    dimer_csv = Path("./pbe_reference_configs/dimer/energies.csv")

    dimer_objective_func = create_obj_funcs_from_csv(
        dimer_csv, default_params, parametrization_key
    )

    name = "dimer_stretch"

    # Only optimize on the dimers
    optimal_params, initial_params = run_scme_fitting(
        adjustable_params=adjustable_params,
        default_params=default_params,
        objective_function=dimer_objective_func,
        budget=budget,
    )

    write_output(
        name,
        parametrization_key=parametrization_key,
        objective_function=dimer_objective_func,
        initial_params=initial_params,
        optimal_params=optimal_params,
        default_params=default_params,
    )

    # adjustable_params = optimal_params

    # # Only optimize on the clusters
    # paths, tags, energies, weights = create_input_data(
    #     functional="pbe", dimer=False, clusters=True, ice=False
    # )
    # optimal_params = run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     adjustable_params=adjustable_params,
    #     budget=BUDGET,
    #     name="generalized_cluster",
    #     default_params=SCMEParams(),
    #     paths=paths,
    #     tags=tags,
    #     energies=energies,
    #     # weights=weights,
    # )
    # adjustable_params = optimal_params

    # # Optimize on the clusters + dimer
    # paths, tags, energies, weights = create_input_data(
    #     functional="pbe", dimer=True, clusters=True, ice=False
    # )
    # run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     adjustable_params=adjustable_params,
    #     budget=BUDGET,
    #     name="generalized_dimer_cluster",
    #     default_params=SCMEParams(),
    #     paths=paths,
    #     tags=tags,
    #     energies=energies,
    #     # weights=weights,
    # )

    # paths, tags, energies, weights = create_input_data(
    #     functional="pbe", dimer=False, clusters=False, ice=True
    # )
    # run_scme_fitting(
    #     parametrization_key="component_PBE_fullrange_reflect_8_12",
    #     adjustable_params=adjustable_params_disp,
    #     budget=10,
    #     name="generalized_dimer_cluster_ice",
    #     default_params=SCMEParams(),
    #     paths=paths,
    #     tags=tags,
    #     energies=energies,
    #     weights=weights,
    # )
