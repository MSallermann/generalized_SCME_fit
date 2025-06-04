from fit_scme import *

if __name__ == "__main__":
    logging.basicConfig(filename="fit_scme_blyp.log", level=logging.INFO)

    default_params = SCMEParams()

    path_to_scme_expansions = Path("./input/scme_expansions_BLYP.hdf5")
    parametrization_key = "component_BLYP_fullrange_reflect_8_12"

    adjustable_params = ["te", "td", "Ar", "Br", "Cr", "C6", "C8", "C10"]
    budget = 500

    #########################################
    #     Construct objective functions
    #########################################

    ###### DIMER ######
    dimer_csv = Path("./blyp_dimer_stretch.csv")

    dimer_objective_func = create_obj_funcs_from_csv(
        dimer_csv,
        default_params,
        parametrization_key,
        path_to_scme_expansions=path_to_scme_expansions,
    )

    dimer_objective_func.weights[0] = (
        0.0  # We exclude the very first point because it's weird
    )
    dimer_objective_func.weights[-1] = (
        0.0  # We exclude the very last point because it's weird
    )

    cluster_csv = Path("./blyp_small_clusters.csv")

    cluster_objective_func = create_obj_funcs_from_csv(
        cluster_csv,
        default_params,
        parametrization_key,
        path_to_scme_expansions=path_to_scme_expansions,
    )

    ###### Clusters ######
    cluster_csv = Path("./blyp_small_clusters.csv")

    cluster_objective_func = create_obj_funcs_from_csv(
        cluster_csv,
        default_params,
        parametrization_key,
        path_to_scme_expansions=path_to_scme_expansions,
    )

    name = "dimer_cluster_blyp_good_dimer"

    dimer_cluster_obj_func = combined_objective_functions_flat(
        [dimer_objective_func, cluster_objective_func],
        weights=[
            10.0 / dimer_objective_func.n_terms(),
            1.0 / cluster_objective_func.n_terms(),
        ],
    )

    optimal_params, initial_params, progress = run_scme_fitting(
        adjustable_params=adjustable_params,
        default_params=default_params,
        objective_function=dimer_cluster_obj_func,
        budget=budget,
        plot_path=f"progress_{name}.png",
    )

    write_output(
        name,
        parametrization_key=parametrization_key,
        objective_function=dimer_cluster_obj_func,
        initial_params=initial_params,
        optimal_params=optimal_params,
        default_params=default_params,
    )
