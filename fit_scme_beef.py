from fit_scme import *

if __name__ == "__main__":
    logging.basicConfig(filename="fit_scme_beef.log", level=logging.INFO)

    default_params = SCMEParams()

    path_to_scme_expansions = Path("./input/scme_expansions_BEEF.hdf5")
    parametrization_key = "component_BEEF_fullrange_reflect_8_12"

    adjustable_params = ["te", "td", "Ar", "Br", "Cr", "r_Br", "C6", "C8", "C10"]
    budget = 200

    # Construct objective functions
    # dimer
    # dimer_csv = Path("./pbe_reference_configs/dimer/energies.csv")
    dimer_csv = Path("./beef_dimer_stretch.csv")
    dimer_objective_func = create_obj_funcs_from_csv(
        dimer_csv,
        default_params,
        parametrization_key,
        path_to_scme_expansions=path_to_scme_expansions,
    )
    # dimer_objective_func.weights[0] *= 10
    # dimer_objective_func.weights[1] *= 10
    # dimer_objective_func.weights[-1] = (
    #     0.0  # We exclude the very last point because it's weird
    # )

    # clusters
    cluster_csv = Path("./pbe_reference_configs/clusters/energies.csv")
    cluster_csv = Path("./beef_small_clusters.csv")
    cluster_objective_func = create_obj_funcs_from_csv(
        cluster_csv,
        default_params,
        parametrization_key,
        path_to_scme_expansions=path_to_scme_expansions,
    )

    # ice
    ice_csv = Path("./pbe_reference_configs/ice/energies.csv")
    ice_csv = Path("./beef_ice.csv")
    __ice_objective_func = create_obj_funcs_from_csv(
        ice_csv,
        default_params,
        parametrization_key,
        path_to_scme_expansions=path_to_scme_expansions,
    )

    # Let's only take the smallest and the largest ice
    ice_objective_func = CombinedObjectiveFunction(
        [
            __ice_objective_func.objective_functions[0],
            __ice_objective_func.objective_functions[-1],
        ]
    )

    ##### DIMER only #####
    # name = "dimer_only_beef"

    # optimal_params, initial_params, progress = run_scme_fitting(
    #     adjustable_params=adjustable_params,
    #     default_params=default_params,
    #     objective_function=dimer_objective_func,
    #     budget=budget,
    #     plot_path=f"progress_{name}.png",
    # )

    # write_output(
    #     name,
    #     parametrization_key=parametrization_key,
    #     objective_function=dimer_objective_func,
    #     initial_params=initial_params,
    #     optimal_params=optimal_params,
    #     default_params=default_params,
    # )

    # ##### Cluster only #####
    # name = "cluster_only_beef"

    # optimal_params, initial_params, progress = run_scme_fitting(
    #     adjustable_params=adjustable_params,
    #     default_params=default_params,
    #     objective_function=cluster_objective_func,
    #     budget=budget,
    #     plot_path=f"progress_{name}.png",
    # )

    # write_output(
    #     name,
    #     parametrization_key=parametrization_key,
    #     objective_function=cluster_objective_func,
    #     initial_params=initial_params,
    #     optimal_params=optimal_params,
    #     default_params=default_params,
    # )

    #### Cluster + Dimer #####
    name = "dimer_cluster_beef"

    dimer_cluster_obj_func = combined_objective_functions_flat(
        [dimer_objective_func, cluster_objective_func], weights=[1.0, 0.5]
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

    ##### Cluster + Dimer + Ice #####

    name = "dimer_cluster_ice_beef"

    dimer_cluster_ice_obj_func = combined_objective_functions_flat(
        [dimer_objective_func, cluster_objective_func, ice_objective_func],
        weights=[1.0, 0.5, 0.1],
    )

    optimal_params, initial_params, progress = run_scme_fitting(
        adjustable_params=adjustable_params,
        default_params=default_params,
        objective_function=dimer_cluster_ice_obj_func,
        budget=500,
        plot_path=f"progress_{name}.png",
    )

    write_output(
        name,
        parametrization_key=parametrization_key,
        objective_function=dimer_cluster_ice_obj_func,
        initial_params=initial_params,
        optimal_params=optimal_params,
        default_params=default_params,
    )
