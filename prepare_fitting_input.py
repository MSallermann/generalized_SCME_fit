from pathlib import Path
import numpy as np
import pandas as pd


def assert_exists(path: Path):
    if not path.exists():
        raise Exception(f"{path = } does not exist")


def get_dimer_stretch_data(base_path: Path):
    assert_exists(base_path)

    energies_txt_file = list((base_path).glob("PES*.txt"))[0]

    energies = np.loadtxt(energies_txt_file)[:, 1]

    paths = list(base_path.glob("*/CONTCAR"))
    sorted_paths = sorted(paths, key=lambda p: float(p.parent.name))
    tags = [f"dimer_C1_{p.parent.name}" for p in sorted_paths]

    return {"paths": sorted_paths, "energies": energies, "tags": tags}


def get_scmall_clusters(base_path: Path, functional_str: str):
    assert_exists(base_path)

    paths = list(base_path.glob(f"*/*/{functional_str}/CONTCAR"))
    energies = []
    tags = []

    for p in paths:
        cluster_name = p.parts[-4]  # Dimer, Hexamer, etc
        configuration_name = p.parts[-3]  # C1, C2v, Ci etc

        energy_file_path = list(
            p.parent.parent.parent.glob(f"E_*{functional_str}*.txt")
        )[0]

        df = pd.read_csv(energy_file_path, delimiter=" ", skiprows=1, header=None)
        n_cols = len(df.columns)

        names = [f"col{i + 1}" for i in range(n_cols)]
        names[:4] = ["cluster", "dE/n_mon", "dE", "E_cluster"]
        df.columns = names

        energy = np.array(df["dE"][df["cluster"] == configuration_name])[0]

        tags.append(f"{cluster_name}_{configuration_name}")
        energies.append(energy)

    return {"paths": paths, "energies": energies, "tags": tags}


def get_ice(base_path: Path, functional_str: str):
    from ase.io import read, write

    assert_exists(base_path)

    path_to_xyz = list((base_path / functional_str).glob("ice-Ih*.xyz"))[0]

    atoms_list = read(path_to_xyz, index=":")

    [write(path_to_xyz.parent / f"ice_{i}.xyz", a) for i, a in enumerate(atoms_list)]
    paths = [path_to_xyz.parent / f"ice_{i}.xyz" for i, a in enumerate(atoms_list)]

    e_v_data = np.loadtxt(base_path / functional_str / "E_V.data")

    tags = [f"Ice_rescaled_{p}_percent" for p in e_v_data[:, 0]]
    energies = e_v_data[:, 1]

    return {"paths": paths, "energies": energies, "tags": tags}


path_to_scme_input = Path(
    "/home/moritz/SCME/generalized_SCME_interatomic_fit/scme_input"
)

path_to_dimer_stretch = path_to_scme_input / "Intermolecular/PES/Dimer/C1"
dimer_stretch_PBE = get_dimer_stretch_data(path_to_dimer_stretch / "PBE")
df_pbe_dimer_stretch = pd.DataFrame(dimer_stretch_PBE)
df_pbe_dimer_stretch.to_csv("pbe_dimer_stretch.csv")

dimer_stretch_beef = get_dimer_stretch_data(path_to_dimer_stretch / "BEEF-vdW")
df_beef_dimer_stretch = pd.DataFrame(dimer_stretch_beef)
df_beef_dimer_stretch.to_csv("beef_dimer_stretch.csv")


path_to_small_clusters = path_to_scme_input / "Intermolecular/Clusters/Small"
small_clusters_PBE = get_scmall_clusters(path_to_small_clusters, functional_str="PBE")
df_small_clusters_PBE = pd.DataFrame(small_clusters_PBE)
df_small_clusters_PBE.to_csv("pbe_small_clusters.csv")

small_clusters_beef = get_scmall_clusters(
    path_to_small_clusters, functional_str="BEEF-vdW"
)
df_small_clusters_beef = pd.DataFrame(small_clusters_beef)
df_small_clusters_beef.to_csv("beef_small_clusters.csv")

path_to_ice = path_to_scme_input / "Intermolecular/Crystals/Ice-IH"
ice_pbe = get_ice(path_to_ice, functional_str="PBE")
df_ice_pbe = pd.DataFrame(ice_pbe)
df_ice_pbe.to_csv("pbe_ice.csv")

ice_beef = get_ice(path_to_ice, functional_str="BEEF-vdW")
df_ice_beef = pd.DataFrame(ice_beef)
df_ice_beef.to_csv("beef_ice.csv")
