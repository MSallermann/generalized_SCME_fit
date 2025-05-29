from pathlib import Path
import numpy as np
import pandas as pd


def assert_exists(path: Path):
    if not path.exists():
        raise Exception(f"{path = } does not exist")


def get_dimer_stretch_data(base_path: Path, E_monomer: float):
    assert_exists(base_path)

    energies_txt_file = list((base_path).glob("PES*.txt"))[0]

    total_energies = np.loadtxt(energies_txt_file)[:, 2]
    n_monomers = 2

    binding_energies = total_energies - n_monomers * E_monomer

    paths = list(base_path.glob("*/CONTCAR"))
    sorted_paths = sorted(paths, key=lambda p: float(p.parent.name))
    tags = [f"dimer_C1_{p.parent.name}" for p in sorted_paths]

    return {"path": sorted_paths, "reference_energy": binding_energies, "tag": tags}


def get_small_clusters(base_path: Path, functional_str: str, E_monomer: float):
    assert_exists(base_path)

    paths = list(base_path.glob(f"*/*/{functional_str}/CONTCAR"))
    energies = []
    tags = []
    n_monomers_list = []

    for p in paths:
        cluster_name = p.parts[-4]  # Dimer, Hexamer, etc
        configuration_name = p.parts[-3]  # C1, C2v, Ci etc

        energy_file_path = (
            p.parent.parent.parent / f"E_{cluster_name.lower()}_{functional_str}.txt"
        )

        assert_exists(energy_file_path)

        df = pd.read_csv(energy_file_path, delimiter=" ", skiprows=1, header=None)
        n_cols = len(df.columns)

        names = [f"col{i + 1}" for i in range(n_cols)]
        names[:4] = ["cluster", "dE/n_mon", "dE", "E_cluster"]

        df.columns = names

        n_monomers = len(names) - 4
        n_monomers_list.append(n_monomers)

        mask = df["cluster"] == configuration_name

        total_energy = np.array(df["dE"][mask])[0]
        binding_energy = total_energy - n_monomers * E_monomer

        tags.append(f"{cluster_name}_{configuration_name}")
        energies.append(binding_energy)

    return {
        "path": paths,
        "reference_energy": energies,
        "tag": tags,
        "n_monomers": n_monomers_list,
    }


def get_ice(base_path: Path, functional_str: str, E_monomer: float):
    from ase.io import read, write

    assert_exists(base_path)

    path_to_xyz = list((base_path / functional_str).glob("ice-Ih*.xyz"))[0]

    atoms_list = read(path_to_xyz, index=":")

    [write(path_to_xyz.parent / f"ice_{i}.xyz", a) for i, a in enumerate(atoms_list)]
    paths = [path_to_xyz.parent / f"ice_{i}.xyz" for i, a in enumerate(atoms_list)]

    e_v_data = np.loadtxt(base_path / functional_str / "E_V.data")

    tags = [f"Ice_volume_rescaled_{p}_percent" for p in e_v_data[:, 0]]

    total_energies = e_v_data[:, 2]
    n_monomers = 96

    binding_energies = total_energies - n_monomers * E_monomer

    return {"path": paths, "reference_energy": binding_energies, "tag": tags}


### Monomer energies
E_mon_PBE = -0.14217301e02
E_mon_BEEF_vdW = -0.12808179e02
E_mon_BLYP = -13.9499590600
E_mon_RPBE = -14.1550696200

path_to_scme_input = Path(
    "/home/moritz/SCME/generalized_SCME_interatomic_fit/scme_input"
)


#### Dimer stretch
path_to_dimer_stretch = path_to_scme_input / "Intermolecular/PES/Dimer/C1"

####### PBE
dimer_stretch_PBE = get_dimer_stretch_data(
    path_to_dimer_stretch / "PBE", E_monomer=E_mon_PBE
)
df_pbe_dimer_stretch = pd.DataFrame(dimer_stretch_PBE)
df_pbe_dimer_stretch.to_csv("pbe_dimer_stretch.csv")


####### BEEF
dimer_stretch_beef = get_dimer_stretch_data(
    path_to_dimer_stretch / "BEEF-vdW", E_monomer=E_mon_BEEF_vdW
)
df_beef_dimer_stretch = pd.DataFrame(dimer_stretch_beef)
df_beef_dimer_stretch.to_csv("beef_dimer_stretch.csv")

#### Small clusters
path_to_small_clusters = path_to_scme_input / "Intermolecular/Clusters/Small"

####### PBE
small_clusters_PBE = get_small_clusters(
    path_to_small_clusters, functional_str="PBE", E_monomer=E_mon_PBE
)
df_small_clusters_PBE = pd.DataFrame(small_clusters_PBE)
df_small_clusters_PBE.to_csv("pbe_small_clusters.csv")

####### BEEF
small_clusters_beef = get_small_clusters(
    path_to_small_clusters, functional_str="BEEF-vdW", E_monomer=E_mon_BEEF_vdW
)
df_small_clusters_beef = pd.DataFrame(small_clusters_beef)
df_small_clusters_beef.to_csv("beef_small_clusters.csv")

#### ICE
path_to_ice = path_to_scme_input / "Intermolecular/Crystals/Ice-IH"
####### PBE
ice_pbe = get_ice(path_to_ice, functional_str="PBE", E_monomer=E_mon_PBE)
df_ice_pbe = pd.DataFrame(ice_pbe)
df_ice_pbe.to_csv("pbe_ice.csv")

####### BEEF
ice_beef = get_ice(path_to_ice, functional_str="BEEF-vdW", E_monomer=E_mon_BEEF_vdW)
df_ice_beef = pd.DataFrame(ice_beef)
df_ice_beef.to_csv("beef_ice.csv")
