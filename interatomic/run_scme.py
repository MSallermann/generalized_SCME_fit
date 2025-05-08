import util
from ase.io import write, read
from ase.optimize import FIRE
from pathlib import Path
from typing import Optional
import numpy as np


def relax_dimer(
    oo_distance: float,
    scme_params: util.SCMEParams,
    parametrization_key: Optional[str],
    initial_xyz_path: Optional[Path] = None,
    output_xyz_path: Optional[Path] = None,
    fmax: float = 1e-4,
    dt: float = 1e-4,
    max_steps: int = 1000,
) -> float:
    if initial_xyz_path is not None:
        atoms = read(initial_xyz_path)
        util.move_dimer_apart(atoms, target_oo_distance=oo_distance)
    else:
        atoms = util.setup_dimer(oo_distance=oo_distance)

    util.constrain_dimer(atoms)

    atoms.calc = util.setup_calculator(
        atoms=atoms, scme_params=scme_params, parametrization_key=parametrization_key
    )

    dyn = FIRE(atoms, dt=dt)
    dyn.run(fmax=fmax, steps=max_steps)

    if output_xyz_path is not None:
        write(output_xyz_path, atoms)

    energy = atoms.get_potential_energy()

    return energy


def get_dimer_binding_curve(
    oo_distances: list[float],
    output_folder: Path,
    initial_xyz_path: Path,
    scme_params: util.SCMEParams,
    parametrization_key: Optional[str],
) -> list[float]:
    output_folder.mkdir(exist_ok=True, parents=True)

    energies = []

    for dist in oo_distances:
        energy = relax_dimer(
            dist,
            output_xyz_path=output_folder / f"dimer_{dist:.1f}.xyz",
            initial_xyz_path=initial_xyz_path,
            scme_params=scme_params,
            parametrization_key=parametrization_key,
        )
        energies.append(energy)

    result = np.vstack((oo_distances, energies)).T

    np.savetxt(
        output_folder / "results.txt", result, header="distance [angstrom], energy [eV]"
    )

    plt.plot(oo_distances, energies)
    plt.xlabel("O-O distance [angstrom]")
    plt.ylabel("Energy [eV]")
    plt.savefig(output_folder / "curve.png", dpi=200)
    plt.close()

    return energies


if __name__ == "__main__":
    import logging
    import matplotlib.pyplot as plt

    logging.basicConfig(filename="test.log", level=logging.INFO)

    oo_distances = np.linspace(2.3, 5.0, 28, endpoint=True)

    scme_params = util.DEFAULT_PARAMETERS.copy()

    binding_energies = get_dimer_binding_curve(
        oo_distances,
        scme_params=scme_params,
        output_folder=Path("./binding_curve"),
        initial_xyz_path=Path("./initial_dimer.xyz"),
        parametrization_key="component_PBE_fullrange_reflect_4_5",
    )
