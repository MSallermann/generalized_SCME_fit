import util
from ase.io import write
from ase.optimize import FIRE, FIRE2
from pathlib import Path


def relax_dimer(
    oo_distance: float,
    scme_params=util.DEFAULT_PARAMS,
    output_xyz_path: Path = "dimer.xyz",
    fmax: float = 1e-4,
    dt: float = 1e-9,
    max_steps: int = 1000,
):
    atoms = util.setup_dimer(oo_distance=oo_distance)
    util.constrain_dimer(atoms)

    # atoms = util.setup_monomer()

    atoms.calc = util.setup_calculator(atoms=atoms, scme_params=scme_params)

    dyn = FIRE(atoms, dt=dt)
    dyn.run(fmax=fmax, steps=max_steps)

    write(output_xyz_path, atoms)


if __name__ == "__main__":
    import logging

    logging.basicConfig(filename="test.log", level=logging.INFO)
    relax_dimer(5.0, max_steps=1000)
