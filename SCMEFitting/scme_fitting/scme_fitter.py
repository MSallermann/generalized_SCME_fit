from .scme_setup import setup_calculator, SCMEParams
from ase import Atoms, Atom
from ase.io import read, write
from ase.geometry import find_mic
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SCMEObjectiveFunction:
    def __init__(
        self,
        default_scme_params: SCMEParams,
        parametrization_key: str,
        adjustable_params: list[str],
        paths_to_reference_configuration: list[Path],
        reference_energies: list[float],
    ):
        self.default_scme_params = default_scme_params
        self.parametrization_key = parametrization_key
        self.adjustable_params = adjustable_params

        # Make sure that we have a reference energy for each configuration
        assert len(paths_to_reference_configuration) == len(reference_energies)

        self.paths_to_reference_configuration = paths_to_reference_configuration
        self.reference_energies = reference_energies

        self.atoms_list = self.create_list_of_atom_objects()

    def assure_params(self, parameters: dict):
        """Make sure that all the necessary keys are in our parameter dict"""
        for k in self.adjustable_params:
            if k not in parameters:
                raise Exception(f"Could not find key {k} in `parameters`")

    def arange_water_in_OHH_order(self, atoms: Atoms):
        """Takes an atoms object and re-arranges it in the OHH order that the SCME expects"""
        n_atoms = len(atoms)

        # Some asserts
        assert n_atoms % 3 == 0
        mask_O = atoms.numbers == 8
        mask_H = atoms.numbers == 1
        assert 2 * sum(mask_O) == sum(mask_H)

        # Now we create a list of new positions
        new_atoms = []
        for atom_O in atoms[mask_O]:
            assert atom_O.number == 8
            new_atoms.append(atom_O)

            # sort the hydrogens by ascending distance from the current oxygen
            H_sorted = sorted(
                atoms[mask_H],
                key=lambda a: find_mic(atom_O.position - a.position, cell=atoms.cell)[
                    1
                ],
            )

            new_atoms.append(H_sorted[0])
            new_atoms.append(H_sorted[1])

        result = atoms.copy()
        result.set_constraint() # Make sure to explicitly delete any constraints
        result.set_atomic_numbers([a.number for a in new_atoms])
        result.set_positions([a.position for a in new_atoms])

        return result

    def check_water_is_in_OHH_order(self, atoms: Atoms, OH_distance_tol: float = 2.0):
        """Asserts that an atoms object contains water in the OHH order"""

        n_atoms = len(atoms)

        assert n_atoms % 3 == 0

        n_water = n_atoms // 3

        for iwater in range(n_water):
            idxO = 3 * iwater
            idxH1 = idxO + 1
            idxH2 = idxO + 2

            assert atoms.numbers[idxO] == 8
            assert atoms.numbers[idxH1] == 1
            assert atoms.numbers[idxH2] == 1

            OH_dist1 = atoms.get_distance(idxO, idxH1, mic=True)
            OH_dist2 = atoms.get_distance(idxO, idxH2, mic=True)

            if OH_dist1 > OH_distance_tol or OH_dist2 > OH_distance_tol:
                raise Exception(f"OH_distance too big for idx {idxO} ({OH_dist1}, {OH_dist2})")

    def dump_test_configurations(self, path_to_folder: Path):
        path_to_folder = Path(path_to_folder)

        path_to_folder.mkdir(exist_ok=True, parents=True)

        for i, atoms in enumerate(self.atoms_list):
            write(path_to_folder / f"atoms_{i}.xyz", atoms)

        np.savetxt(path_to_folder / "energies.txt", self.reference_energies)

    def create_atoms_object_from_configuration(self, path_to_configuration: Path):
        logger.debug(
            f"creating atoms object from path {path_to_configuration}"
        )

        atoms = read(path_to_configuration)
        atoms = self.arange_water_in_OHH_order(atoms)
        self.check_water_is_in_OHH_order(atoms)

        scme_params = self.default_scme_params.copy()

        setup_calculator(
            atoms, scme_params=scme_params, parametrization_key=self.parametrization_key
        )

        return atoms

    def create_list_of_atom_objects(self):
        atoms_list = []
        for p in self.paths_to_reference_configuration:
            atoms_list.append(self.create_atoms_object_from_configuration(p))
        return atoms_list

    def __call__(self, idx: int, parameters: dict):
        self.assure_params(parameters)

        atoms = self.atoms_list[idx]
        n_atoms = len(atoms)
        target_energy = self.reference_energies[idx]

        ## Update the params of the calculator
        for k, v in parameters.items():
            logger.debug(f"Updating {k}")
            logger.debug(f"  Prev value = {getattr(atoms.calc.scme, k)}")
            logger.debug(f"  New value = {v}")
            atoms.calc.scme.__setattr__(k, v)

        # We have to make sure to trigger the update of the energy manually,
        # because ase will think it can use the cached energy values,
        # since none of the coordinates has changed
        atoms.calc.calculate(atoms)

        # Retrieve the potential energy
        energy = atoms.get_potential_energy()

        logger.debug(f"Calculated energy: {energy}")

        logger.debug(f"  {atoms.calc.energy_electrostatic = }")
        logger.debug(f"  {atoms.calc.energy_dispersion = }")
        logger.debug(f"  {atoms.calc.energy_core = }")
        logger.debug(f"  {atoms.calc.energy_monomer = }")

        objective_function_contribution = (energy - target_energy) ** 2 / n_atoms

        logger.debug(f"Objective function value = {objective_function_contribution}")

        return objective_function_contribution
