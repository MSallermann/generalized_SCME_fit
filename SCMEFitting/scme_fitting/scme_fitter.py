from .scme_setup import setup_calculator, SCMEParams
from ase.io import read
from pathlib import Path

import logging

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
        for k in self.adjustable_params:
            if k not in parameters:
                raise Exception(f"Could not find key {k} in `parameters`")

    def create_atoms_object_from_configuration(self, path_to_configuration: Path):
        atoms = read(path_to_configuration)
        n_atoms = len(atoms)

        logger.debug(
            f"creating atoms object from path {path_to_configuration}. n_atoms is {n_atoms}"
        )

        scme_params = self.default_scme_params.copy()

        setup_calculator(
            atoms, scme_params=scme_params, parametrization_key=self.parametrization_key
        )

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
            logger.debug(f"Setting attr {k} = {v}")
            atoms.calc.__setattr__(k, v)

        # Compute the energy
        energy = atoms.get_potential_energy()
        logger.debug(f"Calculated energy: {energy}")

        objective_function_contribution = (energy - target_energy) ** 2 / n_atoms

        logger.debug(f"Objective function value = {objective_function_contribution}")

        return objective_function_contribution
