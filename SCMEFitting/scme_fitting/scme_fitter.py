from .scme_setup import setup_calculator, SCMEParams
from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic
from pathlib import Path
import logging
from typing import Optional

import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)


class SCMEObjectiveFunction:
    def __init__(
        self,
        default_scme_params: SCMEParams,
        parametrization_key: str,
        paths_to_reference_configuration: list[Path],
        reference_energies: list[float],
        divide_by_n_atoms: bool = False,
        tags: Optional[list[str]] = None,
    ):
        """
        Functor for computing squared-error energy contributions using SCME.

        This class:
        1. Loads reference configurations from file paths.
        2. Reorders water molecules into OHH order as required by SCME.
        3. Attaches an ASE SCME calculator with default parameters.
        4. Computes per-configuration energy and returns (energy - reference)^2.

        Parameters
        ----------
        default_scme_params : SCMEParams
            Default SCME parameter object to copy for each configuration.
        parametrization_key : str
            Key selecting the parametrization for the moment and monomer energy expansion.
        paths_to_reference_configuration : Sequence[Path]
            File paths for each reference configuration (xyz files).
        reference_energies : Sequence[float]
            Target energies corresponding to each reference configuration.
        divide_by_n_atoms : bool = False
            Wether to divide the objective function by n_atoms. Leading to contributions
            of the form: (energy - reference)^2 / n_atoms
        tags: Optional[Sequence[str]] = None
            Optional tags for each configuration (used for logging and output purposes)

        Raises
        ------
        ValueError
            If lengths of `paths_to_reference_configuration` and
            `reference_energies` do not match.
        """

        n_contributions = len(reference_energies)

        if len(paths_to_reference_configuration) != n_contributions:
            raise ValueError(
                f"Mismatch: {len(paths_to_reference_configuration)} paths vs. "
                f"{len(reference_energies)} energies"
            )

        self.divide_by_n_atoms = divide_by_n_atoms

        if tags is None:
            self.tags = [""] * n_contributions
        else:
            if len(tags) != n_contributions:
                raise ValueError(
                    f"Mismatch: {len(tags)} tags vs. {len(reference_energies)} energies"
                )
            self.tags = tags

        self.default_scme_params: SCMEParams = default_scme_params
        self.parametrization_key: str = parametrization_key
        self.paths_to_reference_configuration: List[Path] = list(
            paths_to_reference_configuration
        )
        self.reference_energies: List[float] = list(reference_energies)

        self.atoms_list: List[Atoms] = self._create_list_of_atom_objects()

    def arange_water_in_OHH_order(self, atoms: Atoms) -> Atoms:
        """
        Reorder atoms so each water molecule appears as O, H, H.

        Parameters
        ----------
        atoms : Atoms
            Original Atoms object containing water molecules.

        Returns
        -------
        Atoms
            New Atoms object with OHH ordering and no constraints.

        Raises
        ------
        ValueError
            If atom counts or ratios are inconsistent with water.
        """
        n_atoms = len(atoms)
        if n_atoms % 3 != 0:
            raise ValueError(f"Number of atoms {n_atoms} is not a multiple of 3")

        mask_O = atoms.numbers == 8
        mask_H = atoms.numbers == 1
        if 2 * mask_O.sum() != mask_H.sum():
            raise ValueError("Mismatch between O and H counts for water molecules")

        new_order: List[Atoms] = []
        for atom_O in atoms[mask_O]:
            new_order.append(atom_O)
            H_sorted = sorted(
                atoms[mask_H],
                key=lambda a: find_mic(atom_O.position - a.position, cell=atoms.cell)[
                    1
                ],
            )
            new_order.extend(H_sorted[:2])

        result = atoms.copy()
        result.set_constraint()
        result.set_atomic_numbers([a.number for a in new_order])
        result.set_positions([a.position for a in new_order])
        return result

    def check_water_is_in_OHH_order(
        self, atoms: Atoms, OH_distance_tol: float = 2.0
    ) -> bool:
        """
        Validate that each water molecule is ordered O, H, H and within tolerance.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to validate.
        OH_distance_tol : float, optional
            Maximum allowed O-H distance (default is 2.0 Å).

        Raises
        ------
        ValueError
            If ordering or distances violate water OHH assumptions.
        """
        n_atoms = len(atoms)
        if n_atoms % 3 != 0:
            raise ValueError("Total atoms not divisible by 3 for water molecules")

        good = True
        for i in range(n_atoms // 3):
            idxO, idxH1, idxH2 = 3 * i, 3 * i + 1, 3 * i + 2
            if (
                atoms.numbers[idxO] != 8
                or atoms.numbers[idxH1] != 1
                or atoms.numbers[idxH2] != 1
            ):
                logger.warn(f"Atom types not OHH at indices {idxO},{idxH1},{idxH2}")
                good = False
                break

            d1 = atoms.get_distance(idxO, idxH1, mic=True)
            d2 = atoms.get_distance(idxO, idxH2, mic=True)
            if d1 > OH_distance_tol or d2 > OH_distance_tol:
                logger.warn(
                    f"O-H distances {(d1, d2)} exceed tolerance {OH_distance_tol}"
                )
                good = False
                break

        return good

    def dump_test_configurations(self, path_to_folder: Path):
        """
        Write reference configurations and energies to disk for inspection.

        Parameters
        ----------
        path_to_folder : Path
            Directory where to save `atoms_{i}_{tag[i]}.xyz` and `energies.csv`.
        """
        path_to_folder = Path(path_to_folder)

        path_to_folder.mkdir(exist_ok=True, parents=True)

        filenames = []
        for i, atoms in enumerate(self.atoms_list):
            name = f"atoms_{i}_{self.tags[i]}.xyz"
            filenames.append(name)
            write(path_to_folder / name, atoms)

        df_data = {
            "tag": self.tags,
            "reference_energy": self.reference_energies,
            "file": filenames,
            "n_atoms": [len(a) for a in self.atoms_list],
        }

        df = pd.DataFrame(df_data)
        df.to_csv(path_to_folder / "energies.csv")

    def create_atoms_object_from_configuration(
        self, path_to_configuration: Path
    ) -> Atoms:
        """
        Load atoms from a configuration file, reorder them to conform to OHH order
        and attach the SCME calculator.

        Parameters
        ----------
        path_to_configuration : Path
            File path to an ASE-readable structure (e.g. .xyz).

        Returns
        -------
        Atoms
            Atoms object with SCME calculator attached and ready for energy eval.
        """
        logger.debug(f"Loading configuration from {path_to_configuration}")
        atoms = read(path_to_configuration)

        # If the first check does not pass, we will try to fix the order of atoms
        if not self.check_water_is_in_OHH_order(atoms):
            logger.warn("Will try to fix atoms object order")
            atoms = self.arange_water_in_OHH_order(atoms)

        # If we are not able to fix the order of atoms we raise an exception
        if not self.check_water_is_in_OHH_order(atoms):
            logger.critical("Could not fix atoms object order")
            raise ValueError("Atoms not in OHH order")

        scme_params = self.default_scme_params.copy()
        setup_calculator(
            atoms,
            scme_params=scme_params,
            parametrization_key=self.parametrization_key,
        )
        return atoms

    def _create_list_of_atom_objects(self) -> List[Atoms]:
        """
        Internal: Build Atoms objects for all reference configurations.

        Returns
        -------
        List[Atoms]
            Prepared Atoms objects for each reference path.
        """
        return [
            self.create_atoms_object_from_configuration(p)
            for p in self.paths_to_reference_configuration
        ]

    def get_energy(self, idx: int, parameters: Dict[str, float]) -> float:
        """
        Compute SCME energy for configuration `idx` with `parameters` applied.

        Parameters
        ----------
        idx : int
            Index of the reference configuration.
        parameters : Dict[str, float]
            SCME parameter values to set before evaluation.

        Returns
        -------
        float
            Potential energy from the ASE Atoms object.

        Raises
        ------
        KeyError
            If required parameters are missing.
        """

        atoms = self.atoms_list[idx]
        for key, value in parameters.items():
            if hasattr(atoms.calc.scme, key):
                setattr(atoms.calc.scme, key, value)
            else:
                raise KeyError(
                    f"There was a key in the parameters dict, which cannot be set on the scmecpp.Potential object. The offending key was {key}"
                )

        # We have to make sure to trigger the update of the energy manually,
        # because ase will think it can use the cached energy values,
        # since none of the coordinates has changed.
        # Therefore, we explicitly call the `calculate` function
        atoms.calc.calculate(atoms)
        energy = atoms.get_potential_energy()
        logger.debug(
            f"Calculated energy for idx {idx} (tag = {self.tags[idx]}): {energy}"
        )
        return energy

    def __call__(self, idx: int, parameters: dict):
        """
        Compute squared-error contribution for configuration `idx`.

        Parameters
        ----------
        idx : int
            Index of the reference configuration.
        parameters : Dict[str, float]
            SCME parameter values to apply.

        Returns
        -------
        float
            Squared difference between computed and reference energy.
        """

        energy = self.get_energy(idx, parameters)

        target_energy = self.reference_energies[idx]
        objective_function_contribution = (energy - target_energy) ** 2

        if self.divide_by_n_atoms:
            n_atoms = len(self.atoms_list[idx])
            objective_function_contribution /= n_atoms

        logger.debug(f"Current params = {parameters}")
        logger.debug(f"Objective function value = {objective_function_contribution}")

        return objective_function_contribution
