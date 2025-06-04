from scme_fitting.scme_setup import setup_calculator, SCMEParams, setup_monomer
from ase.optimize import FIRE2
from pathlib import Path
import numpy as np

import pyscme

atoms = setup_monomer()

params = {
    "te": 2.267671351004314,
    "td": 3.9978279903677074,
    "Ar": 299.49338092158087,
    "Br": -0.5515,
    "Cr": -1.8359803320512469,
    "r_Br": 1.8897261258369282,
    "rc_Disp": 15.117809006695426,
    "rc_Core": 14.172945943776963,
    "rc_Elec": 17.007535132532354,
    "C6": 46.443,
    "C8": 1141.7,
    "C10": 33441.0,
    "w_rc_Elec": 3.7794522516738565,
    "w_rc_Core": 3.7794522516738565,
    "w_rc_Disp": 3.7794522516738565,
    "max_iter_scf": 100,
    "scf_convcrit": 1e-08,
    "dms": False,
    "qms": False,
    "NC": [0, 0, 0],
}

optimal_params = {
    "te": 2.1270441941898586,
    "td": 2.134469356359326,
    "Ar": 204.7933329977729,
    "Br": -0.4062281538691688,
    "Cr": -1.9007824976059151,
    "C6": 32.00784215100572,
    "C8": 511.35232980087875,
    "C10": 2655.3308723266855,
}

params.update(optimal_params)

scme_params = SCMEParams(**params)

path_to_scme_expansions = Path(
    "/home/moritz/SCME/generalized_SCME_interatomic_fit/input/scme_expansions_BLYP.hdf5"
)
parametrization_key = "component_BLYP_fullrange_reflect_8_12"

setup_calculator(
    atoms=atoms,
    scme_params=scme_params,
    path_to_scme_expansions=path_to_scme_expansions,
    parametrization_key=parametrization_key,
)

dt = 1e-2
fmax = 1e-14

opt = FIRE2(atoms, dt=dt)
opt.run(fmax=fmax)

rO, rH1, rH2 = atoms.get_positions()
mO = atoms.get_masses()[0]
mH = atoms.get_masses()[1]
local_frame_info = pyscme.get_local_frame(rO, rH1, rH2, mO, mH, atoms.calc.scme.box)
RT = np.ascontiguousarray(np.array(local_frame_info.rotation_matrix).T)

d = pyscme.rotate_tensor(RT, atoms.calc.scme.dipole_moments[0])
q = pyscme.rotate_tensor(RT, atoms.calc.scme.quadrupole_moments[0])
o = pyscme.rotate_tensor(RT, atoms.calc.scme.octupole_moments[0])
h = pyscme.rotate_tensor(RT, atoms.calc.scme.hexadecapole_moments[0])
dd = pyscme.rotate_tensor(RT, atoms.calc.scme.dip_dip_polarizability[0])
dq = pyscme.rotate_tensor(RT, atoms.calc.scme.dip_quad_polarizability[0])
qq = pyscme.rotate_tensor(RT, atoms.calc.scme.quad_quad_polarizability[0])

np.savez(
    "blyp_monomer_moments.npz",
    dipole=d,
    quadrupole=q,
    octupole=o,
    hexadecapole=h,
    dip_dip=dd,
    dip_quad=dq,
    quad_quad=qq,
)
