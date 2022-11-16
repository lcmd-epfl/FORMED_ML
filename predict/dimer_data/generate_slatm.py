#!/usr/bin/env python3
import os

import numpy as np
import psutil
import qml
from qml.representations import get_slatm_mbtypes

namelist = []
for name in os.listdir(
    "."
):  # names contains the path to the xyz files to be loaded as compounds which will then be slatmized
    namelist.append(qml.Compound(xyz=str(here * "/" + name)))

compounds = np.asarray(namelist, dtype=object)  # WARNING: REMOVE SLICING
print("Generated compounds; RAM memory % used:", psutil.virtual_memory()[2], flush=True)
print("Total RAM:", psutil.virtual_memory()[0], flush=True)
print("Available RAM:", psutil.virtual_memory()[1], flush=True)
if os.path.exists("mbtypes.npy"):
    mbtypes = np.load("mbtypes.npy", allow_pickle=True)
else:
    mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
    mbtypes = np.array(mbtypes)
    np.save("mbtypes.npy", mbtypes)
# replace this number with the size of the mbtypes
SIZEOFSLATM = 50518
X = np.zeros((len(compounds), SIZEOFSLATM), dtype=np.float16)
N = []
print(
    "Generated empty representation matrix; RAM memory % used:",
    psutil.virtual_memory()[2],
    flush=True,
)
for i, mol in enumerate(compounds):
    print(f"Tackling representation of {namelist[i]}", flush=True)
    mol.generate_slatm(mbtypes, local=False, dgrids=[0.1, 0.1])
    print(mol.representation.shape)
    X[i, :] = np.float16(mol.representation)
    N.append(mol.name)
    print(
        "Filled in one representation vector; RAM memory % used:",
        psutil.virtual_memory()[2],
        flush=True,
    )
    del mol

N = np.array(N)
np.save("repr.npy", X)
np.save("names.npy", N)
