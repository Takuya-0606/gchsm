# Copyright 2025 The GCHSM Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Takuya HASHIMOTO
#

from __future__ import annotations
from typing import Iterable, List, Tuple
import numpy as np
from .nonelst_types import Atom, Site

# Claverie-style constants used by the DISREP model implemented in GAMESS.
A6_CONST = 0.214
CREP_CONST = 47000.0
GAMREP_NUM = 12.35

# Default solute element parameters (DKA, RWA) adopted from the original DISREP table.
_DKA_RWA_TABLE: dict[str, Tuple[float, float]] = {
    "H": (1.00, 1.20),
    "C": (1.00, 1.72),
    "N": (1.10, 1.60),
    "O": (1.36, 1.50),
    "P": (2.10, 1.85),
    "S": (1.40, 1.80),
    "BE": (1.00, 1.72),
    "B": (1.00, 1.72),
}


def _dka_rwa_for_element(elem: str) -> Tuple[float, float]:
    e = elem.upper()
    if e in _DKA_RWA_TABLE:
        return _DKA_RWA_TABLE[e]
    return 1.00, 1.72


def build_atom_list(mol) -> List[Atom]:
    coords_A = mol.atom_coords(unit="Angstrom")
    atoms: List[Atom] = []
    for ia in range(mol.natm):
        atoms.append(Atom(mol.atom_symbol(ia), np.array(coords_A[ia], float)))
    return atoms


def parse_site_tokens(tokens: Iterable[str]) -> List[Site]:
    sites: List[Site] = []
    for token in tokens:
        if not token:
            continue
        parts = token.split(":")
        if len(parts) < 4:
            raise ValueError(
                "Site specification must be LABEL:Nt:Rt:DKT[:RWT] (Å)."
            )
        label = parts[0].strip()
        Nt = int(parts[1])
        Rt = float(parts[2])
        DKT = float(parts[3])
        RWT = float(parts[4]) if len(parts) >= 5 else None
        sites.append(Site(label=label, Nt=Nt, Rt_A=Rt, DKT=DKT, RWT_A=RWT))
    if not sites:
        raise ValueError("At least one solvent site must be provided for GAMESS-89 model.")
    return sites


def compute_disp_rep_gamess_from_surface(
    centers: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    types: np.ndarray | None,
    atoms: List[Atom],
    sites: List[Site],
    rho_number_A3: float,
    normals_outward: bool,
    rcut: float = 1.0e-6,
) -> Tuple[float, float]:
    centers = np.asarray(centers, float)
    normals = np.asarray(normals, float)
    areas = np.asarray(areas, float).reshape(-1)
    sign = 1.0 if normals_outward else -1.0

    Edisp = 0.0
    Erep = 0.0

    for site in sites:
        Nt = int(site.Nt)
        DKT = float(site.DKT)
        _, RWT = site.radii_pair()

        for atom in atoms:
            DKA, RWA = _dka_rwa_for_element(atom.element)

            RWAT = 2.0 * np.sqrt(RWA * RWT)
            DAT = A6_CONST * DKA * DKT * (RWAT ** 6)
            BAT = CREP_CONST * DKA * DKT
            ALPH = GAMREP_NUM / RWAT

            rvec = centers - np.asarray(atom.position, float)
            r = np.linalg.norm(rvec, axis=1)
            r = np.maximum(r, rcut)

            rdotn = sign * np.einsum("ij,ij->i", rvec, normals)

            disp_kernel = -(DAT / 3.0) * (rdotn / (r ** 6))
            S1 = ALPH * r
            rep_kernel = (BAT * np.exp(-S1)) * (
                (1.0 / S1) + (2.0 / (S1 ** 2)) + (2.0 / (S1 ** 3))
            ) * rdotn

            contrib = Nt * rho_number_A3
            Edisp += contrib * np.sum(disp_kernel * areas)
            Erep += contrib * np.sum(rep_kernel * areas)

    return float(Edisp), float(Erep)
