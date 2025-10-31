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
import inspect
import numpy as np
from typing import Sequence, Optional, Union
from pyscf import gto
from pyscf.dft import gen_grid
from pyscf.solvent import pcm as _pcm
from pyscf.solvent.pcm import gen_surface, get_F_A_gcm, get_D_S_gcm, modified_Bondi
from scipy.special import erf

BOHR = 0.529177210903  # Angstrom per Bohr
PI = np.pi

def _filter_kwargs(fn, kwargs):
    """Return kwargs accepted by fn signature (robust across API versions)."""
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

class UserCavityPCM(_pcm.PCM):
    def __init__(
        self,
        mol: gto.Mole,
        centers: Sequence[Sequence[float]],
        radii: Optional[Union[Sequence[float], float]] = None,
        *,
        unit: str = "Angstrom",
        apply_scale_to_user: bool = False,
        assign_real_atoms: bool = True,
        **kw,
    ) -> None:
        # Store cavity centers internally in Bohr
        self._centers = np.asarray(centers, float)
        self._radii_user: Optional[np.ndarray] = None
        if unit.lower().startswith("a"):
            # Convert Angstrom -> Bohr
            self._centers /= BOHR
            if radii is not None:
                self._radii_user = np.asarray(radii, float) / BOHR
        else:
            if radii is not None:
                self._radii_user = np.asarray(radii, float)

        # Whether to apply (vdw_scale * r + r_probe) to user-specified radii
        self.apply_scale_to_user = bool(apply_scale_to_user)
        # Whether to assign real atom species to dummy molecule (True) or use 1..N pseudo charges (False)
        self.assign_real_atoms = bool(assign_real_atoms)

        _solver = kw.pop("solver", kw.pop("method", None))
        _eps    = kw.pop("eps", None)

        super().__init__(mol, **_filter_kwargs(_pcm.PCM.__init__, kw))

        if _solver is not None:
            # Some versions use .solver, others .method
            if hasattr(self, "solver"):
                self.solver = _solver
            else:
                self.method = _solver
        if _eps is not None:
            self.eps = _eps

    def _ensure_radii_table(self):
        if self.radii_table is None:
            # Default convention: scaled modified_Bondi table plus probe radius (Bohr)
            self.radii_table = self.vdw_scale * modified_Bondi + self.r_probe

    def _user_radii_vector(self, n: int) -> Optional[np.ndarray]:
        if self._radii_user is None:
            return None
        ru = np.asarray(self._radii_user, dtype=float)
        if ru.ndim == 0:
            ru = np.full(n, float(ru), dtype=float)
        elif ru.size != n:
            raise ValueError(f"Length mismatch: got {ru.size} radii for {n} centers.")
        if self.apply_scale_to_user:
            ru = self.vdw_scale * ru + self.r_probe
        return ru

    def _build_dummy_and_rad_table_real(self, ng: int):
        ncenters = len(self._centers)
        if ncenters != self.mol.natm:
            raise ValueError(
                f"Number of centers ({ncenters}) must equal number of atoms ({self.mol.natm}) "
                f"when assign_real_atoms=True."
            )

        self._ensure_radii_table()
        # Real atomic numbers (int) per atom
        Zs = np.asarray(self.mol.atom_charges(), dtype=int)
        unique_Z = np.unique(Zs)

        # Start from default table and override only the elements present
        rad_table = np.array(self.radii_table, dtype=float, copy=True)

        # User radii handling
        ru_vec = self._user_radii_vector(ncenters)
        if ru_vec is None:
            # No user radii: keep default per-element values already in rad_table
            pass
        else:
            # For each element, make sure all atoms share the same requested radius
            for z in unique_Z:
                idx = np.where(Zs == z)[0]
                vals = ru_vec[idx]
                if not np.allclose(vals, vals[0], rtol=0.0, atol=1e-14):
                    raise ValueError(
                        "assign_real_atoms=True requires the same radius for all atoms of the same element. "
                        f"Found element Z={z} with non-uniform radii: {vals}."
                    )
                rad_table[z] = float(vals[0])

        # Construct dummy molecule with real symbols; coordinates = centers (Bohr)
        dummy = gto.Mole()
        dummy.unit = "Bohr"
        dummy.atom = [(self.mol.atom_symbol(i), coord.tolist())
                      for i, coord in enumerate(self._centers)]
        dummy.basis = {}
        dummy.charge = 0
        dummy.spin = 0
        dummy.build(unit="Bohr", verbose=0)

        return dummy, rad_table

    def _build_dummy_and_rad_table_unique(self, ng: int):
        ncenters = len(self._centers)
        self._ensure_radii_table()

        # Real atom Zs to pick element-default radii if user radii not given
        Zs = np.asarray(self.mol.atom_charges(), dtype=int)

        # Per-center radii vector
        ru_vec = self._user_radii_vector(ncenters)
        if ru_vec is None:
            # Use element default per atom
            ru_vec = np.array([self.radii_table[z] for z in Zs], dtype=float)

        # Build a "table" of length ncenters+1 with entries 1..N = per-center radii
        rad_table = np.zeros(ncenters + 1, dtype=float)
        rad_table[1:] = ru_vec

        # Construct dummy molecule with integer charges 1..N (unique labels)
        dummy = gto.Mole()
        dummy.unit = "Bohr"
        dummy.atom = [((i + 1), coord.tolist()) for i, coord in enumerate(self._centers)]
        dummy.basis = {}
        dummy.charge = 0
        dummy.spin = 0
        dummy.build(unit="Bohr", verbose=0)

        return dummy, rad_table

    def build(self, ng: Optional[int] = None):
        ncenters = len(self._centers)
        if ncenters == 0:
            raise ValueError("No cavity centers provided.")

        # Angular grid size (Lebedev points per center)
        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        # Build dummy molecule and radius table according to the chosen strategy
        if self.assign_real_atoms:
            dummy, rad_table = self._build_dummy_and_rad_table_real(ng)
        else:
            dummy, rad_table = self._build_dummy_and_rad_table_unique(ng)

        # Generate cavity surface
        self.surface = gen_surface(dummy, rad=rad_table, ng=ng)

        # Intermediates (S, D, A, K, R)
        self._intermediates = {}
        F, A = get_F_A_gcm(self.surface)
        D, S, O = get_D_S_gcm(self.surface, with_S=True, with_D=True)

        eps = self.eps
        mth = self.method.upper()
        if mth in ["C-PCM", "CPCM"]:
            f_eps = (eps - 1.0) / eps if eps != float("inf") else 1.0
            K = S
            R = -f_eps * np.eye(K.shape[0])
        elif mth == "COSMO":
            f_eps = (eps - 1.0) / (eps + 0.5) if eps != float("inf") else 1.0
            K = S
            R = -f_eps * np.eye(K.shape[0])
        else:
            raise RuntimeError(f"Unknown PCM method: {self.method}")

        self._intermediates.update({"S": S, "D": D, "A": A, "K": K, "R": R, "f_epsilon": f_eps})

        # Nuclear potential on surface
        zeta = self.surface["charge_exp"]
        grid_coords = self.surface["grid_coords"]
        nuc_coords = self.mol.atom_coords(unit="B")  # Bohr
        nuc_chg = self.mol.atom_charges()

        v_ng = np.zeros(len(grid_coords), dtype=float)
        for i, ri in enumerate(grid_coords):
            # Smooth nuclear potential via erf(zeta * r) / r
            zi = zeta[i]
            for RA, ZA in zip(nuc_coords, nuc_chg):
                r = np.linalg.norm(ri - RA)
                if r < 1e-12:
                    continue
                v_ng[i] += ZA * erf(zi * r) / r
        self.v_grids_n = v_ng
        return self

    def reset(self, mol: Optional[gto.Mole] = None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        self.intopt = None
        return self
