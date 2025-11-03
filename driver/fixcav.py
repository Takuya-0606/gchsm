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

import copy
import inspect
from typing import Optional, Sequence, Union

import numpy as np
from pyscf import gto, solvent
from pyscf.dft import gen_grid
from pyscf.solvent import pcm as _pcm
from pyscf.solvent.pcm import (
    gen_surface,
    get_D_S_gcm,
    get_F_A_gcm,
    modified_Bondi,
)
from pyscf.data import radii as _radii

BOHR = _radii.BOHR  # Angstrom per Bohr (for conversions)


def _filter_kwargs(fn, kwargs):
    """Return kwargs accepted by fn signature (robust across API versions)."""

    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class UserCavityPCM(_pcm.PCM):
    """PCM solver with a frozen user-specified cavity (centres and radii)."""

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
        # Cache cavity centres (detached copy so geometry updates do not move them)
        self._centers = np.array(centers, dtype=float, copy=True)

        radii_array: Optional[np.ndarray]
        if radii is None or np.isscalar(radii):
            radii_array = None
        else:
            radii_array = np.array(radii, dtype=float, copy=True)

        if unit.lower().startswith("a"):
            # Convert Angstrom -> Bohr
            self._centers /= BOHR
            if radii_array is not None:
                radii_array /= BOHR

        self._radii_user = radii_array
        self.apply_scale_to_user = bool(apply_scale_to_user)
        self.assign_real_atoms = bool(assign_real_atoms)

        solver = kw.pop("solver", kw.pop("method", None))
        eps = kw.pop("eps", None)

        super().__init__(mol, **_filter_kwargs(_pcm.PCM.__init__, kw))

        if solver is not None:
            if hasattr(self, "solver"):
                self.solver = solver
            else:
                self.method = solver
        if eps is not None:
            self.eps = eps

        if radii is not None and np.isscalar(radii):
            # Mirror behaviour of reference implementation: scalar -> vdw_scale
            self.vdw_scale = float(radii)

    def _ensure_radii_table(self) -> None:
        if self.radii_table is None:
            # Reference implementation uses scaled modified_Bondi + probe radius (Angstrom)
            self.radii_table = self.vdw_scale * modified_Bondi + self.r_probe / BOHR

    def _user_radii_vector(self, n: int) -> Optional[np.ndarray]:
        if self._radii_user is None:
            return None
        vec = np.asarray(self._radii_user, dtype=float)
        if vec.ndim == 0:
            vec = np.full(n, float(vec), dtype=float)
        elif vec.size != n:
            raise ValueError(
                f"Length mismatch: got {vec.size} radii for {n} centres."
            )
        if self.apply_scale_to_user:
            vec = self.vdw_scale * vec + self.r_probe / BOHR
        return vec

    def _build_dummy_and_rad_table_real(self, ng: int):
        ncentres = len(self._centers)
        if ncentres != self.mol.natm:
            raise ValueError(
                "Number of centres must equal number of atoms when assign_real_atoms=True."
            )

        self._ensure_radii_table()
        rad_table = np.array(self.radii_table, dtype=float, copy=True)

        radii_vec = self._user_radii_vector(ncentres)
        if radii_vec is not None:
            zs = np.asarray(self.mol.atom_charges(), dtype=int)
            for z in np.unique(zs):
                idx = np.where(zs == z)[0]
                vals = radii_vec[idx]
                if not np.allclose(vals, vals[0], rtol=0.0, atol=1e-14):
                    raise ValueError(
                        "assign_real_atoms=True requires identical radii for atoms of the same element."
                    )
                rad_table[z] = float(vals[0])

        dummy = gto.Mole()
        dummy.unit = "Bohr"
        dummy.atom = [
            (self.mol.atom_symbol(i), coord.tolist())
            for i, coord in enumerate(self._centers)
        ]
        if self.mol.basis is not None:
            dummy.basis = copy.deepcopy(self.mol.basis)
        else:
            dummy.basis = {}
        dummy.charge = 0
        dummy.spin = 0
        dummy.build(unit="Bohr", verbose=0)

        return dummy, rad_table

    def _build_dummy_and_rad_table_unique(self, ng: int):
        ncentres = len(self._centers)
        self._ensure_radii_table()

        zs = np.asarray(self.mol.atom_charges(), dtype=int)
        radii_vec = self._user_radii_vector(ncentres)
        if radii_vec is None:
            radii_vec = np.array([self.radii_table[z] for z in zs], dtype=float)

        rad_table = np.zeros(ncentres + 1, dtype=float)
        rad_table[1:] = radii_vec

        dummy = gto.Mole()
        dummy.unit = "Bohr"
        dummy.atom = [((i + 1), coord.tolist()) for i, coord in enumerate(self._centers)]
        if self.mol.basis is not None:
            dummy.basis = copy.deepcopy(self.mol.basis)
        else:
            dummy.basis = {}
        dummy.charge = 0
        dummy.spin = 0
        dummy.build(unit="Bohr", verbose=0)

        return dummy, rad_table

    # Public API
    def build(self, ng: Optional[int] = None):
        if len(self._centers) == 0:
            raise ValueError("No cavity centres provided.")

        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        if self.assign_real_atoms:
            dummy, rad_table = self._build_dummy_and_rad_table_real(ng)
        else:
            dummy, rad_table = self._build_dummy_and_rad_table_unique(ng)

        self.surface = gen_surface(dummy, rad=rad_table, ng=ng)

        self._intermediates = {}
        _, A = get_F_A_gcm(self.surface)
        D, S, _ = get_D_S_gcm(self.surface, with_S=True, with_D=True)

        eps = self.eps
        method = self.method.upper()
        if method in ("C-PCM", "CPCM"):
            f_eps = (eps - 1.0) / eps if eps != float("inf") else 1.0
            K = S
            R = -f_eps * np.eye(K.shape[0])
        elif method == "COSMO":
            f_eps = (eps - 1.0) / (eps + 0.5) if eps != float("inf") else 1.0
            K = S
            R = -f_eps * np.eye(K.shape[0])
        else:
            raise RuntimeError(f"Unknown PCM method: {self.method}")

        self._intermediates.update(
            {"S": S, "D": D, "A": A, "K": K, "R": R, "f_epsilon": f_eps}
        )

        charge_exp = self.surface["charge_exp"]
        grid_coords = self.surface["grid_coords"]
        atom_coords = self.mol.atom_coords(unit="B")
        atom_charges = self.mol.atom_charges()

        int2c2e = self.mol._add_suffix("int2c2e")
        fakemol = gto.fakemol_for_charges(grid_coords, expnt=charge_exp**2)
        fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol)
        self.v_grids_n = np.dot(atom_charges, v_ng)

        return self

    def reset(self, mol: Optional[gto.Mole] = None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        if hasattr(self, "intopt"):
            self.intopt = None
        return self

    def attach_to(
        self,
        mf,
        *,
        build_surface: bool = True,
        lebedev_order: Optional[int] = None,
        conv_tol: float = 1e-10,
        conv_tol_grad: float = 1e-6,
    ):
        if lebedev_order is not None:
            self.lebedev_order = lebedev_order
        if build_surface:
            self.build()
        mf_pcm = solvent.PCM(mf)
        mf_pcm.with_solvent = self
        mf_pcm.conv_tol = conv_tol
        mf_pcm.conv_tol_grad = conv_tol_grad
        return mf_pcm
~
