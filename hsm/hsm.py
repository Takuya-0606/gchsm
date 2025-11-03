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
import inspect
import numpy as np
from typing import Sequence, Optional, Union, Tuple
from pyscf import gto, scf, solvent
from pyscf.dft import gen_grid
from pyscf.solvent import pcm as _pcm
from pyscf.solvent.pcm import gen_surface, get_F_A_gcm, get_D_S_gcm, modified_Bondi
from pyscf.data import radii as _radii
BOHR = _radii.BOHR

def _filter_kwargs(fn, kwargs):
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

class fixcav(_pcm.PCM):
    def __init__(
        self,
        mol: gto.Mole,
        centers: Sequence[Sequence[float]],
        radii: Optional[Union[float, int]] = None,
        *,
        unit: str = "Angstrom",
        **kw,
    ) -> None:
        self._centers = np.asarray(centers, float)
        if unit.lower().startswith("a"):
            self._centers /= BOHR

        _solver = kw.pop("solver", kw.pop("method", None))
        _eps = kw.pop("eps", None)

        super().__init__(mol, **_filter_kwargs(_pcm.PCM.__init__, kw))

        if _solver is not None:
            if hasattr(self, "solver"):
                self.solver = _solver
            else:
                self.method = _solver
        if _eps is not None:
            self.eps = _eps

        if radii is not None:
            self.vdw_scale = float(radii)

    def build(self, ng: Optional[int] = None):
        """Generate PCM surface using user centers (kept across resets)."""
        if self.radii_table is None:
            radii_table = self.vdw_scale * modified_Bondi + self.r_probe / BOHR
        else:
            radii_table = np.asarray(self.radii_table, float)

        if len(self._centers) != self.mol.natm:
            raise ValueError("Number of centers must equal number of atoms.")

        dummy = gto.Mole()
        dummy.unit = "Bohr"
        dummy.atom = [(self.mol.atom_symbol(i), coord.tolist())
                      for i, coord in enumerate(self._centers)]
        dummy.build()

        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        self.surface = gen_surface(dummy, rad=radii_table, ng=ng)

        self._intermediates = {}
        _, A = get_F_A_gcm(self.surface)
        D, S, _ = get_D_S_gcm(self.surface, with_S=True, with_D=True)

        eps = self.eps
        mu = self.method.upper()
        if mu in ["C-PCM", "CPCM"]:
            f_eps = (eps - 1.0) / eps if eps != float("inf") else 1.0
            K = S
            R = -f_eps * np.eye(K.shape[0])
        elif mu == "COSMO":
            f_eps = (eps - 1.0) / (eps + 0.5) if eps != float("inf") else 1.0
            K = S
            R = -f_eps * np.eye(K.shape[0])
        else:
            raise RuntimeError(f"Unknown PCM method: {self.method}")

        self._intermediates.update({"S": S, "D": D, "A": A, "K": K, "R": R, "f_epsilon": f_eps})

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

    def attach_to(self, mf, *, build_surface: bool = True, lebedev_order: Optional[int] = None,
                  conv_tol: float = 1e-10, conv_tol_grad: float = 1e-6):
        if lebedev_order is not None:
            self.lebedev_order = lebedev_order
        if build_surface:
            self.build()
        mf_pcm = solvent.PCM(mf)
        mf_pcm.with_solvent = self
        mf_pcm.conv_tol = conv_tol
        mf_pcm.conv_tol_grad = conv_tol_grad
        return mf_pcm

    @classmethod
    def build_scf(cls,
                  mol: gto.Mole,
                  *,
                  centers: Sequence[Sequence[float]],
                  radii: Optional[Union[float, int]] = None,  # vdw_scale
                  unit: str = "Angstrom",
                  method: str = "C-PCM",
                  eps: float = 78.3553,
                  lebedev_order: int = 17,
                  scf_method: str = "RHF",
                  **pcm_kw) -> Tuple[scf.hf.SCF, "fixcav"]:
        scf_method = scf_method.upper()
        if scf_method == "RHF":
            mf = scf.RHF(mol)
        elif scf_method == "UHF":
            mf = scf.UHF(mol)
        else:
            raise ValueError("scf_method must be RHF or UHF for MP2 base.")
        pcm = cls(mol, centers=centers, radii=radii, unit=unit,
                  method=method, eps=eps, **pcm_kw)
        pcm.lebedev_order = lebedev_order
        mf_pcm = pcm.attach_to(mf, build_surface=True)
        return mf_pcm, pcm
