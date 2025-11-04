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
import math
from dataclasses import dataclass
from typing import Dict, Any
from collections.abc import Mapping
import numpy as np
from scipy.constants import R, k, h, c
from tabulate import tabulate
from pyscf.data import nist
from pyscf.hessian import thermo as th  # use original functions

LINEAR_INERTIA_TOL = 1e-4

def _print_list(title: str, freqs_cm1) -> None:
    """Print a frequency list (cm^-1) with simple Real/Imag labels (no colors)."""
    print()
    print("Vibration frequency calculation (cm^-1)")
    print(title)
    print("-" * 40)
    for i, f in enumerate(np.ravel(freqs_cm1)):
        if np.iscomplexobj(f) and getattr(f, "imag", 0) != 0:
            tag = "[Imag]"
            val = f"{abs(np.imag(f)):9.2f} i   "
        else:
            tag = "[Real]"
            val = f"{np.real(f):9.2f}     "
        print(f"Mode {i+1:2d}: {val} {tag}")
    print("-" * 40)
    print()


# TR quasi-frequencies

def _rotor_type_from_inertia(mass, coords, linear_tol=LINEAR_INERTIA_TOL):
    mass = np.asarray(mass, dtype=float)
    coords = np.asarray(coords, dtype=float)

    if mass.size <= 1:
        return "ATOM"

    com = np.average(coords, axis=0, weights=mass)
    shifted = coords - com

    inertia = np.zeros((3, 3), dtype=float)
    for m, (x, y, z) in zip(mass, shifted):
        inertia[0, 0] += m * (y * y + z * z)
        inertia[1, 1] += m * (x * x + z * z)
        inertia[2, 2] += m * (x * x + y * y)
        inertia[0, 1] -= m * (x * y)
        inertia[0, 2] -= m * (x * z)
        inertia[1, 2] -= m * (y * z)

    inertia[1, 0] = inertia[0, 1]
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]

    eigvals = np.sort(np.abs(np.linalg.eigvalsh(inertia)))
    largest = eigvals[-1]
    if largest < 1e-12:
        return "ATOM"
    if eigvals[0] / largest < linear_tol:
        return "LINEAR"
    return "REGULAR"


def _classify_rotor_type(mass, coords):
    rot_const = th.rotation_const(mass, coords)
    rotor_type = th._get_rotor_type(rot_const)
    if rotor_type == "REGULAR":
        inertia_type = _rotor_type_from_inertia(mass, coords)
        if inertia_type != "REGULAR":
            rotor_type = inertia_type
    return rotor_type


def compute_tr_frequencies(hess, mass, coords):
    natm = len(mass)

    # Center the coordinates (PySCF does this internally before building TR).
    coords = np.asarray(coords, dtype=float)
    mass = np.asarray(mass, dtype=float)
    mass_center = np.einsum("z,zx->x", mass, coords) / mass.sum()
    coords_centered = coords - mass_center

    # Reshape to 3N x 3N and symmetrize (numerical safety).
    H = hess.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
    Hs = 0.5 * (H + H.T)

    # Mass-weighted Hessian.
    mhalf = np.repeat(mass ** -0.5, 3)
    H_mass = Hs * mhalf[:, None] * mhalf[None, :]

    # Build TR basis from original helper.
    tr_raw = th._get_TR(mass, coords_centered)  # using original private helper

    # Determine how many rotation vectors we need.
    rotor_type = _classify_rotor_type(mass, coords_centered)
    rot_idx = []
    if rotor_type == "LINEAR":
        rot_idx = [3, 4]
    elif rotor_type not in ("ATOM", "LINEAR"):
        rot_idx = [3, 4, 5]

    # Selected raw vectors in the desired order (translations first).
    selected = [tr_raw[i].astype(float) for i in (0, 1, 2, *rot_idx)]
    categories = ["trans"] * 3 + ["rot"] * len(rot_idx)

    # Orthonormalize the selected vectors (Gram-Schmidt) so the projection does not
    # mix vibrational modes back into the TR subspace.
    ortho = []
    used_categories = []
    for vec, cat in zip(selected, categories):
        w = vec.copy()
        for b in ortho:
            w -= np.dot(b, w) * b
        norm = np.linalg.norm(w)
        if norm < 1e-10:
            continue
        ortho.append(w / norm)
        used_categories.append(cat)

    if len(ortho) < 3:
        raise ValueError("Failed to build three linearly independent translational vectors")

    B = np.vstack(ortho)

    # Force constants in the TR subspace (diagonalise the projected Hessian).
    H_sub = B @ H_mass @ B.T
    eigvals, eigvecs = np.linalg.eigh(0.5 * (H_sub + H_sub.T))

    # Convert eigenvalues to frequencies (cm^-1).
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2)) ** 0.5 / (2 * np.pi)
    freqs = np.sqrt(np.clip(eigvals, 0.0, None)) * au2hz / nist.LIGHT_SPEED_SI * 1e-2

    # Classify modes by their contribution to the translation/rotation subspaces.
    mode_info = []
    cats = np.array(used_categories)
    for freq, coeffs in zip(freqs, eigvecs.T):
        trans_weight = float(np.sum(coeffs[cats == "trans"] ** 2))
        rot_weight = float(np.sum(coeffs[cats == "rot"] ** 2))
        mode_info.append((freq, trans_weight, rot_weight))

    mode_info.sort(key=lambda x: x[0])
    trans_modes, rot_modes = [], []
    for freq, t_w, r_w in mode_info:
        prefer_trans = t_w >= r_w
        if prefer_trans and len(trans_modes) < 3:
            trans_modes.append(freq)
        elif len(rot_modes) < len(mode_info) - 3:
            rot_modes.append(freq)
        else:
            trans_modes.append(freq) if len(trans_modes) < 3 else rot_modes.append(freq)

    freq_cm1 = np.array([*trans_modes, *rot_modes])

    _print_list("with projection of translations & rotations", freq_cm1)
    return freq_cm1




def _n_tr(mass, coords) -> int:
    """Return the number of overall TR modes (3/5/6) for atom/linear/nonlinear."""
    rotor_type = _classify_rotor_type(mass, coords)
    return 3 if rotor_type == "ATOM" else 5 if rotor_type == "LINEAR" else 6

def collect_freq(mass, coords, results_from_harmonic, freqs_tr_cm1):
    natm = coords.shape[0]
    nTR = _n_tr(mass, coords)

    tr = np.asarray(freqs_tr_cm1).reshape(-1)[:nTR]

    # th.harmonic_analysis may return either (3N) or (3N-nTR) frequencies.
    arr = np.asarray(results_from_harmonic["freq_wavenumber"]).reshape(-1)
    expected_vib = 3 * natm - nTR
    if arr.size == expected_vib:
        vib = arr
    elif arr.size == 3 * natm:
        vib = arr[-expected_vib:]
    else:
        raise ValueError(
            f"size of freq_wavenumber = {arr.size} (expected {expected_vib} or {3*natm})"
        )

    full = np.concatenate([tr, vib])
    return tr, vib, full, nTR


def _mode_type(i: int, ntr: int) -> str:
    if i < 3:
        return "quasi-trans"
    if i < ntr:
        return "quasi-rot"
    return "vib"


def print_table_tabulate(freq_all_cm1, ntr: int) -> None:
    """Pretty-print a mode table using tabulate."""
    rows = [[i + 1, _mode_type(i, ntr), float(np.real(v))] for i, v in enumerate(freq_all_cm1)]
    print("Final Result")
    print(
        tabulate(
            rows,
            headers=["Mode", "type", "vibrational frequency (cm^-1)"],
            tablefmt="psql",
            colalign=("right", "left", "decimal"),
            floatfmt=(".0f", "", ".4f"),
        )
    )


def show_frequencies(mass, coords, hess, results_from_harmonic) -> Dict[str, Any]:
    """
    Convenience wrapper:
      1) compute TR quasi-frequencies
      2) collect TR + vib
      3) pretty-print a 3N table
    """
    freq_wo = results_from_harmonic["freq_wavenumber"]
    freqs_tr = compute_tr_frequencies(hess, mass, coords)
    tr, vib, full, nTR = collect_freq(mass, coords, results_from_harmonic, freqs_tr)
    print_table_tabulate(full, ntr=nTR)

    return {
        "freq_wavenumber_wo": freq_wo,  # 3N
        "freqs_tr": freqs_tr,           # 3/5/6
        "freq_vib": vib,                # 3N - nTR
        "freq_all": full,               # 3N
        "nTR": nTR,
    }


# HSM thermo
def _S_one(v_cm1: float, T: float) -> float:
    """Entropy contribution [J/mol/K] of one mode at wavenumber v (cm^-1)."""
    x = 100.0 * h * c / (k * T)
    t = x * v_cm1
    if t < 1e-12:
        return 0.0
    e = math.exp(-t)
    return R * (t * e / (1.0 - e) - math.log(1.0 - e))


def _E_one(v_cm1: float, T: float) -> float:
    """Energy contribution [J/mol] of one mode at wavenumber v (cm^-1)."""
    x = 100.0 * h * c / (k * T)
    t = x * v_cm1
    if t < 1e-12:
        return 0.0
    e = math.exp(-t)
    return R * T * (t * (0.5 + e / (1.0 - e)))

@dataclass
class HSMResult(Mapping):
    T: float
    freq: Dict[str, np.ndarray]
    S: Dict[str, float]
    E: Dict[str, float]

    def __getitem__(self, key: str):
        if key == "T":
            return self.T
        if key == "freq":
            return self.freq
        if key == "S":
            return self.S
        if key == "E":
            return self.E
        raise KeyError(key)

    def __iter__(self):
        return iter("T", "freq", "S", "E")

    def __len__(self) -> int:
        return 4

def calc_hsm(tr_cm1, vib_cm1, T: float = 298.15) -> HSMResult:
    tr = np.asarray(tr_cm1).reshape(-1)
    if tr.size < 3:
        raise ValueError("Need at least 3 TR frequencies (x, y, z translations).")
    freq_trans = tr[:3]
    freq_rot = tr[3:]
    freq_vib = np.real_if_close(np.asarray(vib_cm1), tol=1e8).astype(float, copy=False)

    # Entropy [J/mol/K]
    S_trans = float(sum(_S_one(v, T) for v in freq_trans))
    S_rot = float(sum(_S_one(v, T) for v in freq_rot))
    S_vib = float(sum(_S_one(v, T) for v in freq_vib))
    S_total = S_trans + S_rot + S_vib

    # Internal energy [J/mol]
    E_trans = 0.5 * float(sum(_E_one(v, T) for v in freq_trans))
    E_rot = 0.5 * float(sum(_E_one(v, T) for v in freq_rot))
    E_vib = float(sum(_E_one(v, T) for v in freq_vib))
    E_total = E_trans + E_rot + E_vib

    freq_all = np.concatenate([freq_trans, freq_rot, freq_vib])

    return HSMResult(
        T=T,
        freq={"trans": freq_trans, "rot": freq_rot, "vib": freq_vib, "all": freq_all},
        S={"trans": S_trans, "rot": S_rot, "vib": S_vib, "total": S_total},
        E={"trans": E_trans, "rot": E_rot, "vib": E_vib, "total": E_total},
    )


def print_hsm_tables(hsm: HSMResult) -> None:
    """Pretty-print HSM summary in kJ/mol and kcal/mol."""
    T = hsm.T
    S = hsm.S
    E = hsm.E
    kcal = 0.000239006  # kJ -> kcal

    print(f"\nThermochemistry  [T = {T:.2f} K]")
    rows_kj = [
        ["E (kJ/mol)",   E["trans"] / 1000, E["rot"] / 1000, E["vib"] / 1000, E["total"] / 1000],
        ["S (kJ/mol/K)", S["trans"] / 1000, S["rot"] / 1000, S["vib"] / 1000, S["total"] / 1000],
    ]
    print(
        tabulate(
            rows_kj,
            headers=["Quantity", "Trans", "Rot", "Vib", "Total"],
            tablefmt="psql",
            colalign=("left", "decimal", "decimal", "decimal", "decimal"),
            floatfmt=("", ".5f", ".5f", ".5f", ".5f"),
        )
    )

    print("\nkJ/mol -> kcal/mol")
    rows_kcal = [
        ["E (kcal/mol)",   E["trans"] * kcal, E["rot"] * kcal, E["vib"] * kcal, E["total"] * kcal],
        ["S (kcal/mol/K)", S["trans"] * kcal, S["rot"] * kcal, S["vib"] * kcal, S["total"] * kcal],
    ]
    print(
        tabulate(
            rows_kcal,
            headers=["Quantity", "Trans", "Rot", "Vib", "Total"],
            tablefmt="psql",
            colalign=("left", "decimal", "decimal", "decimal", "decimal"),
            floatfmt=("", ".5f", ".5f", ".5f", ".5f"),
        )
    )

if __name__ == "__main__":
    from pyscf import gto, hessian

    # Build a small test molecule.
    mol = gto.M(atom="O 0 0 0; H 0 .757 .587; H 0 -.757 .587")
    mf = mol.apply("HF").run()

    # Original calls from PySCF.
    H = hessian.RHF(mf).kernel()
    res = th.harmonic_analysis(mol, H)
    th.dump_normal_mode(mol, res)

    # Add-on display and HSM.
    mass = mol.atom_mass_list(isotope_avg=True)
    coords = mol.atom_coords()
    freqs = show_frequencies(mass, coords, H, res)

    hsm = calc_hsm(freqs["freqs_tr"], freqs["freq_vib"], T=298.15)
    print_hsm_tables(hsm)

    # Original thermo (for comparison).
    tres = th.thermo(mf, res["freq_au"], 298.15, 101325)
    th.dump_thermo(mol, tres)
