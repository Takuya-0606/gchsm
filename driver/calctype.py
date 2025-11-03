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
from typing import Dict, Any, Iterable, List
import numpy as np
from pyscf import mp
from pyscf.geomopt import geometric_solver
from pyscf.hessian import thermo
from . import hsm_thermo
from pyscf.tools import finite_diff as fdtools
from pyscf.solvent.hessian import pcm as pcm_hess

KJ_PER_J   = 1e-3
KCAL_PER_J = 0.000239006

def run_sp(mf) -> Dict[str, Any]:
    e = mf.kernel()
    print(f"\n=== SCF energy (Eh) = {e:.12f}")
    return {"scf_energy": float(e)}

def run_opt(mf) -> Dict[str, Any]:
    print("\n=== Geometry Optimization (geomeTRIC) ===")
    mol_opt = geometric_solver.optimize(mf, maxsteps=200)
    return {"opt_coords_ang": mol_opt.atom_coords(unit="Angstrom").tolist()}

def run_sp_mp2(mf) -> Dict[str, Any]:
    """Single-point MP2 energy on a given SCF reference."""
    # Ensure SCF reference is available
    if not getattr(mf, "converged", False):
        escf = mf.kernel()
    else:
        escf = mf.e_tot
    from pyscf import mp as _mp
    mymp = _mp.MP2(mf)
    pcm = getattr(mf, "with_solvent", None)
    if pcm is not None:
        mymp = mymp.PCM(pcm)
    emp2, _ = mymp.kernel()  # returns (e_corr, t2) ; mymp.e_tot = escf + e_corr
    out = {
        "scf_energy": float(escf),
        "mp2_correlation_energy": float(emp2),
        "mp2_total_energy": float(mymp.e_tot),
    }
    print(f"\n=== MP2 single-point ===\nSCF (Eh) = {escf:.12f}\nMP2 corr (Eh) = {emp2:.12f}\nMP2 total (Eh) = {mymp.e_tot:.12f}")
    return out


def run_opt_mp2(mf) -> Dict[str, Any]:
    """Geometry optimization at the MP2 level using analytic MP2 gradients.
    Implemented via geomeTRIC + Gradients.as_scanner() + addons.as_pyscf_method.
    """
    print("\n=== Geometry Optimization at MP2 (geomeTRIC) ===")
    from pyscf.geomopt import geometric_solver, addons
    from pyscf import mp as _mp
    mymp = _mp.MP2(mf)
    pcm = getattr(mf, "with_solvent", None)
    if pcm is not None:
        mymp = mymp.PCM(pcm)
    scan = mymp.nuc_grad_method().as_scanner()  # [mol] -> (E, grad)
    mwrap = addons.as_pyscf_method(mf.mol, scan)
    mol_opt = geometric_solver.optimize(mwrap, maxsteps=200)
    return {"opt_coords_ang": mol_opt.atom_coords(unit="Angstrom").tolist(), "mp2_optimized": True}

def _is_pcm_hessian(obj) -> bool:
    """Heuristics: return True if Hessian object is already PCM-aware."""
    try:
        # Class name check (e.g., 'PCMHessian', 'WithSolventHess', etc.)
        name = obj.__class__.__name__.lower()
        if "pcm" in name or "with_solvent" in name:
            return True
    except Exception:
        pass
    # Some branches attach .base.with_solvent on the driver
    try:
        base = getattr(obj, "base", None)
        if base is not None and getattr(base, "with_solvent", None) is not None:
            return True
    except Exception:
        pass
    return False

def _make_pcm_hessian_object_robust(mf):
    """
    Return a Hessian driver that is PCM-aware, avoiding MRO and base-attribute issues
    across PySCF branches.
    Strategy:
      1) Use mf.Hessian() directly; if already PCM-aware -> return as-is
      2) If NOT PCM-aware, try pcm_hess.make_hess_object(raw_hess)
      3) Fallback: try pcm_hess.make_hess_object(mf)
    """
    last_exc = None

    # 1) Prefer the SCF-provided Hessian object
    try:
        raw = mf.Hessian()
        if _is_pcm_hessian(raw):
            return raw  # already PCM-aware; do NOT wrap again
        # 2) Not PCM-aware: try to wrap the Hessian object
        try:
            wrapped = pcm_hess.make_hess_object(raw)
            return wrapped
        except Exception as e2:
            last_exc = e2
    except Exception as e1:
        last_exc = e1

    # 3) Last resort: attempt wrapping the SCF object itself
    try:
        return pcm_hess.make_hess_object(mf)
    except Exception as e3:
        # Raise a clear error with breadcrumbs
        raise RuntimeError(
            "Failed to build PCM Hessian driver via all strategies:\n"
            f" - mf.Hessian(): {type(last_exc).__name__ if last_exc else 'n/a'}\n"
            f" - pcm_hess.make_hess_object(mf): {type(e3).__name__}: {e3}"
        )

def thermo_tables_from_freqs(tr_cm1: np.ndarray, vib_cm1: np.ndarray, temps: Iterable[float]) -> List[dict]:
    """Build thermochemistry tables (E,S) for each T using your thermo helpers."""
    out = []
    for T in temps:
        t = hsm_thermo.calc_hsm(tr_cm1, vib_cm1, T=T)  # returns J/mol and J/mol/K
        E = t["E"]; S = t["S"]
        row = {
            "T": float(T),
            "E_kJmol": {
                "vib": E["vib"]*KJ_PER_J,
                "rot": E["rot"]*KJ_PER_J,
                "trans": E["trans"]*KJ_PER_J,
                "total": E["total"]*KJ_PER_J,
            },
            "E_kcalmol": {
                "vib": E["vib"]*KCAL_PER_J,
                "rot": E["rot"]*KCAL_PER_J,
                "trans": E["trans"]*KCAL_PER_J,
                "total": E["total"]*KCAL_PER_J,
            },
            "S_kJmolK": {
                "vib": S["vib"]*KJ_PER_J,
                "rot": S["rot"]*KJ_PER_J,
                "trans": S["trans"]*KJ_PER_J,
                "total": S["total"]*KJ_PER_J,
            },
        }
        out.append(row)
    return out

def _is_linear_by_inertia(masses_amu: np.ndarray, coords_bohr: np.ndarray, thresh: float = 1e-4) -> bool:
    masses = np.asarray(masses_amu, dtype=float)
    R = np.asarray(coords_bohr, dtype=float)
    com = np.average(R, axis=0, weights=masses)
    X = R - com
    I = np.zeros((3,3), dtype=float)
    for m, r in zip(masses, X):
        x, y, z = r
        I[0,0] += m*(y*y + z*z); I[1,1] += m*(x*x + z*z); I[2,2] += m*(x*x + y*y)
        I[0,1] -= m*(x*y);       I[0,2] -= m*(x*z);       I[1,2] -= m*(y*z)
    I[1,0] = I[0,1]; I[2,0] = I[0,2]; I[2,1] = I[1,2]
    w = np.sort(np.abs(np.linalg.eigvalsh(I)))
    return (w[-1] > 0.0) and (w[0]/w[-1] < thresh)

def thermo_from_loaded_freqs(mol, freqs_cm1: Iterable[complex], temps: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray([complex(x) for x in freqs_cm1]).ravel()
    vib_cm1 = np.array([abs(a.imag) if (a.imag != 0.0 and a.real == 0.0) else abs(a.real)
                        for a in arr], dtype=float)

    natm   = mol.natm
    mass   = mol.atom_mass_list(isotope_avg=True)
    coords = mol.atom_coords(unit="Bohr")
    hess0  = np.zeros((natm, natm, 3, 3), dtype=float)
    tr_cm1 = thermo.compute_tr_frequencies(hess0, mass, coords)  # cm^-1

    tables = thermo_tables_from_freqs(np.asarray(tr_cm1), np.asarray(vib_cm1), temps)

    return {
        "total_mass_amu": float(np.sum(mass)),
        "tr_frequencies_cm-1":  np.asarray(tr_cm1).ravel().tolist(),
        "vib_frequencies_cm-1": np.asarray(vib_cm1).ravel().tolist(),
        "thermo_tables": tables,
    }

def _harmonic_analysis_from_hessian(mf, hess: np.ndarray, *, project: bool, freqtemps: Iterable[float]) -> Dict[str, Any]:
    mass   = mf.mol.atom_mass_list(isotope_avg=True)
    coords = mf.mol.atom_coords(unit="Bohr")

    info = thermo.harmonic_analysis(
        mf.mol, hess, imaginary_freq=True, exclude_trans=False, exclude_rot=False
    )

    freqs_tr = thermo.compute_tr_frequencies(hess, mass, coords)
    tr, vib, full, nTR = thermo.collect_freq(mass, coords, info, freqs_tr)

    _ = thermo.show_frequencies(mass, coords, hess, info)

    raw_freqs = np.asarray(info["freq_wavenumber"]).ravel()
    projected_full = np.asarray(full).ravel()

    if project:
        freqs_to_report = projected_full
    else:
        freqs_to_report = raw_freqs

    analysis: Dict[str, Any] = {
        "project": bool(project),
        "frequencies_cm-1": freqs_to_report,
        "tr_frequencies_cm-1": np.asarray(freqs_tr).ravel(),
        "nTR": nTR,
        "total_mass_amu": float(np.sum(mass)),
        "tr_frequencies_cm-1": np.asarray(tr).ravel(),
        "vib_frequencies_cm-1": np.asarray(vib).ravel(),
        "raw_frequencies_cm-1": raw_freqs,
        "projected_frequencies_cm-1": projected_full,
    }

    analysis["thermo_tables"] = thermo_tables_from_freqs(
        np.asarray(tr).ravel(),
        np.asarray(vib).ravel(),
        temps=list(freqtemps),
    )

    return analysis


def run_hess(mf, *, project: bool = False, freqtemps: Iterable[float] = (298.15,)) -> Dict[str, Any]:
    """
    Solvent Hessian + vibrational analysis with optional TR/rot projection.

    project = False : report raw 3N from harmonic_analysis (unprojected)
    project = True  : report 3N after replacing the lowest nTR with TR-projected values
    """
    out: Dict[str, Any] = {"project": bool(project)}

    # SCF
    e = mf.kernel()
    print(f"\n=== SCF energy (Eh) = {e:.12f}")

    # Analytic solvent Hessian (robust factory expected to handle .base MRO issues)
    print("\n=== Numerical Hessian (solvent) ===")
    if getattr(mf, "with_solvent", None) is None:
        print("[warn] SCF Hessian is vacuum. (mf.with_solvent is not found.）")
    hobj = _make_pcm_hessian_object_robust(mf)
    hess = hobj.kernel()  # shape: (natm, natm, 3, 3)
    out["hessian"] = hess

    # Frequencies
    print("\n=== Harmonic analysis / frequencies ===")
    out.update(
        _harmonic_analysis_from_hessian(
            mf,
            np.asarray(hess),
            project=project,
            freqtemps=freqtemps,
        )
    )
    return out

def run_mp2(
    mf,
    *,
    project: bool = False,
    freqtemps: Iterable[float] = (298.15,),
    fd_step: float = 5.0e-3,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"project": bool(project)}

    # Reference SCF
    escf = mf.kernel()
    print(f"\n=== SCF reference energy (Eh) = {escf:.12f}")
    out["scf_energy"] = float(escf)

    # MP2 energy
    print("\n=== MP2 correlation ===")
    mp2_solver = mp.MP2(mf)
    emp2, _ = mp2_solver.kernel()
    out["mp2_correlation_energy"] = float(emp2)
    out["mp2_total_energy"] = float(mp2_solver.e_tot)
    print(f"MP2 correlation energy (Eh) = {emp2:.12f}")
    print(f"MP2 total energy (Eh)       = {mp2_solver.e_tot:.12f}")

    # Numerical Hessian via finite differences of MP2 gradients
    print("\n=== Numerical Hessian (MP2) ===")
    grad_method = mp2_solver.nuc_grad_method()
    grad_method.verbose = 4
    if hasattr(grad_method, "conv_tol"):
        grad_method.conv_tol = getattr(grad_method, "conv_tol") or 1e-6
    if hasattr(grad_method, "stepsize"):
        grad_method.stepsize = fd_step

    # Numerical Hessian via *finite difference of analytic MP2 gradients*
    # Use the general driver added to PySCF (tools.finite_diff). This takes a GradientsBase
    # object and returns the full Cartesian Hessian (natm,natm,3,3) including nuclear repulsion.
    hobj = fdtools.Hessian(grad_method)  # grad_method = mp2_solver.nuc_grad_method()
    hobj.displacement = float(fd_step)   # in Bohr; central difference
    hess = hobj.kernel()
    out["hessian"] = hess

    # Harmonic analysis / thermo
    print("\n=== Harmonic analysis / frequencies (MP2) ===")
    out.update(
        _harmonic_analysis_from_hessian(
            mf,
            hess,
            project=project,
            freqtemps=freqtemps,
        )
    )
    return out
