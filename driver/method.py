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
from typing import Optional, Tuple
import re
import importlib

from pyscf import scf, dft


# -------------------- parsing helpers -------------------- #
def _split_xc_and_disp(xc_raw: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Split a possibly-suffixed functional into (base_xc, disp_tag).
    disp_tag in {'d3bj','d3zero','d4', None}.

    Examples:
        "B3LYPG-D3BJ"  -> ("B3LYPG", "d3bj")
        "PBE0-GD3"     -> ("PBE0",   "d3zero")
        "WB97XD3"      -> ("WB97X",  "d3zero")
        "PBE-D4"       -> ("PBE",    "d4")
        "B3LYP"        -> ("B3LYP",  None)
        None           -> ("",       None)
    """
    if not xc_raw:
        return "", None

    s = xc_raw.strip()
    up = s.upper()

    disp: Optional[str] = None

    # D4 first (avoid matching D3 within D4)
    if "D4" in up:
        disp = "d4"
        # remove all variants like "-D4" or "D4"
        s = re.sub(r'(?i)-?D4', '', s)

    # D3BJ (and GD3BJ)
    elif "D3BJ" in up or "GD3BJ" in up:
        disp = "d3bj"
        s = re.sub(r'(?i)-?G?D3BJ', '', s)

    # D3 (and GD3)
    elif "D3" in up or "GD3" in up:
        disp = "d3zero"
        s = re.sub(r'(?i)-?G?D3', '', s)

    # cleanup separators duplicated by removal
    s = re.sub(r'[\s\-_]+$', '', s).strip()
    s = re.sub(r'[\s\-_]{2,}', '-', s)

    # Some known patterns like "WB97XD3" become "WB97X" (valid).
    return s, disp


def _check_disp_backend(disp: Optional[str]) -> None:
    """
    Best-effort import check so that the user gets an early, clear warning.
    PySCF side will still attempt to handle it during kernel().
    """
    if not disp:
        return
    mod = None
    try:
        if disp == "d4":
            mod = importlib.import_module("pyscf.dispersion.dftd4")
        else:  # d3zero or d3bj
            mod = importlib.import_module("pyscf.dispersion.dftd3")
    except Exception as e:
        tag = "D4" if disp == "d4" else ("D3(BJ)" if disp == "d3bj" else "D3")
        print(f"[warn] Dispersion {tag} requested but backend import failed: {e}")
        print("       Install/enable pyscf-dispersion so that gradients/Hessians include dispersion.")
        return

    if mod is not None:
        tag = "D4" if disp == "d4" else ("D3(BJ)" if disp == "d3bj" else "D3")
        print(f"[info] Dispersion backend detected: {tag} via {mod.__name__}")


# ------------------------- API ---------------------------- #
def build_scf(mol, *, method: str, dfttyp: Optional[str]):
    """
    Create and configure a PySCF SCF object (HF/DFT), with optional D3/D4.

    Args
    ----
    mol : pyscf.gto.Mole
    method : "HF" or "DFT"
    dfttyp : If method == "DFT", the functional name possibly suffixed with
             -D3BJ / -D3 / -GD3BJ / -GD3 / -D4. For HF, can be None or a
             bare dispersion hint (we'll parse and apply mf.disp).

    Returns
    -------
    mf : SCF object ready for use.  (mf.disp is set when dispersion requested)
    """
    method_u = (method or "").upper().strip()
    open_shell = bool(getattr(mol, "spin", 0))

    if method_u == "HF":
        mf = scf.UHF(mol) if open_shell else scf.RHF(mol)
        # Allow specifying dispersion via DFTTYP like "D3BJ" or "HF-D3BJ"
        base, disp = _split_xc_and_disp(dfttyp)
        # base is ignored for HF
        if disp:
            mf.disp = disp
            _check_disp_backend(disp)

    elif method_u == "DFT":
        if not dfttyp:
            raise ValueError("DFTTYP must be provided for METHOD=DFT.")
        base_xc, disp = _split_xc_and_disp(dfttyp)
        if not base_xc:
            raise ValueError(f"Invalid DFTTYP after stripping dispersion: {dfttyp!r}")
        mf = dft.UKS(mol) if open_shell else dft.RKS(mol)
        mf.xc = base_xc
        if disp:
            mf.disp = disp
            _check_disp_backend(disp)
    else:
        raise ValueError(f"Unknown METHOD: {method!r} (expected 'HF' or 'DFT')")

    # Generic SCF knobs
    mf.conv_tol = 1e-9
    mf.max_cycle = 200

    # Minimal logging for clarity
    try:
        mname = "UHF" if open_shell and method_u == "HF" else \
                "RHF" if method_u == "HF" else \
                "UKS" if open_shell else "RKS"
        if method_u == "DFT":
            disp_msg = f", disp={getattr(mf, 'disp', 'none')}" if hasattr(mf, "disp") else ""
            print(f"[info] Built {mname} with xc={mf.xc}{disp_msg}")
        else:
            disp_msg = f" with disp={getattr(mf, 'disp', 'none')}" if hasattr(mf, "disp") else ""
            print(f"[info] Built {mname}{disp_msg}")
    except Exception:
        pass

    return mf
