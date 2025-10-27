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
import os
from collections.abc import Iterable
from typing import Dict, Any, Optional
import numpy as np

W = 58
STAR_LINE = "*" * W
DASH_LINE = "-" * W

_ASCII_LETTERS = {
    "G": [
        "  GGGGGGG  ",
        "  G        ",
        "  G        ",
        "  G  GGGG  ",
        "  G     G  ",
        "  G     G  ",
        "  GGGGGGG  ",
    ],
    "C": [
        "  CCCCCCC  ",
        "  C        ",
        "  C        ",
        "  C        ",
        "  C        ",
        "  C        ",
        "  CCCCCCC  ",
    ],
    "H": [
        "  H     H  ",
        "  H     H  ",
        "  H     H  ",
        "  HHHHHHH  ",
        "  H     H  ",
        "  H     H  ",
        "  H     H  ",
    ],
    "S": [
        "  SSSSSSS  ",
        "  S        ",
        "  S        ",
        "  SSSSSSS  ",
        "        S  ",
        "        S  ",
        "  SSSSSSS  ",
    ],
    "M": [
        "  M     M  ",
        "  MM   MM  ",
        "  M M M M  ",
        "  M  M  M  ",
        "  M     M  ",
        "  M     M  ",
        "  M     M  ",
    ],
}

def _ascii_gchsm_lines() -> list[str]:
    letters = ["G", "C", "H", "S", "M"]
    return [" ".join(_ASCII_LETTERS[ch][i] for ch in letters).rstrip() for i in range(7)]

def header_block(version: str = "1.1") -> str:
    lines = [STAR_LINE, ""]
    lines.extend(_ascii_gchsm_lines())
    lines.append("")
    lines.append(f"  VERSION {version} (Oct 2025)")
    lines.append("")
    lines.append(STAR_LINE)
    return "\n".join(lines)

def input_comment_block(inp_path: str) -> str:
    if not os.path.isfile(inp_path):
        return ""
    with open(inp_path, "r", encoding="utf-8") as f:
        src = f.read().splitlines()
    raw = [ln for ln in src if ln.lstrip().startswith("#")]
    if not raw:
        return ""
    cleaned = []
    for ln in raw:
        s = ln.lstrip()
        if s.startswith("#"): s = s[1:]
        if s.startswith(" "): s = s[1:]
        cleaned.append(s)
    return f"{DASH_LINE}\n" + "\n".join(cleaned) + f"\n{DASH_LINE}"

def _fmt_table(rows: Iterable[Iterable[str]], sep: str = "  ") -> str:
    rows = [list(r) for r in rows]
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    widths = [0] * ncol
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(cell))
    lines = []
    for r in rows:
        padded = [(r[j] if j < len(r) else "").ljust(widths[j]) for j in range(ncol)]
        lines.append(sep.join(padded).rstrip())
    return "\n".join(lines)

# --- core blocks --------------------------------------------------------------

def system_block(mol, mf) -> str:
    if mol is None:
        return "SYSTEM\n......\n(unavailable: molecule was not built)"
    try:
        nat = getattr(mol, "natm", None)
        if nat is None: nat = len(mol._atom)
    except Exception:
        nat = 0
    try:
        ne = getattr(mol, "nelectron", 0)
    except Exception:
        ne = 0
    try:
        chg  = int(getattr(mol, "charge", 0))
    except Exception:
        chg = 0
    try:
        mult = int(getattr(mol, "spin", 0)) + 1
    except Exception:
        mult = 1
    try:
        nbf = mol.nao_nr()
    except Exception:
        try: nbf = mol.nao()
        except Exception: nbf = 0

    nocc = None
    try:
        mo_occ = getattr(mf, "mo_occ", None)
        if mo_occ is not None:
            if isinstance(mo_occ, (tuple, list)):
                nocc = int(np.count_nonzero(mo_occ[0] > 0) + np.count_nonzero(mo_occ[1] > 0))
            else:
                nocc = int(np.count_nonzero(mo_occ > 0))
    except Exception:
        nocc = None
    if nocc is None and isinstance(ne, int):
        nocc = ne // 2

    rows = [
        ("Total number of atoms",       f"{int(nat):8d}" if isinstance(nat, int) else str(nat)),
        ("Number of electrons",         f"{int(ne):8d}"  if isinstance(ne, int) else str(ne)),
        ("Charge of system",            f"{int(chg):8d}"),
        ("Spin multiplicity",           f"{int(mult):8d}"),
        ("Number of basis functions",   f"{int(nbf):8d}" if isinstance(nbf, int) else str(nbf)),
        ("Number of occupied orbitals", f"{int(nocc):8d}" if isinstance(nocc, int) else str(nocc)),
    ]
    return "------\nSYSTEM\n------\n" + _fmt_table(rows) + "\n"

def method_block(kv: Dict[str, str], d3_applied: bool, results: Dict[str, Any] | None = None) -> str:
    method = kv.get("METHOD", "").upper()
    dfttyp = kv.get("DFTTYP", "")
    basis  = kv.get("BASIS", "")
    calctype = (kv.get("CALCTYPE", "") or "").upper()

    if method == "FREQ":
        rows = [
            ("Method",         "FREQ (read freq.out)"),
            ("Reference",      results.get("freq_source", "freq.out") if results else "freq.out"),
            ("Basis",          basis or "-"),
            ("Dispersion",     "-"),
        ]
        return "---------------------\nComputational Details\n---------------------\n" + _fmt_table(rows)

    if calctype == "MP2":
        ref = method or "HF"
        rows = [
            ("Method",         f"MP2 (ref={ref})"),
            ("DFT functional", "-"),
            ("Basis",          basis or "-"),
            ("Dispersion",     "None"),
        ]
        return "---------------------\nComputational Details\n---------------------\n" + _fmt_table(rows)
    
    rows = [
        ("Method",         method or "-"),
        ("DFT functional", dfttyp if method == "DFT" else "-"),
        ("Basis",          basis or "-"),
        ("Dispersion",     "DFT-D3(BJ)/D3" if d3_applied else "None"),
    ]
    return "---------------------\nComputational Details\n---------------------\n" + _fmt_table(rows)

def pcm_block(hsm_cfg: Dict[str, Any], used_method: str) -> str:
    if not hsm_cfg or not hsm_cfg.get("enable", True):
        return "----------\nPCM Details\n------------\nDisabled"
    scrf  = str(hsm_cfg.get("SCRF", "CPCM"))
    eps   = float(hsm_cfg.get("EPS", 80.1510))
    alpha = float(hsm_cfg.get("ALPHA", 1.2))
    grid  = int(hsm_cfg.get("ORDER", 17))
    def _ff(x, n): return f"{x:.{n}f}"
    rows = [
        ("SCRF (PCM)",  used_method or scrf),
        ("Dielectric",  _ff(eps, 4)),
        ("vdW scale alpha", _ff(alpha, 3)),
        ("Lebedev order", str(grid)),
    ]
    return "-----------\nPCM Details\n-----------\n" + _fmt_table(rows) + "\n"


def nonelst_block(ne_data: Dict[str, Any] | None) -> str:
    if not ne_data:
        return ""

    surf = ne_data.get("surface", {})
    disp = ne_data.get("disp_rep", {})
    cav = ne_data.get("cavitation", {})
    total = ne_data.get("total_kcal_mol", None)

    lines: list[str] = []
    lines.append("------------------------------")
    lines.append("Non-electrostatic Contributions")
    lines.append("------------------------------")
    lines.append("")

    if surf:
        lines.append("[Surface]")
        lines.append(f"  Kind                   : {surf.get('kind', '-')}")
        lines.append(f"  Area (Bohr^2)          : {surf.get('S_A2', 0.0):.6f}")
        lines.append(f"  Volume (Bohr^3)        : {surf.get('V_A3', 0.0):.6f}")
        if surf.get("kind", "").upper() == "PCM":
            lines.append(f"  Number of surface point :  {surf.get('n_tessera', 0)}")
        elif surf.get("kind", "").upper() == "SES":
            lines.append(f"  Number of tesserae      : {surf.get('n_tessera', 0)}")
            lines.append(f"  Delta shift (Bohr.)   : {surf.get('delta_A', 0.0):.4f}")
        lines.append("")

    if disp:
        lines.append("[Dispersion / Repulsion]")
        lines.append(f"  Model             : {disp.get('model', '-')}")
        lines.append(f"  Density           : {disp.get('rho_A3', 0.0):.6e}")
        lines.append(f"  E_disp (kcal/mol) : {disp.get('Edisp_kcal_mol', 0.0): .6f}")
        lines.append(f"  E_rep  (kcal/mol) : {disp.get('Erep_kcal_mol', 0.0): .6f}")
        sites = disp.get("sites", []) or []
        if sites:
            rows = [("Label", "Nt", "Rt (Ang)", "DKT", "RWT (Ang)")]
            for s in sites:
                rows.append(
                    (
                        str(s.get("label", "-")),
                        str(s.get("Nt", "-")),
                        f"{float(s.get('Rt_A', 0.0)):.3f}",
                        f"{float(s.get('DKT', 0.0)):.3f}",
                        f"{float(s.get('RWT_A', 0.0)):.3f}",
                    )
                )
            lines.append("")
            lines.append("-----")
            lines.append("Sites")
            lines.append("-----")
            table = _fmt_table(rows)
            lines.extend([ln for ln in table.splitlines()])
        lines.append("")

    if cav:
        lines.append("[Cavitation]")
        lines.append(f"  Model             : {cav.get('model', '-')}")
        lines.append(f"  G_cav (kcal/mol)  : {cav.get('Ecav_kcal_mol', 0.0): .6f}")
        Hc = cav.get("Hc_kcal_mol")
        TSc = cav.get("TSc_kcal_mol")
        if Hc is not None:
            lines.append(f"  H_c   (kcal/mol)  : {Hc: .6f}")
        if TSc is not None:
            lines.append(f"  T*S_c (kcal/mol)  : {TSc: .6f}")
        lines.append("")

    if total is not None:
        lines.append(f"Total non-electrostatic energy (kcal/mol): {total: .6f}")

    return "\n".join(lines)

def projection_block(project: bool, ntr: int | None = None) -> str:
    line = f"Projection of trans/rot: {'ON' if project else 'OFF'}"
    if project and ntr is not None:
        line += f"  (nTR = {ntr})"
    return "-----------------\nVIBRATION OPTIONS\n-----------------\n" + line + "\n"

def scf_energy_block(results: Dict[str, Any]) -> str:
    if "scf_energy" not in results:
        return ""
    e = float(results["scf_energy"])
    rows = [
        ("SCF energy (Eh)",       f"{e:.12f}"),
        ("SCF energy (kJ/mol)",   f"{e*2625.499638: .2f}"),
        ("SCF energy (kcal/mol)", f"{e*627.509474: .2f}"),
    ]
    return "RESULTS: SINGLE POINT ENERGY\n----------------------------\n" + _fmt_table(rows)


def mp2_energy_block(results: Dict[str, Any]) -> str:
    if "mp2_total_energy" not in results:
        return ""

    e_tot = float(results["mp2_total_energy"])
    e_corr = float(results.get("mp2_correlation_energy", 0.0))

    rows = [
        ("MP2 total energy (Eh)",       f"{e_tot:.12f}"),
        ("MP2 corr. energy (Eh)",       f"{e_corr:.12f}"),
        ("MP2 total energy (kJ/mol)",   f"{e_tot*2625.499638: .2f}"),
        ("MP2 total energy (kcal/mol)", f"{e_tot*627.509474: .2f}"),
        ("MP2 corr. energy (kJ/mol)",   f"{e_corr*2625.499638: .2f}"),
        ("MP2 corr. energy (kcal/mol)", f"{e_corr*627.509474: .2f}"),
    ]
    return "RESULTS: MP2 ENERGY\n---------------------\n" + _fmt_table(rows)
    
def geometry_block(opt_coords_ang: Optional[Iterable[Iterable[float]]], symbols: Iterable[str] | None) -> str:
    if not opt_coords_ang:
        return ""
    if symbols is None:
        rows = [("Index", "X / Ang", "Y / Ang", "Z / Ang")]
        for i, (x, y, z) in enumerate(opt_coords_ang, 1):
            rows.append((f"{i}", f"{x: .8f}", f"{y: .8f}", f"{z: .8f}"))
        return "OPTIMIZED GEOMETRY (Angstrom)\n------------------------------\n" + _fmt_table(rows)
    rows = [("Atom", "X / Ang", "Y / Ang", "Z / Ang")]
    for sym, (x, y, z) in zip(symbols, opt_coords_ang):
        rows.append((sym, f"{x: .8f}", f"{y: .8f}", f"{z: .8f}"))
    return "OPTIMIZED GEOMETRY (Angstrom)\n------------------------------\n" + _fmt_table(rows)

def freq_block(results: Dict[str, Any]) -> str:
    if "frequencies_cm-1" not in results:
        return ""
    freqs = np.asarray(results["frequencies_cm-1"]).ravel()

    lines: list[str] = []
    lines.append("--------------------")
    lines.append("HARMONIC FREQUENCIES")
    lines.append("--------------------")
    lines.append("")
    lines.append("Mode  Frequency(cm-1)  Note")
    lines.append("---------------------------")

    for i, w in enumerate(freqs, 1):
        # robust formatting: treat complex freq as imaginary (Gaussian-like style)
        if np.iscomplexobj(w) and getattr(w, "imag", 0.0) != 0.0:
            val = abs(np.imag(w))
            lines.append(f"{i:4d}  {val:9.2f}  imag")
        else:
            val = float(np.real(w))
            note = "low" if val < 10.0 else ""
            lines.append(f"{i:4d}  {val:9.2f}  {note}")

    lines.append("---------------------------")
    return "\n".join(lines)

# --- new: thermochemistry block ---------------------------------------------

def _fmt_right(val: float, width: int = 10, prec: int = 2) -> str:
    return f"{val:>{width}.{prec}f}"

def thermo_blocks(results: Dict[str, Any]) -> str:
    tables = results.get("thermo_tables")
    if not tables:
        return ""
    total_mass = results.get("total_mass_amu", None)

    out_lines: list[str] = []
    for t in tables:
        T = float(t["T"])
        out_lines.append("-----------------------------")
        out_lines.append(f"Thermochemistry at {T:.2f} K")
        out_lines.append("-----------------------------")
        out_lines.append("")
        out_lines.append(f"Temperature      : {T:.2f}  K")
        out_lines.append(f"Pressure         : 1.00    atm")
        if total_mass is not None:
            out_lines.append(f"Total Mass       : {total_mass:.3f}  Amu")
        out_lines.append("")
        out_lines.append("---------------")
        out_lines.append("Internal energy")
        out_lines.append("---------------")
        E_kJ = t["E_kJmol"]; E_kcal = t["E_kcalmol"]
        out_lines.append("")
        out_lines.append(f"E(vib)  : {_fmt_right(E_kJ['vib'],10,2)} kJ/mol   {_fmt_right(E_kcal['vib'],8,2)} kcal/mol")
        out_lines.append(f"E(rot)  : {_fmt_right(E_kJ['rot'],10,2)} kJ/mol   {_fmt_right(E_kcal['rot'],8,2)} kcal/mol")
        out_lines.append(f"E(trans): {_fmt_right(E_kJ['trans'],10,2)} kJ/mol   {_fmt_right(E_kcal['trans'],8,2)} kcal/mol")
        out_lines.append(f"Total E : {_fmt_right(E_kJ['total'],10,2)} kJ/mol   {_fmt_right(E_kcal['total'],8,2)} kcal/mol")
        out_lines.append("")
        out_lines.append("-------")
        out_lines.append("Entropy")
        out_lines.append("-------")
        S_kJ = t["S_kJmolK"]
        out_lines.append("")
        out_lines.append(f"S(vib)  : {_fmt_right(S_kJ['vib'],10,5)} kJ/mol/K")
        out_lines.append(f"S(rot)  : {_fmt_right(S_kJ['rot'],10,5)} kJ/mol/K")
        out_lines.append(f"S(trans): {_fmt_right(S_kJ['trans'],10,5)} kJ/mol/K")
        out_lines.append(f"Total S : {_fmt_right(S_kJ['total'],10,5)} kJ/mol/K")
        out_lines.append("")
    return "\n".join(out_lines)

# --- top-level report ---------------------------------------------------------

def start_banner_for(calctype: str, method: str | None = None, results: Dict[str, Any] | None = None) -> str:
    method_u = (method or "").strip().lower()
    if calctype == "sp":
        return "*** Start single point energy calculation ***"
    if calctype == "opt":
        return "*** Start geometry optimization ***"
    if calctype == "hess":
        if method_u == "mp2" or (results and "mp2_total_energy" in results):
            return "*** Start MP2 vibrational frequency calculation ***"
        return "*** Start vibrational frequency calculation ***"
    if calctype == "freq":
        return "*** Start vibrational frequency analysis from freq.out ***"
    return "*** Start calculation ***"

def error_block(err_msg: str, exc_text: str | None = None) -> str:
    body = ""
    tb_anchor = "Traceback (most recent call last):"
    if exc_text:
        i = exc_text.find(tb_anchor)
        body = exc_text[i:].rstrip() if i >= 0 else exc_text.rstrip()
    if not body:
        body = (err_msg or "").strip()
    lines = ["", "!!! Error termination detected !!!", "", "Detail:", body, ""]
    return "\n".join(lines)

def build_report(
    inp_path: str,
    kv: Dict[str, str],
    hsm_cfg: Dict[str, Any],
    results: Dict[str, Any],
    mol,
    mf,
    used_pcm_method: str = "",
    opt_coords_ang: Optional[Iterable[Iterable[float]]] = None,
    *,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> str:
    d3_applied = "D3" in (kv.get("DFTTYP", "").upper())
    calctype = (kv.get("CALCTYPE", "sp") or "sp").strip().lower()

    parts: list[str] = []
    parts.append(header_block(version="1.0"))
    parts.append("")
    if started_at:
        parts.append(f"Execution of GCHSM begun {started_at}")
        parts.append("")

    comments = input_comment_block(inp_path)
    if comments:
        parts.append(comments)
        parts.append("")

    mol_obj = mf.mol if hasattr(mf, "mol") else mol
    parts.append(system_block(mol_obj, mf))
    parts.append("")
    if "frequencies_cm-1" in results:
        parts.append(projection_block(bool(results.get("project", False)), results.get("nTR")))
        parts.append("")
    parts.append(method_block(kv, d3_applied, results))
    parts.append("")
    parts.append(pcm_block(hsm_cfg, used_pcm_method))
    parts.append("")

    ne_section = nonelst_block(results.get("non_electrostatic"))
    if ne_section:
        parts.append(ne_section)
        parts.append("")
    parts.append(start_banner_for(calctype))
    parts.append("")

    if "scf_energy" in results:
        parts.append(scf_energy_block(results))
        parts.append("")
    if "mp2_total_energy" in results:
        parts.append(mp2_energy_block(results))
        parts.append("")
    if opt_coords_ang:
        try:
            symbols = [mol_obj.atom_symbol(i) for i in range(mol_obj.natm)] if mol_obj is not None else None
        except Exception:
            symbols = None
        parts.append(geometry_block(opt_coords_ang, symbols))
        parts.append("")
    if "frequencies_cm-1" in results:
        parts.append(freq_block(results))
        parts.append("")
    if results.get("thermo_tables"):
        parts.append(thermo_blocks(results))
        parts.append("")

    err_present = ("error" in results) or ("exception" in results)
    if ended_at:
        parts.append(f"Execution of GCHSM terminated {'with error' if err_present else 'normally'} {ended_at}")
    else:
        parts.append("Execution of GCHSM terminated")
    if err_present:
        parts.append(error_block(str(results.get("error", "")), results.get("exception")))
    return "\n".join(s for s in parts if s is not None)
