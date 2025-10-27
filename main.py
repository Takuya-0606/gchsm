#!/usr/bin/env python3
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
import json
import os
import re
import sys
import traceback
from typing import List, Union, Dict, Tuple, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from pyscf import gto

# Local modules
from .driver.calctype import run_sp, run_opt, run_hess, run_mp2, run_sp_mp2, run_opt_mp2, thermo_tables_from_freqs, thermo_from_loaded_freqs
from .driver.method   import build_scf
from .driver.hsm      import parse_hsm, attach_pcm
from .driver.report   import build_report
from .driver.nonelst  import parse_nonelst_config, evaluate_nonelst
from .driver.freqonly import analyze_plaintext_freqout


# time helpers
def now_tokyo() -> datetime:
    """Return current datetime in Asia/Tokyo timezone."""
    return datetime.now(ZoneInfo("Asia/Tokyo"))


def ts_legacy(dt: datetime) -> str:
    """Format datetime like 'Mon Oct 06 12:34:56 2025'."""
    return dt.strftime("%a %b %d %H:%M:%S %Y")


# input parsing
def read_sections(path: str) -> Tuple[str, str, str]:
    """
    Read entire input file and extract %MAIN and %GEOMETRY blocks.
    Returns:
        main_clean: %MAIN content without blank lines or comment lines
        geometry_raw: raw %GEOMETRY content (lines kept as-is)
        full_text: full file text (may be used for comment extraction elsewhere)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    m_main = re.search(r"%MAIN(.*?)%END", raw, flags=re.S | re.I)
    m_geom = re.search(r"%GEOMETRY(.*?)%END", raw, flags=re.S | re.I)
    if not m_main or not m_geom:
        raise ValueError("%MAIN section or %GEOMETRY section was not found.")

    # Clean %MAIN: strip blanks and remove lines starting with '#'
    lines = []
    for ln in m_main.group(1).splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    main_clean = "\n".join(lines)

    geometry_raw = m_geom.group(1).strip()
    return main_clean, geometry_raw, raw


def parse_keyvals(main: str) -> Dict[str, str]:
    """Parse KEY=VALUE lines from %MAIN block."""
    kv: Dict[str, str] = {}
    for ln in main.splitlines():
        m = re.match(r"([A-Za-z_]+)\s*=\s*(.+)$", ln)
        if m:
            key = m.group(1).upper()
            val = m.group(2).strip()
            kv[key] = val
    return kv


def parse_geometry_block(geometry: str) -> Tuple[int, int, str]:
    """
    Parse %GEOMETRY block.
    First non-empty line: "<charge> <multiplicity>"
    Remaining lines: atomic coordinates (Angstrom)
    """
    lines = [ln.strip() for ln in geometry.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("%GEOMETRY section is too short.")
    head = lines[0]
    m = re.match(r"([+-]?\d+)\s+(\d+)", head)
    if not m:
        raise ValueError("First line of %GEOMETRY must be 'charge multiplicity'.")
    charge = int(m.group(1))
    multiplicity = int(m.group(2))
    geom_text = "\n".join(lines[1:])
    return charge, multiplicity, geom_text

def _parse_freqtemps(value: Union[str, float, int, list, tuple, None]) -> List[float]:
    """Return a list of temperatures (K) from various input forms.
    Accepts:
      - None -> [298.15]
      - "298.15, 300  350" (commas/whitespace)
      - 298.15 (single number)
      - [298.15, "300", 350] (mixed list/tuple)
    """
    default = [298.15]
    if value is None:
        return default
    # string path
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        parts = re.split(r"[,\s;]+", s)
        out: List[float] = []
        for p in parts:
            if not p:
                continue
            try:
                out.append(float(p))
            except Exception:
                # ignore tokens that are not numbers
                pass
        return out or default

    # single number
    if isinstance(value, (int, float)):
        return [float(value)]

    # iterable path (list/tuple/etc.)
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            out.extend(_parse_freqtemps(item))  # recursive flatten
        return out or default

    # any other type -> default
    return default

def build_mol(geom_text: str, basis: str, charge: int, multiplicity: int) -> gto.Mole:
    """Construct a PySCF Mole in Angstrom with given basis/charge/multiplicity."""
    mol = gto.M(
        atom=geom_text,
        unit="Angstrom",
        basis=basis,
        charge=charge,
        spin=multiplicity - 1,
        verbose=4,
    )
    return mol


# main
def main():
    # Start timestamp (printed in the report)
    t_begin = now_tokyo()

    # Set up default containers so we can always build a report in finally
    results: Dict[str, object] = {}
    mol = None
    mf  = None
    used_pcm_method = ""              # Reserved for future use
    opt_coords_ang: Optional[list[list[float]]] = None
    inp_path: Optional[str] = None
    kv: Dict[str, str] = {}
    hsm_cfg: Dict[str, object] = {}

    nonelst_cfg = None

    method_lower = ""
    freqtemps: List[float] = [298.15]
    calctype = "sp"
    freq_only_mode = False
    freq_mole_kind = "NONLINEAR"
    ntr_expected = 6

    try:
        # parse CLI
        if len(sys.argv) < 2:
            raise ValueError("Usage: python3 main.py input.inp")
        inp_path = sys.argv[1]

        # read & parse input sections
        main_sec, geom_sec, _full_text = read_sections(inp_path)
        kv = parse_keyvals(main_sec)

        # CALCTYPE - strict validation (no alias acceptance)
        calctype = (kv.get("CALCTYPE", "sp") or "sp").strip().lower()
        if calctype not in {"sp", "opt", "hess", "freq"}:
            raise ValueError(f"Unknown CALCTYPE: {kv.get('CALCTYPE')}")

        freq_only_mode = (calctype == "freq")
        if freq_only_mode:
            freq_mole_kind = (kv.get("MOLE", "NONLINEAR") or "NONLINEAR").strip().upper()
            if freq_mole_kind not in {"LINEAR", "NONLINEAR"}:
                raise ValueError("When CALCTYPE=freq, MOLE must be LINEAR or NONLINEAR.")
            ntr_expected = 5 if freq_mole_kind == "LINEAR" else 6
      
        # method options
        method   = kv.get("METHOD", "HF").strip()
        method_lower = method.lower()
        dfttyp   = (kv.get("DFTTYP", "") or "").strip() or None
        basis    = kv.get("BASIS", "sto-3g").strip()
        hsm_cfg  = parse_hsm(kv.get("HSM"))
        nonelst_cfg = parse_nonelst_config(kv)

        
        # optional PROJECT switch (effective only for hess)
        project_flag = (kv.get("PROJECT", "TRUE").strip().upper() == "TRUE")

        # freqtemp
        freqtemps = _parse_freqtemps(kv.get("FREQTEMP"))

        # geometry
        charge, mult, geom_text = parse_geometry_block(geom_sec)
        mol = build_mol(geom_text, basis=basis, charge=charge, multiplicity=mult)

        if freq_only_mode:
            mf = None
            print("[info] CALCTYPE=freq detected: skipping hessian matrix calculation")
        else:
            scf_method = method
            if method_lower == "mp2":
                scf_method = "HF"
                print("[info] METHOD=MP2 detected: building HF/UHF reference for MP2 correlation")
            # SCF + PCM
            mf  = build_scf(mol, method=scf_method, dfttyp=dfttyp)
            mf  = attach_pcm(mf, mol, hsm_cfg)

            if method_lower == "mp2":
                ref_name = getattr(getattr(mf, "__class__", None), "__name__", "") or "HF"
                results["mp2_reference"] = ref_name.upper()
              
        if nonelst_cfg and nonelst_cfg.enable and not freq_only_mode:
            if nonelst_cfg.cavity_kind == "pcm" and not hsm_cfg.get("enable", True):
                raise ValueError("NONELST requires PCM but HSM=FALSE was specified.")
            ne_data = evaluate_nonelst(mf, mol, nonelst_cfg)
            if ne_data:
                results["non_electrostatic"] = ne_data

        freq_out_path = None
        
        # run calculation AFTER conditions are known (report prints them first)
        if freq_only_mode:
            freq_out_path = os.path.join(os.path.dirname(os.path.abspath(inp_path)), "freq.out")
            if not os.path.isfile(freq_out_path):
                raise FileNotFoundError(f"freq.out was not found at {freq_out_path}")

            with open(freq_out_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            freq_payload = None
            freqs_loaded: List[complex] = []
            freq_source_label: Optional[str] = None
            try:
                freq_payload = json.loads(raw_text)
            except json.JSONDecodeError:
                freq_payload = None

            if isinstance(freq_payload, dict) and "frequencies_cm-1" in freq_payload:
                # Legacy JSON cache support (kept for backward compatibility)
                freqs_primary = np.asarray(freq_payload.get("frequencies_cm-1", []), dtype=float)
                tr_loaded = np.asarray(freq_payload.get("tr_frequencies_cm-1", []), dtype=float)
                rot_loaded = np.asarray(freq_payload.get("rot_frequencies_cm-1", []), dtype=float)
                raw_payload = freq_payload.get("raw_frequencies_cm-1")
                proj_payload = freq_payload.get("projected_frequencies_cm-1")
                raw_loaded = np.asarray(raw_payload if raw_payload is not None else [], dtype=float)
                projected_loaded = np.asarray(proj_payload if proj_payload is not None else [], dtype=float)
                stored_project = bool(freq_payload.get("project", False))
                stored_mass = freq_payload.get("total_mass_amu")

                if project_flag and projected_loaded.size:
                    freqs_selected = projected_loaded
                    project_used = True
                elif (not project_flag) and raw_loaded.size:
                    freqs_selected = raw_loaded
                    project_used = False
                elif projected_loaded.size and not raw_loaded.size:
                    freqs_selected = projected_loaded
                    project_used = True
                elif raw_loaded.size and not projected_loaded.size:
                    freqs_selected = raw_loaded
                    project_used = False
                else:
                    freqs_selected = freqs_primary
                    project_used = stored_project

                freqs_loaded = [complex(val) for val in np.asarray(freqs_selected, dtype=float).ravel().tolist()]
                freq_source_label = "freq.out (legacy JSON)"

                if stored_mass is not None:
                    try:
                        results["total_mass_amu"] = float(stored_mass)
                    except Exception:
                        pass

                if project_flag != project_used:
                    if project_flag and not projected_loaded.size:
                        print("[warn] PROJECT=TRUE requested but projected frequencies unavailable; using stored data.")
                    elif (not project_flag) and not raw_loaded.size:
                        print("[warn] PROJECT=FALSE requested but raw frequencies unavailable; using stored data.")
                print("[info] Loaded legacy JSON frequency cache from freq.out")

            else:
                try:
                    analysis = analyze_plaintext_freqout(raw_text, freqtemps, ntr_expected)
                except ValueError as parse_exc:
                    raise ValueError(f"freq.out at {freq_out_path} did not contain readable frequencies") from parse_exc

                freq_all = analysis.freq_all.astype(float, copy=False)
                results["frequencies_cm-1"] = freq_all.ravel().tolist()
                results["project"] = False
                results["freq_source"] = "freq.out (table)"
                results["nTR"] = analysis.ntr_total
                results["tr_frequencies_cm-1"] = np.concatenate(
                    [analysis.freq_trans, analysis.freq_rot]
                ).ravel().astype(float).tolist()
                results["freq_trans_cm-1"] = analysis.freq_trans.ravel().astype(float).tolist()
                results["freq_rot_cm-1"] = analysis.freq_rot.ravel().astype(float).tolist()
                results["freq_vib_cm-1"] = analysis.freq_vib.ravel().astype(float).tolist()
                results["thermo_tables"] = analysis.thermo_tables
                if analysis.mode_table_text:
                    results["freq_mode_table_text"] = analysis.mode_table_text
                if analysis.thermo_text:
                    results["freq_thermo_text"] = analysis.thermo_text
                if analysis.per_temperature:
                    results["freq_thermo_per_temperature"] = [
                        {
                            "T": rec["T"],
                            "E": {k: float(v) for k, v in rec["E"].items()},
                            "S": {k: float(v) for k, v in rec["S"].items()},
                        }
                        for rec in analysis.per_temperature
                    ]
                if analysis.mode_table_text:
                    print(analysis.mode_table_text)
                if analysis.thermo_text:
                    print("")
                    print(analysis.thermo_text)
                if project_flag:
                    print("[warn] PROJECT=TRUE ignored for CALCTYPE=freq; using stored frequencies as-is.")
                
            results["freq_molecule_kind"] = freq_mole_kind
            results["freq_temperatures_K"] = [float(t) for t in freqtemps]
            if mol is not None:
                try:
                    results["total_mass_amu"] = float(np.sum(mol.atom_mass_list(isotope_avg=True)))
                except Exception:
                    pass

            print(f"[info] Loaded {len(freqs_loaded)} frequencies from {freq_out_path}")

        elif calctype == "sp":
            if method_lower == "mp2":
                results.update(run_sp_mp2(mf))
            else:
                results.update(run_sp(mf))
        elif calctype == "opt":
            if method_lower == "mp2":
                results.update(run_opt_mp2(mf))
            else:
                results.update(run_opt(mf))
            opt_coords_ang = results.get("opt_coords_ang")  # optional in report
        elif calctype == "hess":
            if method_lower == "mp2":
                results.update(run_mp2(mf, project=project_flag, freqtemps=freqtemps))
            else:
                results.update(run_hess(mf, project=project_flag, freqtemps=freqtemps))

    except Exception as ex:
        # Capture traceback string so report can show "Traceback ... " onward
        tb = traceback.format_exc()
        results["error"] = f"{type(ex).__name__}: {ex}"
        results["exception"] = tb

    finally:
        # End timestamp at the very end (printed in the report)
        t_end = now_tokyo()

        # Always build and write the .out, even on errors or partial setup
        try:
            report = build_report(
                inp_path=inp_path or "input.inp",
                kv=kv,
                hsm_cfg=hsm_cfg,
                results=results,
                mol=(mf.mol if (mf is not None and hasattr(mf, "mol")) else mol),
                mf=(mf if mf is not None else object()),
                used_pcm_method=used_pcm_method,
                opt_coords_ang=opt_coords_ang,
                started_at=ts_legacy(t_begin),
                ended_at=ts_legacy(t_end),
            )
        except Exception:
            # As a last resort, emit a minimal plain-text message
            # if report generation itself fails.
            fallback = []
            fallback.append("GCHSM REPORT GENERATION FAILED")
            fallback.append("")
            fallback.append(f"Execution of GCHSM begun {ts_legacy(t_begin)}")
            fallback.append(f"Execution of GCHSM terminated with error {ts_legacy(t_end)}")
            if "exception" in results:
                tb = results.get("exception") or ""
                anchor = "Traceback (most recent call last):"
                i = tb.find(anchor)
                fallback.append("")
                fallback.append("Detail:")
                fallback.append(tb[i:].rstrip() if i >= 0 else str(tb).rstrip())
                fallback.append("")
            report = "\n".join(fallback)

        outpath = os.path.splitext(inp_path or "input.inp")[0] + ".out"
        try:
            if calctype != "freq" and "frequencies_cm-1" in results:
                freq_out_path = freq_out_path or os.path.join(os.path.dirname(os.path.abspath(inp_path or "input.inp")), "freq.out")
                freqs_to_write = np.asarray(results.get("frequencies_cm-1", [])).ravel()
                with open(freq_out_path, "w", encoding="utf-8") as f_freq:
                    for freq_val in freqs_to_write:
                        if np.iscomplexobj(freq_val) and getattr(freq_val, "imag", 0.0) != 0.0:
                            f_freq.write(f"i {abs(float(np.imag(freq_val))):.12f}\n")
                        else:
                            f_freq.write(f"{float(np.real(freq_val)):.12f}\n")
                print(f"[info] Frequency summary written to {freq_out_path}")
        except Exception as dump_exc:
            print(f"[warn] Failed to write freq.out: {dump_exc}")

        with open(outpath, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[done] Results saved to {outpath}")


if __name__ == "__main__":
    main()
