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
# Author: Takuya HAHSIMOTO

from __future__ import annotations
import re
from typing import Dict, Tuple

def read_sections(path: str) -> Tuple[str, str]:
    """Read %MAIN and %GEOMETRY sections."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    m_main = re.search(r"%MAIN(.*?)%END", raw, flags=re.S | re.I)
    m_geom = re.search(r"%GEOMETRY(.*?)%END", raw, flags=re.S | re.I)
    if not m_main or not m_geom:
        raise ValueError("%MAIN or %GEOMETRY section not found.")
    main = "\n".join(
        ln.strip() for ln in m_main.group(1).splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )
    geometry = m_geom.group(1).strip()
    return main, geometry

def parse_keyvals(main: str) -> Dict[str, str]:
    """Parse KEY=VALUE lines inside %MAIN."""
    kv: Dict[str, str] = {}
    for ln in main.splitlines():
        m = re.match(r"([A-Za-z_]+)\s*=\s*(.+)$", ln)
        if m:
            kv[m.group(1).upper()] = m.group(2).strip()
    return kv

def parse_geometry_block(geometry: str) -> Tuple[int, int, str]:
    """Return (charge, multiplicity, geometry_text)."""
    lines = [ln.strip() for ln in geometry.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("%GEOMETRY too short.")
    m = re.match(r"([+-]?\d+)\s+(\d+)", lines[0])
    if not m:
        raise ValueError("First line in %GEOMETRY must be 'charge multiplicity'.")
    charge, mult = int(m.group(1)), int(m.group(2))
    geom_text = "\n".join(lines[1:])
    return charge, mult, geom_text

def parse_hsm(val: str | None):
    """Parse HSM=(SCRF=..., ALPHA=..., EPS=..., ORDER=...) or TRUE/FALSE."""
    defaults = dict(enable=True, SCRF="CPCM", ALPHA=1.2, EPS=80.1510, ORDER=17)
    if val is None:
        return defaults
    v = val.strip()
    if v.upper() == "TRUE":
        return defaults
    if v.upper() == "FALSE":
        d = defaults.copy(); d["enable"] = False; return d
    m = re.match(r"\(\s*(.*?)\s*\)$", v)
    if not m:
        raise ValueError(f"Invalid HSM format: {v}")
    d = defaults.copy()
    for item in m.group(1).split(","):
        if not item.strip():
            continue
        k, _, vv = item.partition("=")
        k, vv = k.strip().upper(), vv.strip()
        if k == "SCRF":   d["SCRF"] = vv.upper()
        elif k == "ALPHA": d["ALPHA"] = float(vv)
        elif k == "EPS":   d["EPS"] = float(vv)
        elif k == "ORDER":  d["ORDER"] = int(vv)
    return d

def get_method_params(kv: Dict[str, str]):
    """Return (METHOD, BASIS, DFTTYP[or None])."""
    method = kv.get("METHOD", "HF").strip()
    basis  = kv.get("BASIS", "sto-3g").strip()
    dfttyp = (kv.get("DFTTYP", "") or "").strip() or None
    return method, basis, dfttyp
