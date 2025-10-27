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
import re
from typing import Tuple, Dict

def read_sections(path: str) -> Tuple[str, str]:
    """Return (%MAIN text, %GEOMETRY text)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    m_main = re.search(r"%MAIN(.*?)%END", raw, flags=re.S | re.I)
    m_geom = re.search(r"%GEOMETRY(.*?)%END", raw, flags=re.S | re.I)
    if not m_main or not m_geom:
        raise ValueError("%MAIN section or %GEOMETRY section was not found.")
    # Strip comments/blank lines in %MAIN
    lines = []
    for ln in m_main.group(1).splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    main_clean = "\n".join(lines)
    return main_clean, m_geom.group(1).strip()

def parse_keyvals(main: str) -> Dict[str, str]:
    """Parse KEY = VALUE lines in %MAIN."""
    kv: Dict[str, str] = {}
    for ln in main.splitlines():
        m = re.match(r"([A-Za-z_]+)\s*=\s*(.+)$", ln)
        if m:
            k, v = m.group(1).upper(), m.group(2).strip()
            kv[k] = v
    return kv

def parse_geometry_block(geometry: str) -> Tuple[int, int, str]:
    """Return (charge, multiplicity, xyz block in Angstrom)."""
    lines = [ln.strip() for ln in geometry.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("%GEOMETRY section is too short.")
    first = lines[0]
    m = re.match(r"([+-]?\d+)\s+(\d+)", first)
    if not m:
        raise ValueError("First line of %GEOMETRY must be 'charge multiplicity'.")
    charge = int(m.group(1))
    multiplicity = int(m.group(2))
    geom_text = "\n".join(lines[1:])
    return charge, multiplicity, geom_text
