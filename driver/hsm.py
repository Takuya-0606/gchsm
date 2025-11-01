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
from typing import Dict, Any
from pyscf.solvent import pcm as pcm_mod

def _load_user_cavity_pcm():
    """Import UserCavityPCM from driver.fixcav and return the class or None."""
    try:
        # IMPORTANT: path must be project-root + 'driver'; __init__.py required
        from .fixcav import UserCavityPCM
        return UserCavityPCM
    except Exception as e:
        print(f"[warn] UserCavityPCM not available (driver/fixcav.py). Reason: {type(e).__name__}: {e}")
        return None

def parse_hsm(val: str | None) -> Dict[str, Any]:
    """
    Parse HSM=(SCRF=..., ALPHA=..., EPS=..., ORDER=...) or TRUE/FALSE.
    """
    defaults = dict(enable=True, SCRF="CPCM", ALPHA=1.2, EPS=80.1510, ORDER=17)
    if val is None:
        return defaults
    v = val.strip()
    if v.upper() == "TRUE":
        return defaults
    if v.upper() == "FALSE":
        d = defaults.copy(); d["enable"] = False; return d
    if not (v.startswith("(") and v.endswith(")")):
        raise ValueError(f"HSM format is incorrect: {v}")
    inner = v[1:-1]
    d = defaults.copy()
    for itm in inner.split(","):
        if not itm.strip():
            continue
        k, eq, vv = itm.partition("=")
        if eq != "=":
            continue
        k = k.strip().upper(); vv = vv.strip()
        if k == "SCRF":  d["SCRF"]  = vv.upper()
        elif k == "ALPHA": d["ALPHA"] = float(vv)
        elif k == "EPS":   d["EPS"]   = float(vv)
        elif k == "ORDER":  d["ORDER"]  = int(vv)
    return d

def attach_pcm(mf, mol, hsm_cfg: Dict[str, Any]):
    """Attach PCM. Prefer UserCavityPCM if available; print what got used."""
    if not hsm_cfg.get("enable", True):
        print("[info] PCM/HSM disabled")
        return mf

    scrf  = str(hsm_cfg.get("SCRF", "CPCM")).upper()
    eps   = float(hsm_cfg.get("EPS", 80.1510))
    alpha = float(hsm_cfg.get("ALPHA", 1.2))
    grid  = int(hsm_cfg.get("ORDER", 17))

    # Try to load the custom cavity implementation
    UserCavityPCM = _load_user_cavity_pcm()

    if UserCavityPCM is not None:
        # fixed cavity centers = current atom positions (Angstrom)
        centers_ang = mol.atom_coords(unit="Angstrom")
        cm = UserCavityPCM(mol, centers=centers_ang, eps=eps, method=scrf)
        if hasattr(cm, "vdw_scale"):       cm.vdw_scale = alpha
        if hasattr(cm, "lebedev_order"):   cm.lebedev_order = grid
        if hasattr(cm, "conv_tol"):        cm.conv_tol = 1e-12
        cm.build()
        mf = mf.PCM(cm)
        # Diagnostics
        try:
            ws = getattr(mf, "with_solvent", None)
            print(f"[info] Using UserCavityPCM: {type(ws).__name__}")
        except Exception:
            print("[info] Using UserCavityPCM (attached)")
        return mf

    # Fallback to standard PCM
    cm = pcm_mod.PCM(mol)
    cm.eps = eps
    cm.method = scrf
    if hasattr(cm, "vdw_scale"):     cm.vdw_scale = alpha
    if hasattr(cm, "lebedev_order"): cm.lebedev_order = grid
    if hasattr(cm, "conv_tol"):      cm.conv_tol = 1e-12
    cm.build()
    mf = mf.PCM(cm)
    try:
        ws = getattr(mf, "with_solvent", None)
        print(f"[info] Using standard PCM: {type(ws).__name__}")
    except Exception:
        print("[info] Using standard PCM (attached)")
    return mf
