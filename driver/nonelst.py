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

"""Parsing and evaluation entry points for non-electrostatic energies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


from .nonelst_cavitation import cavitation_energy_gamma_model, cavitation_energy_spt
from .nonelst_dispersion import (
    build_atom_list,
    compute_disp_rep_gamess_from_surface,
    parse_site_tokens,
)
from .nonelst_surface import (
    build_ses_surface_from_geom,
    extract_pcm_surface,
    surface_area_from_tessera,
    volume_from_closed_surface,
)
from .nonelst_types import Site, VDW_RADII, rho_from_density


@dataclass
class NonElstConfig:
    enable: bool = False
    cavity_kind: str = "pcm"
    probe_radius: float = 1.58
    npoints_per_sphere: int = 8000
    delta_A: float = 0.0
    density_g_cm3: float = 0.997
    molar_mass_g_mol: float = 18.01528
    disp_model: str = "gamess89"
    sites: List[Site] = field(default_factory=list)
    cav_model: str = "spt"
    spt_params: Dict[str, float] = field(default_factory=dict)
    gamma_params: Dict[str, float] = field(default_factory=dict)


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    v = value.strip().upper()
    if v in {"TRUE", "T", "YES", "Y", "1"}:
        return True
    if v in {"FALSE", "F", "NO", "N", "0"}:
        return False
    raise ValueError(f"Invalid boolean token: {value}")


def parse_nonelst_config(kv: Dict[str, str]) -> NonElstConfig:
    cfg = NonElstConfig()
    cfg.enable = _parse_bool(kv.get("NONELST"), default=False)
    if not cfg.enable:
        return cfg

    cfg.cavity_kind = kv.get("NE_CAVITY", cfg.cavity_kind).strip().lower()
    if cfg.cavity_kind not in {"pcm", "ses"}:
        raise ValueError("NE_CAVITY must be PCM or SES.")

    cfg.probe_radius = float(kv.get("NE_PROBE_RADIUS", cfg.probe_radius))
    cfg.npoints_per_sphere = int(kv.get("NE_NPOINTS", cfg.npoints_per_sphere))
    cfg.delta_A = float(kv.get("NE_DELTA_A", cfg.delta_A))
    cfg.density_g_cm3 = float(kv.get("NE_DENSITY", cfg.density_g_cm3))
    cfg.molar_mass_g_mol = float(kv.get("NE_MOLAR_MASS", cfg.molar_mass_g_mol))

    cfg.disp_model = kv.get("NE_MODEL", cfg.disp_model).strip().lower()
    if cfg.disp_model != "gamess":
        raise ValueError("Currently only NE_MODEL=GAMESS is supported.")

    raw_sites = kv.get("NE_SITES", "").strip()
    if not raw_sites:
        raise ValueError("NE_SITES must be provided when NONELST is enabled.")
    site_tokens = [token.strip() for token in raw_sites.split(";")]
    cfg.sites = parse_site_tokens(site_tokens)

    cfg.cav_model = kv.get("NE_CAV_MODEL", cfg.cav_model).strip().lower()
    if cfg.cav_model not in {"spt", "gamma"}:
        raise ValueError("NE_CAV_MODEL must be SPT or GAMMA.")

    if cfg.cav_model == "spt":
        params = {
            "T": float(kv.get("NE_SPT_T", 298.15)),
            "sigma1_A": float(kv.get("NE_SPT_SIGMA", 1.5)),
            "Vm_cm3_mol": float(kv.get("NE_SPT_VM", 18.07)),
            "alpha_P": float(kv.get("NE_SPT_ALPHA", 2.57e-4)),
            "P": float(kv.get("NE_SPT_P", 1.0)),
            "Punit": kv.get("NE_SPT_PUNIT", "atm").strip(),
        }
        cfg.spt_params = params
    else:
        gamma = kv.get("NE_GAMMA_GAMMA", None)
        if gamma is None:
            raise ValueError("NE_GAMMA_GAMMA must be provided for GAMMA cavitation model.")
        params = {
            "gamma_J_m2": float(gamma),
            "P_Pa": float(kv.get("NE_GAMMA_PPA", 0.0)),
        }
        cfg.gamma_params = params

    return cfg


def _shift_centers(centers: np.ndarray, normals: np.ndarray, outward: bool, delta_A: float) -> np.ndarray:
    if abs(delta_A) < 1e-12:
        return centers
    sign = 1.0 if outward else -1.0
    return centers + sign * delta_A * normals


def evaluate_nonelst(mf, mol, cfg: NonElstConfig) -> Dict[str, object]:
    if not cfg.enable:
        return {}

    if cfg.cavity_kind == "pcm":
        if getattr(mf, "with_solvent", None) is None:
            raise RuntimeError("PCM cavity is required for NONELST with NE_CAVITY=PCM.")
        centers, normals, areas, types, outward = extract_pcm_surface(mf)
        centers = _shift_centers(centers, normals, outward, cfg.delta_A)
    else:
        centers, normals, areas, types, outward = build_ses_surface_from_geom(
            mol,
            VDW_RADII,
            probe_radius=cfg.probe_radius,
            npoints_per_sphere=cfg.npoints_per_sphere,
        )

    S = surface_area_from_tessera(areas)
    V = volume_from_closed_surface(centers, normals, areas, outward)

    atoms = build_atom_list(mol)
    rho = rho_from_density(cfg.density_g_cm3, cfg.molar_mass_g_mol)
    Edisp, Erep = compute_disp_rep_gamess_from_surface(
        centers, normals, areas, types, atoms, cfg.sites, rho, outward
    )

    cav_detail: Dict[str, object]
    Ecav: float
    Hc: Optional[float]
    TSc: Optional[float]

    if cfg.cav_model == "spt":
        params = cfg.spt_params
        Ecav, Hc, TSc = cavitation_energy_spt(
            V_ang3=V,
            T=params["T"],
            sigma1_A=params["sigma1_A"],
            Vm_cm3_mol=params["Vm_cm3_mol"],
            alpha_P=params["alpha_P"],
            P_value=params.get("P", 1.0),
            P_unit=params.get("Punit", "atm"),
        )
        cav_detail = {"model": "SPT", **params}
    else:
        params = cfg.gamma_params
        Ecav = cavitation_energy_gamma_model(
            S_ang2=S,
            V_ang3=V,
            gamma_J_m2=params["gamma_J_m2"],
            P_Pa=params.get("P_Pa", 0.0),
        )
        Hc = None
        TSc = None
        cav_detail = {"model": "gammaS+pV", **params}

    total = Edisp + Erep + Ecav

    site_summary = [
        {
            "label": site.label,
            "Nt": site.Nt,
            "Rt_A": site.Rt_A,
            "DKT": site.DKT,
            "RWT_A": site.RWT_A if site.RWT_A is not None else site.Rt_A,
        }
        for site in cfg.sites
    ]

    return {
        "surface": {
            "kind": cfg.cavity_kind.upper(),
            "S_A2": S,
            "V_A3": V,
            "n_tessera": int(len(areas)),
            "delta_A": cfg.delta_A if cfg.cavity_kind == "pcm" else 0.0,
            "outward": bool(outward),
        },
        "disp_rep": {
            "model": cfg.disp_model.upper(),
            "rho_A3": rho,
            "Edisp_kcal_mol": Edisp,
            "Erep_kcal_mol": Erep,
            "sites": site_summary,
        },
        "cavitation": {
            "model": cav_detail["model"],
            "detail": cav_detail,
            "Ecav_kcal_mol": Ecav,
            "Hc_kcal_mol": Hc,
            "TSc_kcal_mol": TSc,
        },
        "total_kcal_mol": total,
    }
