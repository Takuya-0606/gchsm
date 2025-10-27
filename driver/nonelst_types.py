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
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Atom:
    """Solute atom stored in Angstrom units."""

    element: str
    position: np.ndarray  # Å


@dataclass
class Site:
    """Solvent site specification used by the GAMESS-89 DISREP model."""

    label: str
    Nt: int
    Rt_A: float
    DKT: float = 1.0
    RWT_A: float | None = None

    def radii_pair(self) -> Tuple[float, float]:
        """Return (Rt, RWT) with defaulting of the weighted radius."""
        Rt = float(self.Rt_A)
        RWT = float(self.RWT_A if self.RWT_A is not None else Rt)
        return Rt, RWT


VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}
"""Bondi-style van der Waals radii (Å) used for SES construction."""


def rho_from_density(density_g_cm3: float, molar_mass_g_mol: float) -> float:
    """Convert macroscopic density to molecular number density [Å⁻³]."""
    NA = 6.022_140_76e23
    rho_m3 = (density_g_cm3 * 1e3) / (molar_mass_g_mol / 1e3) * NA  # 1/m^3
    return float(rho_m3 * 1e-30)
