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
from typing import Dict, Tuple

ATM2PA = 101_325.0
J2KCAL = 1.0 / 4184.0
NA = 6.022_140_76e23
RGAS = 8.314_462_618


def _y_from_Vm_sigma1(Vm_cm3_mol: float, sigma1_A: float) -> float:
    Vm_m3 = Vm_cm3_mol * 1e-6
    sigma1_m = sigma1_A * 1e-10
    n_number = NA / Vm_m3
    return (math.pi / 6.0) * n_number * (sigma1_m ** 3)


def _r_from_volume(V_ang3: float) -> float:
    return ((3.0 * V_ang3) / (4.0 * math.pi)) ** (1.0 / 3.0)


def _R_from_r_sigma1(r_A: float, sigma1_A: float) -> float:
    return 2.0 * r_A / sigma1_A - 1.0


def _g_dimensionless(
    y: float,
    R: float,
    T: float,
    P_value: float,
    Vm_cm3_mol: float,
    P_unit: str = "atm",
) -> float:
    if P_unit.lower() == "atm":
        P = P_value * ATM2PA
    elif P_unit.lower() == "pa":
        P = P_value
    else:
        raise ValueError("P_unit must be 'atm' or 'pa'.")
    Vm_m3_mol = Vm_cm3_mol * 1e-6

    u = y / (1.0 - y)
    term0 = -math.log(1.0 - y)
    term1 = 3.0 * u * R
    term2 = (3.0 * u + 4.5 * (u ** 2)) * (R ** 2)
    term3 = (P * Vm_m3_mol / (RGAS * T)) * (R ** 3) * y
    return term0 + term1 + term2 + term3


def cavitation_energy_spt(
    V_ang3: float,
    T: float,
    sigma1_A: float,
    Vm_cm3_mol: float,
    alpha_P: float,
    P_value: float = 1.0,
    P_unit: str = "atm",
) -> Tuple[float, float, float]:
    r_A = _r_from_volume(V_ang3)
    Rratio = _R_from_r_sigma1(r_A, sigma1_A)
    y = _y_from_Vm_sigma1(Vm_cm3_mol, sigma1_A)

    g = _g_dimensionless(y, Rratio, T, P_value, Vm_cm3_mol, P_unit=P_unit)
    Gc_J = RGAS * T * g

    if P_unit.lower() == "atm":
        P = P_value * ATM2PA
    else:
        P = P_value

    Vm_m3_mol = Vm_cm3_mol * 1e-6
    poly = (1.0 - y) ** 2 + 3.0 * (1.0 - y) * Rratio + 3.0 * (1.0 + 2.0 * y) * (Rratio ** 2)
    Hc_J = (
        y * alpha_P * RGAS * (T ** 2) * poly / ((1.0 - y) ** 3)
        + y * P * Vm_m3_mol * (Rratio ** 3) / (RGAS * T)
    )

    TS_J = Hc_J - Gc_J
    return Gc_J * J2KCAL, Hc_J * J2KCAL, TS_J * J2KCAL


def cavitation_energy_gamma_model(
    S_ang2: float,
    V_ang3: float,
    gamma_J_m2: float,
    P_Pa: float = 0.0,
) -> float:
    S_m2 = S_ang2 * 1e-20
    V_m3 = V_ang3 * 1e-30
    G_J = gamma_J_m2 * S_m2 + P_Pa * V_m3
    return G_J * J2KCAL
