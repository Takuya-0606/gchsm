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
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# --- Physical constants -------------------------------------------------------
_PLANCK_H = 6.62607015e-34       # Planck constant [J·s]
_SPEED_OF_LIGHT_CM = 2.99792458e10  # speed of light [cm/s]
_BOLTZMANN_K = 1.380649e-23      # Boltzmann constant [J/K]
_GAS_CONSTANT_R = 8.31446261815324  # Gas constant [J/mol/K]

_KJ_PER_J = 1.0e-3
_KCAL_PER_J = 0.000239006


@dataclass
class FreqAnalysisResult:
    """Container holding frequency analysis artifacts."""

    freq_all: np.ndarray
    freq_trans: np.ndarray
    freq_rot: np.ndarray
    freq_vib: np.ndarray
    ntr_total: int
    thermo_tables: list[dict]
    per_temperature: list[dict]
    mode_table_text: str
    thermo_text: str


# ------------------------------------------------------------------------------
# Parsing utilities
# ------------------------------------------------------------------------------
def parse_plaintext_freqout(raw_text: str) -> list[float]:
    """Parse a freq.out-style plain-text payload and return frequencies in cm^-1.

    The parser is permissive: it ignores comments starting with '#', optional
    leading mode indices ("1  81.0"), and accepts comma/space separated tokens.
    Only the final numeric token in each line is used. Imaginary modes may be
    specified via a leading "i" token.
    """

    freqs: list[float] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = [tok for tok in re.split(r"[\s,]+", line) if tok]
        if not tokens:
            continue
        head = tokens[0].lower()
        if head in {"i", "imag", "imaginary"} and len(tokens) >= 2:
            try:
                val = float(tokens[1].rstrip(":"))
            except ValueError:
                continue
            freqs.append(float(abs(val)))
            continue

        value: float | None = None
        for tok in reversed(tokens):
            cleaned = tok.rstrip(":,;")
            try:
                value = float(cleaned)
                break
            except ValueError:
                continue
        if value is None:
            continue
        freqs.append(float(abs(value)))
    return freqs


# ------------------------------------------------------------------------------
# Thermodynamics helpers
# ------------------------------------------------------------------------------
def _mode_type(index: int, ntr_total: int) -> str:
    if index < 3:
        return "quasi-trans"
    if index < ntr_total:
        return "quasi-rot"
    return "vib"


def _entropy_one(v_cm1: float, temperature: float) -> float:
    x = _PLANCK_H * _SPEED_OF_LIGHT_CM / (_BOLTZMANN_K * temperature)
    t = x * v_cm1
    if t < 1e-12:
        return 0.0
    e = math.exp(-t)
    return _GAS_CONSTANT_R * (t * e / (1.0 - e) - math.log(1.0 - e))


def _energy_one(v_cm1: float, temperature: float) -> float:
    x = _PLANCK_H * _SPEED_OF_LIGHT_CM / (_BOLTZMANN_K * temperature)
    t = x * v_cm1
    if t < 1e-12:
        return 0.0
    e = math.exp(-t)
    return _GAS_CONSTANT_R * temperature * (t * (0.5 + e / (1.0 - e)))


def _calc_single_temperature(freq_trans: np.ndarray, freq_rot: np.ndarray, freq_vib: np.ndarray, temperature: float) -> dict:
    freq_trans = np.asarray(freq_trans, dtype=float)
    freq_rot = np.asarray(freq_rot, dtype=float)
    freq_vib = np.asarray(freq_vib, dtype=float)

    S_trans = float(sum(_entropy_one(v, temperature) for v in freq_trans))
    S_rot = float(sum(_entropy_one(v, temperature) for v in freq_rot))
    S_vib = float(sum(_entropy_one(v, temperature) for v in freq_vib))
    S_total = S_trans + S_rot + S_vib

    E_trans = 0.5 * float(sum(_energy_one(v, temperature) for v in freq_trans))
    E_rot = 0.5 * float(sum(_energy_one(v, temperature) for v in freq_rot))
    E_vib = float(sum(_energy_one(v, temperature) for v in freq_vib))
    E_total = E_trans + E_rot + E_vib

    return {
        "T": float(temperature),
        "freq": {
            "trans": freq_trans,
            "rot": freq_rot,
            "vib": freq_vib,
            "all": np.concatenate([freq_trans, freq_rot, freq_vib]) if freq_vib.size else np.concatenate([freq_trans, freq_rot]),
        },
        "S": {"trans": S_trans, "rot": S_rot, "vib": S_vib, "total": S_total},
        "E": {"trans": E_trans, "rot": E_rot, "vib": E_vib, "total": E_total},
    }


def _thermo_table_from_record(record: dict) -> dict:
    temperature = float(record["T"])
    E = record["E"]
    S = record["S"]
    return {
        "T": temperature,
        "E_kJmol": {
            "trans": E["trans"] * _KJ_PER_J,
            "rot": E["rot"] * _KJ_PER_J,
            "vib": E["vib"] * _KJ_PER_J,
            "total": E["total"] * _KJ_PER_J,
        },
        "E_kcalmol": {
            "trans": E["trans"] * _KCAL_PER_J,
            "rot": E["rot"] * _KCAL_PER_J,
            "vib": E["vib"] * _KCAL_PER_J,
            "total": E["total"] * _KCAL_PER_J,
        },
        "S_kJmolK": {
            "trans": S["trans"] * _KJ_PER_J,
            "rot": S["rot"] * _KJ_PER_J,
            "vib": S["vib"] * _KJ_PER_J,
            "total": S["total"] * _KJ_PER_J,
        },
    }


def _format_mode_table(freq_all: Sequence[float], ntr_total: int) -> str:
    if not freq_all:
        return ""
    lines = ["Final Result", "", f"{'Mode':>4}  {'type':<11}  {'vibrational frequency (cm^-1)':>29}", "-" * 52]
    for idx, value in enumerate(freq_all, start=1):
        lines.append(f"{idx:4d}  {_mode_type(idx - 1, ntr_total):<11}  {float(value):>27.4f}")
    return "\n".join(lines)


def _format_thermo_text(records: list[dict]) -> str:
    if not records:
        return ""
    blocks: list[str] = []
    for rec in records:
        T = float(rec["T"])
        E = rec["E"]
        S = rec["S"]
        block = []
        block.append(f"Thermochemistry  [T = {T:.2f} K]")
        block.append(f"{'Quantity':<12}  {'Trans':>10}  {'Rot':>10}  {'Vib':>10}  {'Total':>10}")
        block.append(f"E (kJ/mol)    {E['trans'] * _KJ_PER_J:10.5f}  {E['rot'] * _KJ_PER_J:10.5f}  {E['vib'] * _KJ_PER_J:10.5f}  {E['total'] * _KJ_PER_J:10.5f}")
        block.append(f"S (kJ/mol/K)  {S['trans'] * _KJ_PER_J:10.5f}  {S['rot'] * _KJ_PER_J:10.5f}  {S['vib'] * _KJ_PER_J:10.5f}  {S['total'] * _KJ_PER_J:10.5f}")
        block.append("")
        block.append("(kJ/mol -> kcal/mol)")
        block.append(f"E (kcal/mol)  {E['trans'] * _KCAL_PER_J:10.3f}  {E['rot'] * _KCAL_PER_J:10.3f}  {E['vib'] * _KCAL_PER_J:10.3f}  {E['total'] * _KCAL_PER_J:10.3f}")
        block.append(f"S (kcal/mol/K){S['trans'] * _KCAL_PER_J:10.3f}  {S['rot'] * _KCAL_PER_J:10.3f}  {S['vib'] * _KCAL_PER_J:10.3f}  {S['total'] * _KCAL_PER_J:10.3f}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------
def analyze_frequencies(freqs: Sequence[float], temps: Iterable[float], ntr_total: int) -> FreqAnalysisResult:
    freq_array = np.asarray([float(abs(v)) for v in freqs], dtype=float)
    if freq_array.size < ntr_total:
        raise ValueError(
            f"Frequency data requires at least {ntr_total} translational/rotational modes; "
            f"only {freq_array.size} values were provided."
        )
    if freq_array[:3].size < 3:
        raise ValueError("Frequency data requires at least three translational modes.")

    freq_trans = freq_array[:3]
    freq_rot = freq_array[3:ntr_total]
    freq_vib = freq_array[ntr_total:]

    temperatures = [float(t) for t in temps]
    per_temp_records: list[dict] = []
    thermo_tables: list[dict] = []
    for T in temperatures:
        record = _calc_single_temperature(freq_trans, freq_rot, freq_vib, T)
        per_temp_records.append(record)
        thermo_tables.append(_thermo_table_from_record(record))

    mode_table_text = _format_mode_table(freq_array.tolist(), ntr_total)
    thermo_text = _format_thermo_text(per_temp_records)

    return FreqAnalysisResult(
        freq_all=freq_array,
        freq_trans=freq_trans,
        freq_rot=freq_rot,
        freq_vib=freq_vib,
        ntr_total=int(ntr_total),
        thermo_tables=thermo_tables,
        per_temperature=per_temp_records,
        mode_table_text=mode_table_text,
        thermo_text=thermo_text,
    )


def analyze_plaintext_freqout(raw_text: str, temps: Iterable[float], ntr_total: int) -> FreqAnalysisResult:
    freqs = parse_plaintext_freqout(raw_text)
    if not freqs:
        raise ValueError("freq.out did not contain any readable frequencies")
    return analyze_frequencies(freqs, temps, ntr_total)
