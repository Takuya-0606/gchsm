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

"""Surface extraction helpers shared by non-electrostatic routines."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

BOHR_TO_ANG = 0.529177210903


def _pick(obj, *candidates):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in candidates:
            if key in obj:
                return obj[key]
        return None
    for key in candidates:
        if hasattr(obj, key):
            return getattr(obj, key)
    return None


def _as_float_array(x):
    return None if x is None else np.asarray(x, dtype=float)


def _from_grid_matrix(grid):
    G = np.asarray(grid, dtype=float)
    if G.ndim != 2 or G.shape[1] < 6:
        raise ValueError(f"Unsupported grid matrix shape: {G.shape}")
    coords = G[:, :3]
    normals = G[:, 3:6]
    areas = G[:, 6] if G.shape[1] >= 7 else None
    return coords, normals, areas


def _reconstruct_normals_from_atoms(centers_A, atom_coords_A):
    C = np.asarray(centers_A, float)
    A = np.asarray(atom_coords_A, float)
    idx = np.argmin(((C[:, None, :] - A[None, :, :]) ** 2).sum(axis=2), axis=1)
    vec = C - A[idx]
    nrm = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    return vec / nrm


def extract_pcm_surface(mf) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
    ws = getattr(mf, "with_solvent", None)
    surface = getattr(ws, "surface", None)
    if surface is None:
        raise RuntimeError("No PCM surface attached to mean-field object.")

    coords = normals = areas = types = None

    for meth in ("as_arrays", "to_arrays", "tesserae", "get_tesserae"):
        if hasattr(surface, meth):
            try:
                out = getattr(surface, meth)()
            except Exception:
                continue
            if isinstance(out, (list, tuple)) and len(out) >= 3:
                c0 = _as_float_array(out[0])
                n0 = _as_float_array(out[1])
                a0 = _as_float_array(out[2])
                t0 = None
                if len(out) >= 4:
                    try:
                        t0 = np.asarray(out[3], dtype=int)
                    except Exception:
                        t0 = None
                if c0 is not None and n0 is not None and a0 is not None:
                    coords, normals, areas, types = c0, n0, a0, t0
                    break

    if coords is None:
        grid = _pick(surface, "grid", "tesserae", "points_matrix", "mesh")
        if grid is not None:
            try:
                coords, normals, areas = _from_grid_matrix(grid)
            except Exception:
                pass

    if coords is None:
        coords = _as_float_array(
            _pick(surface, "grid_coords", "coords", "points", "r", "xyz", "coord", "p")
        )
    if normals is None:
        normals = _as_float_array(
            _pick(surface, "norm_vec", "normals", "nvec", "nv", "normal", "norm", "n")
        )
    if areas is None:
        areas = _as_float_array(_pick(surface, "area", "areas", "weights", "w", "weight"))

    raw_types = _pick(surface, "types", "itype", "labels", "atom_ids", "atype")
    if raw_types is not None:
        try:
            types = np.asarray(raw_types, dtype=int)
        except Exception:
            types = None
    else:
        types = None

    if coords is None or areas is None:
        keys = (
            list(surface.keys())
            if isinstance(surface, dict)
            else [k for k in dir(surface) if not k.startswith("_")]
        )
        raise KeyError(
            "Could not locate tessera arrays on PCM surface.\n"
            "Tried: grid/grid_coords/coords(points/r/xyz), area/weights, normals/norm_vec.\n"
            f"Available keys/attrs: {keys}"
        )

    centers_A = np.asarray(coords, float)
    areas_A2 = np.asarray(areas, float).reshape(-1)

    if normals is None:
        atom_A = mf.mol.atom_coords()
        normals_unit = _reconstruct_normals_from_atoms(centers_A, atom_A)
    else:
        normals_unit = np.asarray(normals, float)
        nrm = np.maximum(np.linalg.norm(normals_unit, axis=1, keepdims=True), 1e-12)
        normals_unit = normals_unit / nrm

    rdotn = np.einsum("ij,ij->i", centers_A, normals_unit)
    V_est = float(np.sum(rdotn * areas_A2) / 3.0)
    outward = V_est >= 0.0
    if not outward:
        normals_unit = -normals_unit

    return centers_A, normals_unit, areas_A2, types, outward


def surface_area_from_tessera(areas: np.ndarray) -> float:
    return float(np.sum(areas))


def volume_from_closed_surface(
    centers: np.ndarray, normals: np.ndarray, areas: np.ndarray, outward: bool
) -> float:
    sign = 1.0 if outward else -1.0
    rdotn = sign * np.einsum("ij,ij->i", centers, normals)
    return float(np.sum(rdotn * areas) / 3.0)


def _fibonacci_sphere(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    from math import cos, pi, sin, sqrt

    pts = np.zeros((n_points, 3), float)
    w = np.full(n_points, 4.0 * np.pi / n_points, float)
    offset = 2.0 / n_points
    inc = pi * (3.0 - sqrt(5.0))
    for i in range(n_points):
        y = ((i * offset) - 1.0) + (offset / 2.0)
        r = (1.0 - y * y) ** 0.5
        phi = (i % n_points) * inc
        x = cos(phi) * r
        z = sin(phi) * r
        pts[i] = (x, y, z)
    return pts, w


def build_ses_surface_from_geom(
    mol,
    radii_map: dict[str, float],
    probe_radius: float = 1.40,
    npoints_per_sphere: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
    natm = mol.natm
    sym = [mol.atom_symbol(i) for i in range(natm)]
    Rvdw = np.array([float(radii_map.get(s, 1.70)) for s in sym], float)
    Ri = Rvdw + float(probe_radius)

    xyz_bohr = mol.atom_coords()
    xyz = np.array(xyz_bohr, float) * BOHR_TO_ANG

    sphere_pts, weights = _fibonacci_sphere(int(npoints_per_sphere))

    centers_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    areas_list: list[np.ndarray] = []

    for i in range(natm):
        Ri_i = Ri[i]
        Ri2 = Ri_i * Ri_i
        candidates = xyz[i] + Ri_i * sphere_pts
        exposed = np.ones(candidates.shape[0], dtype=bool)
        for j in range(natm):
            if j == i:
                continue
            d2 = np.sum((candidates - xyz[j]) ** 2, axis=1)
            exposed &= d2 > (Ri[j] * Ri[j]) - 1e-12
        if np.any(exposed):
            centers_list.append(candidates[exposed])
            normals_list.append(sphere_pts[exposed])
            areas_list.append(weights[exposed] * Ri2)

    if centers_list:
        centers = np.vstack(centers_list)
        normals = np.vstack(normals_list)
        areas = np.concatenate(areas_list)
    else:
        centers = np.zeros((0, 3))
        normals = np.zeros((0, 3))
        areas = np.zeros((0,))

    outward = True
    types = None
    return centers, normals, areas, types, outward
