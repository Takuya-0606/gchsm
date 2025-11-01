import inspect
import numpy as np
import scipy
from typing import Sequence, Optional, Union, Tuple
from pyscf import gto, df, scf, mp, solvent
from pyscf.dft import gen_grid  # <- add import for Lebedev order
from pyscf.solvent import pcm as _pcm
from pyscf.solvent.pcm import (
    gen_surface, get_F_A_gcm, get_D_S_gcm,
    XI, modified_Bondi
)  # type: ignore
from pyscf.lib import logger
from pyscf.tools.finite_diff import Hessian
from scipy.special import erf

BOHR = 0.529177210903  # Å -> Bohr
PI = np.pi


# ---------------- utility ----------------------------------------------------
def _filter_kwargs(fn, kwargs):
    """Return dict with keys accepted by *fn* signature (robust across API)."""
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class fixcav(_pcm.PCM):
    """PCM variant with *user-defined* cavity centers/radii, plus helpers.

    - Keeps the standard PCM interface but overrides `build()` to construct
      the surface from user-provided centers/radii (not molecular atoms).
    - Adds convenience methods to attach the PCM to an SCF object and to run
      MP2 frequencies via finite-difference of analytic MP2 gradients.
    - Designed to work across PySCF ≥2.0..2.6 without API tweaks.
    """
    def __init__(
        self,
        mol: gto.Mole,
        centers: Sequence[Sequence[float]],
        radii: Optional[Union[Sequence[float], float]] = None,
        *,
        unit: str = "Angstrom",
        **kw,
    ) -> None:
        # ----- store user geometry (internal Bohr) --------------------
        self._centers = np.asarray(centers, float)
        self._radii_user = None  # type: Optional[np.ndarray]
        if unit.lower().startswith("a"):
            self._centers /= BOHR
            if radii is not None:
                self._radii_user = np.asarray(radii, float) / BOHR
        else:
            if radii is not None:
                self._radii_user = np.asarray(radii, float)

        # ----- pull aliases safe from API differences -----------------
        _solver = kw.pop("solver", kw.pop("method", None))
        _eps    = kw.pop("eps", None)

        # ----- parent ctor with filtered kwargs -----------------------
        super().__init__(mol, **_filter_kwargs(_pcm.PCM.__init__, kw))

        # ----- set attributes post‑hoc (if field exists) -------------
        if _solver is not None:
            if hasattr(self, "solver"):
                self.solver = _solver
            else:
                self.method = _solver  # very old builds
        if _eps is not None:
            self.eps = _eps

    # ----------------------------------------------------------------
    # surface generation ---------------------------------------------
    # ----------------------------------------------------------------

    def build(self, ng: Optional[int] = None):
        '''
        Generate PCM surface using user centers/radii (kept across resets).
        '''
        # base radii table once
        if self.radii_table is None:
            self.radii_table = self.vdw_scale * modified_Bondi + self.r_probe

        if len(self._centers) != self.mol.natm:
            raise ValueError("Number of centers must equal number of atoms.")
        dummy = gto.Mole()
        dummy.unit = "Bohr"
        dummy.atom = [(self.mol.atom_symbol(i), coord.tolist())
                      for i, coord in enumerate(self._centers)]
        dummy.basis = {}          # no AO functions – avoids integral cost
        dummy.charge = 0
        dummy.build()

        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        # surface generation --------------------------------------------------
        self.surface = gen_surface(dummy, rad=self.radii_table, ng=ng)

        # ---- remainder identical to parent PCM.build() (2010 & 2024 GCM) ----
        self._intermediates = {}
        F, A = get_F_A_gcm(self.surface)
        D, S, O = get_D_S_gcm(self.surface, with_S=True, with_D=True)

        eps = self.eps
        if self.method.upper() in ["C-PCM", "CPCM"]:
            f_eps = (eps - 1.0) / eps if eps != float("inf") else 1.0
            K = S
            R = -f_eps * np.eye(K.shape[0])
        elif self.method.upper() == "COSMO":
            f_eps = (eps - 1.0) / (eps + 0.5) if eps != float("inf") else 1.0
            K = S
            R = -f_eps * np.eye(K.shape[0])
        elif self.method.upper() in ["IEF-PCM", "IEFPCM"]:
            f_eps = (eps - 1.0) / (eps + 1.0) if eps != float("inf") else 1.0
            DA  = D * A
            K = S - f_eps / (2.0 * PI) * (DA @ S)
            R = -f_eps * (np.eye(K.shape[0]) - (DA) / (2.0 * PI))
        elif self.method.upper() == "SS(V)PE":
            f_eps = (eps - 1.0) / (eps + 1.0) if eps != float("inf") else 1.0
            DA  = D * A
            DAS = DA @ S
            K = S - f_eps / (4.0 * PI) * (DAS + DAS.T)
            R = -f_eps * (np.eye(K.shape[0]) - DA / (2.0 * PI))
        else:
            raise RuntimeError(f"Unknown PCM method: {self.method}")

        self._intermediates.update({"S": S, "D": D, "A": A, "K": K, "R": R, "f_epsilon": f_eps})

        # nuclear potential on surface ---------------------------------------
        zeta = self.surface["charge_exp"]
        grid_coords = self.surface["grid_coords"]
        nuc_coords = self.mol.atom_coords(unit="B")
        nuc_chg = self.mol.atom_charges()

        v_ng = np.zeros(len(grid_coords))
        for i, ri in enumerate(grid_coords):
            for RA, ZA in zip(nuc_coords, nuc_chg):
                r = np.linalg.norm(ri - RA)
                if r < 1e-12:
                    continue
                v_ng[i] += ZA * erf(zeta[i] * r) / r
        self.v_grids_n = v_ng
        return self

    # keep surface when resetting (for frozen‑cavity workflows) --------------
    def reset(self, mol: Optional[gto.Mole] = None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        self.intopt = None
        return self

    #2025/08/27
# ---------------------------------------------------------------------
    # helpers integrated from the standalone functions
    # ---------------------------------------------------------------------

    def attach_to(self, mf, *, build_surface: bool = True, lebedev_order: Optional[int] = None,
                  conv_tol: float = 1e-10, conv_tol_grad: float = 1e-6):
        """Attach this PCM object to an existing SCF object and return it.

        Parameters
        ----------
        mf : SCF object (RHF/UHF/RKS/UKS)
        build_surface : bool
            If True, call `self.build()` before attaching.
        lebedev_order : int or None
            Optional override of `self.lebedev_order` prior to building.
        conv_tol / conv_tol_grad : float
            Tighter SCF tolerances recommended for finite differences.
        """
        if lebedev_order is not None:
            self.lebedev_order = lebedev_order
        if build_surface:
            self.build()

        mf_pcm = solvent.PCM(mf)
        mf_pcm.with_solvent = self
        mf_pcm.conv_tol = conv_tol
        mf_pcm.conv_tol_grad = conv_tol_grad
        return mf_pcm

    @classmethod
    def build_scf(cls,
                  mol: gto.Mole,
                  *,
                  centers: Sequence[Sequence[float]],
                  radii: Optional[Union[Sequence[float], float]] = None,
                  unit: str = "Angstrom",
                  method: str = "IEF-PCM",
                  eps: float = 78.3553,
                  lebedev_order: int = 17,
                  scf_method: str = "RHF",
                  jk_df: bool = False,
                  **pcm_kw) -> Tuple[scf.hf.SCF, "fixcav"]:
        """Construct SCF wrapped with fixcav in one call.

        Returns
        -------
        (mf_pcm, pcm) : (SCF, fixcav)
            The SCF object already wrapped with this PCM, and the PCM itself.
        """
        # SCF base
        scf_method = scf_method.upper()
        if scf_method == "RHF":
            mf = scf.RHF(mol)
        elif scf_method == "UHF":
            mf = scf.UHF(mol)
        else:
            raise ValueError("scf_method must be RHF or UHF for MP2 base.")

        if jk_df:
            mf = mf.density_fit()  # RI-JK

        pcm = cls(mol, centers=centers, radii=radii, unit=unit,
                  method=method, eps=eps, **pcm_kw)
        pcm.lebedev_order = lebedev_order
        mf_pcm = pcm.attach_to(mf, build_surface=True)

        return mf_pcm, pcm

    def mp2_frequency(self,
                      mol_or_mf: Union[gto.Mole, scf.hf.SCF],
                      *,
                      mp2_df: bool = True,
                      disp_bohr: float = 5e-3,
                      scf_method: str = "RHF",
                      jk_df: bool = False):
        """Run MP2(PCM) harmonic frequencies via finite-difference of gradients.

        Parameters
        ----------
        mol_or_mf : Mole or SCF
            If a Mole is given, an SCF will be built and wrapped automatically.
            If an SCF is given, it will be used (and attached if not yet).
        mp2_df : bool
            Use DF/RI for MP2 to save time/memory.
        disp_bohr : float
            Central-difference displacement in Bohr (3e-3..5e-3 typical).
        scf_method : str
            Only used when `mol_or_mf` is a Mole; choose "RHF" or "UHF".
        jk_df : bool
            Use RI-JK for the SCF (only when building from Mole).

        Returns
        -------
        (res, H) : (dict, np.ndarray)
            `res` is the dictionary from `thermo.harmonic_analysis` including
            "freq_wavenumber", "zero_point_energy", etc.
            `H` is the (3N,3N) mass-unweighted Hessian in atomic units.
        """
        # Prepare SCF with this PCM attached
        if isinstance(mol_or_mf, gto.Mole):
            mf_pcm, _ = self.build_scf(mol_or_mf, centers=[], radii=None)  # dummy; will replace
            # Replace the PCM built above with *this* instance to keep surface
            # However, we need an SCF first:
            scf_base = scf.RHF(mol_or_mf) if scf_method.upper() == "RHF" else scf.UHF(mol_or_mf)
            if jk_df:
                scf_base = scf_base.density_fit()
            mf_pcm = self.attach_to(scf_base, build_surface=(self.surface is None))
        else:
            mf_pcm = mol_or_mf
            # attach if not already wrapped
            if not isinstance(getattr(mf_pcm, "with_solvent", None), fixcav):
                mf_pcm = self.attach_to(mf_pcm, build_surface=(self.surface is None))

        # SCF
        mf_pcm.kernel()

        # MP2
        mp2obj = mp.MP2(mf_pcm)
        if mp2_df:
            mp2obj = mp2obj.density_fit()
        mp2obj.kernel()

        # Finite-difference Hessian of analytic MP2 gradients
        fd = Hessian(mp2obj.Gradients())
        fd.displacement = disp_bohr
        H = fd.kernel()

#        res = thermo.harmonic_analysis(mf_pcm.mol, H)
        return H
