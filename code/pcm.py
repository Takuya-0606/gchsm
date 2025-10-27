#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Xiaojie Wu <wxj6000@gmail.com>
#

'''
PCM family solvent models
'''

import numpy
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, df
from pyscf.dft import gen_grid
from pyscf.data import radii
from pyscf.solvent import ddcosmo
from pyscf.solvent import _attach_solvent
from scipy.special import erf

@lib.with_doc(_attach_solvent._for_scf.__doc__)
def pcm_for_scf(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = PCM(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casscf.__doc__)
def pcm_for_casscf(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casscf(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casci.__doc__)
def pcm_for_casci(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casci(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_post_scf.__doc__)
def pcm_for_post_scf(method, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(method._scf, 'with_solvent', None), PCM):
            solvent_obj = method._scf.with_solvent
        else:
            solvent_obj = PCM(method.mol)
    return _attach_solvent._for_post_scf(method, solvent_obj, dm)

pcm_for_tdscf = _attach_solvent._for_tdscf


# Inject PCM to other methods
from pyscf import scf
from pyscf import mcscf
from pyscf import mp, ci, cc
from pyscf import tdscf
scf.hf.SCF.PCM    = scf.hf.SCF.PCM    = pcm_for_scf
mp.mp2.MP2.PCM    = mp.mp2.MP2.PCM    = pcm_for_post_scf
ci.cisd.CISD.PCM  = ci.cisd.CISD.PCM  = pcm_for_post_scf
cc.ccsd.CCSD.PCM  = cc.ccsd.CCSD.PCM  = pcm_for_post_scf
tdscf.rhf.TDBase.PCM = tdscf.rhf.TDBase.PCM = pcm_for_tdscf
mcscf.casci.CASCI.PCM = mcscf.casci.CASCI.PCM = pcm_for_casci
mcscf.mc1step.CASSCF.PCM = mcscf.mc1step.CASSCF.PCM = pcm_for_casscf

# TABLE II,  J. Chem. Phys. 122, 194110 (2005)
XI = {
    6: 4.84566077868,
    14: 4.86458714334,
    26: 4.85478226219,
    38: 4.90105812685,
    50: 4.89250673295,
    86: 4.89741372580,
    110: 4.90101060987,
    146: 4.89825187392,
    170: 4.90685517725,
    194: 4.90337644248,
    302: 4.90498088169,
    350: 4.86879474832,
    434: 4.90567349080,
    590: 4.90624071359,
    770: 4.90656435779,
    974: 4.90685167998,
    1202: 4.90704098216,
    1454: 4.90721023869,
    1730: 4.90733270691,
    2030: 4.90744499142,
    2354: 4.90753082825,
    2702: 4.90760972766,
    3074: 4.90767282394,
    3470: 4.90773141371,
    3890: 4.90777965981,
    4334: 4.90782469526,
    4802: 4.90749125553,
    5294: 4.90762073452,
    5810: 4.90792902522,
}

modified_Bondi = radii.VDW.copy()
modified_Bondi[1] = 1.1/radii.BOHR      # modified version
PI = numpy.pi

def switch_h(x):
    '''
    switching function (eq. 3.19)
    J. Chem. Phys. 133, 244111 (2010)
    notice the typo in the paper
    '''
    y = x**3 * (10.0 - 15.0*x + 6.0*x**2)
    y[x<0] = 0.0
    y[x>1] = 1.0
    return y

def gen_surface(mol, ng=302, rad=modified_Bondi, vdw_scale=1.2):
    '''J. Phys. Chem. A 1999, 103, 11060-11079'''
    unit_sphere = gen_grid.MakeAngularGrid(ng)
    atom_coords = mol.atom_coords(unit='B')
    charges = mol.atom_charges()
    N_J = ng * numpy.ones(mol.natm)
    R_J = numpy.asarray([rad[chg] for chg in charges])
    R_sw_J = R_J * (14.0 / N_J)**0.5
    alpha_J = 1.0/2.0 + R_J/R_sw_J - ((R_J/R_sw_J)**2 - 1.0/28)**0.5
    R_in_J = R_J - alpha_J * R_sw_J

    grid_coords = []
    weights = []
    charge_exp = []
    switch_fun = []
    R_vdw = []
    norm_vec = []
    area = []
    gslice_by_atom = []
    p0 = p1 = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        chg = gto.charge(symb)
        r_vdw = rad[chg]

        atom_grid = r_vdw * unit_sphere[:,:3] + atom_coords[ia,:]
        riJ = scipy.spatial.distance.cdist(atom_grid[:,:3], atom_coords)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[:,ia] = 1.0
        diJ[diJ < 1e-8] = 0.0
        fiJ = switch_h(diJ)

        w = unit_sphere[:,3] * 4.0 * PI
        swf = numpy.prod(fiJ, axis=1)
        idx = w*swf > 1e-16

        p0, p1 = p1, p1+sum(idx)
        gslice_by_atom.append([p0,p1])
        grid_coords.append(atom_grid[idx,:3])
        weights.append(w[idx])
        switch_fun.append(swf[idx])
        norm_vec.append(unit_sphere[idx,:3])
        xi = XI[ng] / (r_vdw * w[idx]**0.5)
        charge_exp.append(xi)
        R_vdw.append(numpy.ones(sum(idx)) * r_vdw)
        area.append(w[idx]*r_vdw**2*swf[idx])

    grid_coords = numpy.vstack(grid_coords)
    norm_vec = numpy.vstack(norm_vec)
    weights = numpy.concatenate(weights)
    charge_exp = numpy.concatenate(charge_exp)
    switch_fun = numpy.concatenate(switch_fun)
    area = numpy.concatenate(area)
    R_vdw = numpy.concatenate(R_vdw)

    surface = {
        'ng': ng,
        'gslice_by_atom': gslice_by_atom,
        'grid_coords': grid_coords,
        'weights': weights,
        'charge_exp': charge_exp,
        'switch_fun': switch_fun,
        'R_vdw': R_vdw,
        'norm_vec': norm_vec,
        'area': area,
        'R_in_J': R_in_J,
        'R_sw_J': R_sw_J,
        'atom_coords': atom_coords
    }
    return surface

def get_F_A(surface):
    '''
    generate F and A matrix in  J. Chem. Phys. 133, 244111 (2010)
    '''
    R_vdw = surface['R_vdw']
    switch_fun = surface['switch_fun']
    weights = surface['weights']
    A = weights*R_vdw**2*switch_fun
    return switch_fun, A

# Implemented by Takuya HASHIMOTO
def get_F_A_gcm(surface):
    '''
    generate F and A matrix in J. Chem. Phys., 20, 3169-3183 (2024)
    eq.(12) & (13)
    '''
    import scipy.spatial.distance as distance
    charge_exp  = surface['charge_exp'] # zeta_m
    switch_fun  = surface['switch_fun'] # F_m
    grid_coords = surface['grid_coords']
    ngrids      = len(charge_exp)

    # diagonal element: A_mm = sqrt(2/pi)*zeta_m / switch_fun
    A      = numpy.zeros((ngrids, ngrids))
    A_diag = numpy.sqrt(2/numpy.pi)*charge_exp / switch_fun
    numpy.fill_diagonal(A, A_diag)

    # off-diagonal element: A_mn = erf(zeta_mn*r_mn) / r_mn
    rij = distance.cdist(grid_coords, grid_coords)
    xi_i, xi_j = numpy.meshgrid(charge_exp, charge_exp, indexing='ij')
    xi_mn = xi_i * xi_j / numpy.sqrt(xi_i**2 + xi_j**2)

    with numpy.errstate(divide='ignore', invalid='ignore'):
        A_off = erf(xi_mn*rij) / rij
        numpy.fill_diagonal(A_off, 0.0)
        A    += A_off

    return switch_fun, A

def get_D_S(surface, with_S=True, with_D=False):
    '''
    generate D and S matrix in  J. Chem. Phys. 133, 244111 (2010)
    The diagonal entries of S is not filled
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    norm_vec    = surface['norm_vec']
    R_vdw       = surface['R_vdw']

    xi_i, xi_j = numpy.meshgrid(charge_exp, charge_exp, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    rij = scipy.spatial.distance.cdist(grid_coords, grid_coords)
    xi_r_ij = xi_ij * rij
    numpy.fill_diagonal(rij, 1)
    S = scipy.special.erf(xi_r_ij) / rij
    numpy.fill_diagonal(S, charge_exp * (2.0 / PI)**0.5 / switch_fun)

    D = None
    if with_D:
        drij = numpy.expand_dims(grid_coords, axis=1) - grid_coords
        nrij = numpy.sum(drij * norm_vec, axis=-1)

        D = S*nrij/rij**2 -2.0*xi_r_ij/PI**0.5*numpy.exp(-xi_r_ij**2)*nrij/rij**3
        numpy.fill_diagonal(D, -charge_exp * (2.0 / PI)**0.5 / (2.0 * R_vdw))

    return D, S

# Implemented by Takuya HASHIMOTO
def get_D_S_gcm(surface, with_S=True, with_D=False):
    '''
    generate O, D and S matrix in J. Chem. Phys., 20, 3169-3183 (2024)
    eq.(29)
    '''
    import scipy.spatial.distance as distance
    charge_exp = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun = surface['switch_fun']
    norm_vec = surface['norm_vec']
    R_vdw = surface['R_vdw']

    ngrids = len(charge_exp)
    xi_i, xi_j = numpy.meshgrid(charge_exp, charge_exp, indexing='ij')
    xi_ij = xi_i * xi_j / numpy.sqrt(xi_i**2 + xi_j**2)

    rij_vec = grid_coords[:, None, :] - grid_coords[None, :, :]
    rij = numpy.linalg.norm(rij_vec, axis=-1)
    numpy.fill_diagonal(rij, 1.0)  # prevent division by zero

    xi_r_ij = xi_ij * rij

    # S matrix in eq.(13)
    S = None
    if with_S:
        S = erf(xi_r_ij) / rij
        numpy.fill_diagonal(S, charge_exp * numpy.sqrt(2.0 / numpy.pi) / switch_fun)

    # D matrix (if you need...)
    D = None
    if with_D:
        nrij = numpy.einsum('ijk,ik->ij', rij_vec, norm_vec)
        exp_term = numpy.exp(-xi_r_ij**2)
        D = S * nrij / rij**2 - 2.0 * xi_r_ij / numpy.sqrt(numpy.pi) * exp_term * nrij / rij**3
        numpy.fill_diagonal(D, -charge_exp * numpy.sqrt(2.0 / numpy.pi) / (2.0 * R_vdw))

    # O matrix in eq.(28) & (29)
    O = numpy.zeros((ngrids, ngrids))

    # off-diagonal element
    with numpy.errstate(divide='ignore', invalid='ignore'):
        term1 = 2 * xi_ij / (numpy.sqrt(numpy.pi) * rij) * numpy.exp(-xi_ij**2 * rij**2)
        term2 = erf(xi_ij * rij) / (rij**2)
        O_offdiag = term1 - term2
        numpy.fill_diagonal(O_offdiag, 0.0)
        O += O_offdiag

    # diagonal element
    A_diag = numpy.sqrt(2/numpy.pi)*charge_exp / switch_fun
    O_diag = -A_diag / switch_fun
    numpy.fill_diagonal(O, O_diag)

    return D, S, O

class PCM(lib.StreamObject):
    '''
    PCM Solvent Model

    This class implements the Polarizable Continuum Model (PCM) for solvent effects.

    Input Attributes:
    -----------------
    method : str
        The PCM model. Options include 'C-PCM', 'IEF-PCM', 'COSMO', and 'SS(V)PE'.
        Default is 'C-PCM'.

    vdw_scale : float
        A scaling factor for van der Waals radii. Default is 1.2, consistent with Q-Chem settings.

    r_probe : float
        An additional radius (in Angstrom) added to the van der Waals radii.
        Default is 0.0.

    radii_table : dict
        Custom van der Waals radii for each element. By default, scaled van der Waals radii
        from `vdw_scale` and `r_probe` are used.

    lebedev_order : int
        The order of the Lebedev mesh used for the cavity sphere. Default is 29 (302 grids).

    eps : float
        The dielectric constant of the solvent. Default is 78.3553, the dielectric constant
        for water.

    frozen : bool
        Whether to freeze the potential produced by the solvent during SCF iterations or
        other convergence processes. When frozen=True is set, the solvent is
        assumed to respond slowly, while the electron density relaxes quickly.
        Default is False.

    max_cycle : int
        The maximum number of iterations to relax the solvent.

    conv_tol : float
        The convergence tolerance for total energy during solvent relaxation.

    equilibrium_solvation : bool
        Affects TDDFT and other excited state computations. Controls whether the solvent
        relaxes rapidly with respect to the electron density of the excited state.
        For vertical excitations, it is recommended to set this to False, as the solvent
        typically does not fully relax. In some software packages (e.g., Q-Chem),
        non-equilibrium solvation is applied with an optical dielectric constant of
        eps=1.78. Default is False.

    state_id : int
        Specifies the target state in excited state calculations.
        `state_id=0` corresponds to the ground state, while `state_id=1` corresponds
        to the first excited state. Default is 0.

    Saved Results:
    --------------
    e_tot : float
        The energy contribution from the solvent.

    v : ndarray
        The potential matrix generated by the solvent.

    Intermediate Attributes:
    ------------------------
    These attributes are generated during calculations and should not be modified.
    Additionally, they may not be compatible between GPU and CPU implementations.

    - surface
    - _intermediates
    - v_grids_n
    '''

    _keys = {
        'method', 'vdw_scale', 'surface', 'r_probe',
        'mol', 'radii_table', 'lebedev_order',
        'eps', 'max_cycle', 'conv_tol', 'state_id', 'frozen',
        'equilibrium_solvation', 'e', 'v', 'v_grids_n',
    }

    kernel = ddcosmo.DDCOSMO.kernel

    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.method = 'C-PCM'

        self.vdw_scale = 1.2 # default value in qchem
        self.r_probe = 0.0
        self.radii_table = None
        self.lebedev_order = 17
        self.eps = 78.3553

        self.max_cycle = 20
        self.conv_tol = 1e-7
        self.state_id = 0

        self.frozen = False
        self.equilibrium_solvation = False

        self.surface = {}
        self._intermediates = {}
        self.v_grids_n = None # nuclear potential on grids

        self.e = None
        self.v = None
        self._dm = None

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s (In testing) ********', self.__class__)
        logger.warn(self, 'PCM is an experimental feature. It is '
                    'still in testing.\nFeatures and APIs may be changed '
                    'in the future.')
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        logger.info(self, 'eps = %s'          , self.eps)
        logger.info(self, 'frozen = %s'       , self.frozen)
        #logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        return self

    def to_gpu(self):
        from pyscf.lib import to_gpu
        obj = to_gpu(self)
        return obj.reset()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        self.surface = None
        self.v_grids_n = None
        return self

    def build(self, ng=None):
        if self.radii_table is None:
            vdw_scale = self.vdw_scale
            radii_table = vdw_scale * modified_Bondi + self.r_probe/radii.BOHR
        else:
            radii_table = self.radii_table
        logger.debug2(self, 'radii_table %s', radii_table)
        mol = self.mol
        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        self.surface = gen_surface(mol, rad=radii_table, ng=ng)
        self._intermediates = {}
        #F, A = get_F_A(self.surface)
        F, A = get_F_A_gcm(self.surface)
        #D, S = get_D_S(self.surface, with_S=True, with_D=True)
        D, S, O = get_D_S_gcm(self.surface, with_S=True, with_D=True)

        epsilon = self.eps
        if self.method.upper() in ['C-PCM', 'CPCM']:
            f_epsilon = (epsilon-1.)/epsilon if epsilon != float('inf') else 1.0
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
        elif self.method.upper() == 'COSMO':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0/2.0) if epsilon != float('inf') else 1.0
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
        elif self.method.upper() in ['IEF-PCM', 'IEFPCM']:
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.0
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(2.0*PI) * DAS
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        elif self.method.upper() == 'SS(V)PE':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.0
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(4.0*PI) * (DAS + DAS.T)
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        else:
            raise RuntimeError(f"Unknown implicit solvent model: {self.method}")

        intermediates = {
            'S': S,
            'D': D,
            'A': A,
            'K': K,
            'R': R,
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

        charge_exp  = self.surface['charge_exp']
        zeta        = charge_exp
        grid_coords = self.surface['grid_coords']
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()

        #int2c2e = mol._add_suffix('int2c2e')
        #fakemol = gto.fakemol_for_charges(grid_coords, expnt=charge_exp**2)
        #fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        #v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol)
        #self.v_grids_n = numpy.dot(atom_charges, v_ng)
        v_ng = numpy.zeros(len(grid_coords))
        for i, ri in enumerate(grid_coords):
            for A, (RA, ZA) in enumerate(zip(atom_coords, atom_charges)):
                r = numpy.linalg.norm(ri - RA)
                if r < 1e-10:
                    continue
                erf_term = erf(zeta[i] * r)
                v_ng[i] += ZA * erf_term / r
        self.v_grids_n = v_ng

    def _get_vind(self, dms):
        if not self._intermediates:
            self.build()

        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1,nao,nao)

        K = self._intermediates['K']
        R = self._intermediates['R']
        # eq.(19)
        v_grids_e = self._get_v(dms)
        v_grids   = self.v_grids_n - v_grids_e
        rhs       = numpy.dot(R, v_grids.T)

        n = K.shape[0]
        q = numpy.zeros_like(rhs.T)

        for i in range(rhs.shape[1]):
            A_ext = numpy.zeros((n + 1, n + 1))
            A_ext[:n, :n] = K
            A_ext[:n, n] = 1.0
            A_ext[n, :n] = 1.0
            b_ext = numpy.zeros(n + 1)
            b_ext[:n] = rhs[:, i]
            sol = numpy.linalg.solve(A_ext, b_ext)
            q[i, :] = sol[:n]

        vmat = self._get_vmat(q)
        epcm = 0.5 * numpy.dot(q[0], v_grids[0])
        vK_1 = numpy.linalg.solve(K.T, v_grids.T)
        qt = numpy.dot(R.T, vK_1).T
        q_sym = (q + qt)/2.0

        vmat = self._get_vmat(q_sym)
        epcm = 0.5 * numpy.dot(q_sym[0], v_grids[0])

        self._intermediates['q'] = q[0]
        self._intermediates['q_sym'] = q_sym[0]
        self._intermediates['v_grids'] = v_grids[0]
        self._intermediates['dm'] = dms
        print('Total COSMO charge:', numpy.sum(self._intermediates['q']))
        print('Max abs(q):', numpy.max(numpy.abs(self._intermediates['q'])))
        print('Min abs(q):', numpy.min(numpy.abs(self._intermediates['q'])))
        print('CPCM energy:',epcm)
        return epcm, vmat[0]

    def _get_v(self, dms):
        '''
        return electrostatic potential on surface
        '''
        mol = self.mol
        nao = dms.shape[-1]
        grid_coords = self.surface['grid_coords']
        exponents   = self.surface['charge_exp']
        ngrids = grid_coords.shape[0]
        nset = dms.shape[0]
        v_grids_e = numpy.empty([nset, ngrids])
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))
        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            for i in range(nset):
                v_grids_e[i,p0:p1] = numpy.einsum('ijL,ij->L',v_nj, dms[i])

        return v_grids_e

    def _get_vmat(self, q):
        mol = self.mol
        nao = mol.nao
        grid_coords = self.surface['grid_coords']
        exponents   = self.surface['charge_exp']
        ngrids = grid_coords.shape[0]
        q = q.reshape([-1,ngrids])
        nset = q.shape[0]
        vmat = numpy.zeros([nset,nao,nao])
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))

        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            for i in range(nset):
                vmat[i] += -numpy.einsum('ijL,L->ij', v_nj, q[i,p0:p1])
        return vmat

    def nuc_grad_method(self, grad_method):
        from pyscf.solvent.grad import pcm as pcm_grad
        if self.frozen:
            raise RuntimeError('Frozen solvent model is not supported')
        from pyscf import scf
        from pyscf.solvent import _ddcosmo_tdscf_grad
        if isinstance(grad_method.base, tdscf.rhf.TDBase):
            return _ddcosmo_tdscf_grad.make_grad_object(grad_method)
        else:
            return pcm_grad.make_grad_object(grad_method)

    def Hessian(self, hess_method):
        from pyscf.solvent.hessian import pcm as pcm_hess
        if self.frozen:
            raise RuntimeError('Frozen solvent model is not supported')
        from pyscf import scf
        if isinstance(hess_method.base, (scf.hf.RHF, scf.uhf.UHF)):
            return pcm_hess.make_hess_object(hess_method)
        else:
            raise RuntimeError('Only SCF gradient is supported')

    def _B_dot_x(self, dms):
        if not self._intermediates:
            self.build()
        out_shape = dms.shape
        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)

        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids = -self._get_v(dms)

        b = numpy.dot(R, v_grids.T)
        q = numpy.linalg.solve(K, b).T

        vK_1 = numpy.linalg.solve(K.T, v_grids.T)
        qt = numpy.dot(R.T, vK_1).T
        q_sym = (q + qt)/2.0

        vmat = self._get_vmat(q_sym)
        return vmat.reshape(out_shape)
