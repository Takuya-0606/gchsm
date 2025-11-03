#!/usr/bin/env python

from pyscf import gto, scf
from pyscf.hessian import thermo
from pyscf import gto, scf, mp
from pyscf.hessian import thermo
from pyscf.tools.finite_diff import Hessian as FDHessian
import numpy as np
from hsm import fixcav

def main():
    mol = gto.M(
        atom='''
O   0.00000000000000  0.00000000000000  0.14989271676966
H   0.00000000000000  0.75233197920560 -0.44864635838483
H   0.00000000000000 -0.75233197920560 -0.44864635838484
''',
        basis='cc-pvtz',
        unit='Angstrom',
        verbose=4
    )

    pcm = fixcav(
        mol,
        centers=mol.atom_coords(unit="Angstrom"),
        radii=1.2,
        method="C-PCM",
        eps=80.1510
    )
    pcm.lebedev_order = 17
    pcm.build()

    mf = pcm.attach_to(scf.RHF(mol), build_surface=False)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-6
    print("Running RHF + PCM (frozen cavity) ...")
    mf.kernel()

    mymp2 = mp.MP2(mf)
    mymp2.conv_tol = 1e-10
    print("Running MP2 on PCM-reference ...")
    mymp2.kernel()

    grad = mymp2.nuc_grad_method()
    grad.verbose = 4

    g0 = grad.kernel()
    print("||Grad(HF+PCM)|| =", float(np.linalg.norm(g0)))

    h_step = 5.0e-3
    fdh = FDHessian(grad)
    fdh.displacement = h_step

    print("Building Hessian via finite-difference of analytic gradients (HF+PCM, frozen cavity) ...")
    H = fdh.kernel()

    mass   = mol.atom_mass_list(isotope_avg=True)
    coords = mol.atom_coords(unit='Bohr')

    freq_info = thermo.harmonic_analysis(
        mol, H,
        imaginary_freq=True,
        exclude_trans=False,
        exclude_rot=False,
    )
    freqs_tr = thermo.compute_tr_frequencies(H, mass, coords)
    tr, vib, full, nTR = thermo.collect_freq(mass, coords, freq_info, freqs_tr)

    res = thermo.calc_hsm(tr, vib, T=298.15)
    thermo.print_hsm_tables(res)

if __name__ == "__main__":
    main()
