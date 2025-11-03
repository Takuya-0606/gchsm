#!/usr/bin/env python

from pyscf import gto, scf
from pyscf.hessian import thermo
from hsm import fixcav
from tabulate import tabulate
import numpy

mol = gto.M(atom="""
  O           0.00000000000000      0.00000000000000      0.14989271676966
  H           0.00000000000000      0.75233197920560     -0.44864635838483
  H           0.00000000000000     -0.75233197920560     -0.44864635838484
""", unit="Ang", basis="sto-3g", verbose=1)

pcm = fixcav(mol,
                    centers=mol.atom_coords(unit="Angstrom"),
                    radii=1.2,
                    method="C-PCM",
                    eps=80.1510)
pcm.lebedev_order = 17
pcm.build()

mf = pcm.attach_to(scf.RHF(mol), build_surface=False)
mf.kernel()

hess = pcm.mp2_frequency(mf, disp_bohr=5e-3)

freq_info = thermo.harmonic_analysis(
        mf.mol,
        hess,
        imaginary_freq=True,
        exclude_trans=False,
        exclude_rot=False,
)
mass = mf.mol.atom_mass_list(isotope_avg=True)
coords = mf.mol.atom_coords(unit='Bohr')
freqs_tr = thermo.compute_tr_frequencies(hess, mass, coords)
tr, vib, full, nTR = thermo.collect_freq(mass, coords, freq_info, freqs_tr)
out = thermo.show_frequencies(mass, coords, hess, freq_info)
res = thermo.calc_hsm(tr, vib, T=298.15)
thermo.print_hsm_tables(res)
