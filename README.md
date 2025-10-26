# Gaussian charge distributed harmonic solvation model (GC-HSM)
2025-10-26
* [Documentation](https://)
* [Installation](#installation)
* [Users guide](https://)
  
The source code for GC-HSM functions by connecting with [PySCF](https://github.com/pyscf/pyscf).\
Please follow the steps below to set up your environment.

# Installation
* Install pyscf ver.2.9.0:
```
conda install -c conda-forge pyscf=2.9.0
```
* Certain modules are maintained as PySCF extensions like dispersion calculations, semi-empirical methods, etc...
These modules can be installed using pip:
```
pip install pyscf[all]
```
* Install GCHSM driver:
```
git clone https://hogehoge
```
## Note
Polarizable continuum model based on gaussian charge scheme is also available through the following programs in addition to pyscf:
* [ORCA](https://www.faccts.de/docs#orca)
* [TURBOMOLE](https://www.turbomole.org/)
