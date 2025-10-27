# Gaussian charge distributed harmonic solvation model (GC-HSM)
2025-10-26
* [Document](document)
* [Installation](#installation)
  
The source code for GC-HSM functions by connecting with [PySCF](https://github.com/pyscf/pyscf).\
Please follow the steps below to set up your environment.

# Installation
## GCHSM ver.1.0 (Python package)

GCHSM driver provides a workflow around [PySCF](https://pyscf.org/) for single-point, geometry
optimisation, Hessian, and post-processing of precomputed vibrational frequency
data. \
The package exposes the same logic that was previously distributed as a
collection of loose scripts, but wraps it in an installable Python package so
that it can be invoked anywhere with a single command: `gchsm`.
* Install GCHSM driver:
```
conda install -c conda-forge pyscf=2.9.0
```
## 1. Requirements

- Python **3.10** or later.
- A working C/C++ toolchain is recommended bacause PySCF features compile on first use.
- The package depends on `numpy`, `tabulate`, and `pyscf`. \
  These are installed automatically when you install `gchsm`.

> **Tip for beginners**: if you are new to Python packaging, using a virtual
> environment keeps the installation isolated from the rest of your system.  A
> virtual environment is simply a private copy of Python and its packages.

## 2. Installation

The commands below assume that you are starting from a terminal (macOS/Linux) or PowerShell (Windows).

**2.1. Download the source code.**
   ```bash
   git clone https://github.com/Takuya-0606/gchsm.git
   cd hsm
   ```

**2.2. Create and activate a dedicated Conda environment. (Recommended)**
  ```bash
  conda create -n gchsm python=3.10 numpy tabulate conda-build
  conda activate gchsm
  ```
  This environment contains Python itself, the scientific dependencies, and the tooling required to build the Conda package locally.

**2.3. Build the Conda package.**
  ```bash
  conda build conda
  ```
  Conda places the resulting package (a `.tar.bz2` file) in the directory shown at the end of the build log, typically something like
   `~/miniconda3/conda-bld/noarch/`.

**2.4. Install the newly built package.**
   ```bash
   conda install --use-local gchsm
   ```

   The `--use-local` flag tells Conda to install the package that you just
   produced in the previous step.  After the installation finishes, the `gchsm`
   command becomes available in the active environment.

**2.5. Verify the installation.**
   ```bash
   gchsm --help
   ```

   The program prints a short usage message.  If the command is not found,
   ensure that the Conda environment is active and that the installation step
   completed without errors.

## 3. Running calculation
1. Prepare an input file that contains `%MAIN` and `%GEOMETRY` blocks.\
   An example named `input.inp` is shipped next to this README.

2. From the same directory as the input file, run:
   ```bash
   gchsm input.inp
   ```

3. The program writes a detailed report next to the input file (for the example above, the output is `input.out`).  When `CALCTYPE=freq` is specified, the program expects a `freq.out` file in the same folder that contains tabulated vibrational frequencies; the thermochemical tables are then generated from that data.

## 4. Updating or uninstalling
- To pick up the latest changes from this repository, update the sources (for example with `git pull`), rebuild the Conda package with `conda build conda`, and reinstall it using `conda install --use-local gchsm`.
- To uninstall the package from the active Conda environment, run:
  ```bash
  conda remove gchsm
  ```

## 5. Getting help
- The original text-based workflow remains available through the installed
  `gchsm` command.  Run `gchsm --help` to display the command-line synopsis.
- Review the generated `.out` file if a calculation fails.  Diagnostic
  information, including Python tracebacks, is captured there.

We hope this Conda-first distribution makes it easier to integrate GCHSM into your workflows!

## Note
- For detailed operations of GCHSM, please refer to the [Document](document).
- Polarizable continuum model based on gaussian charge scheme is also available through the following programs in addition to pyscf:
  * [ORCA](https://www.faccts.de/docs#orca)
  * [TURBOMOLE](https://www.turbomole.org/)
