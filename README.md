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

## 1. Requirements

- Python **3.10** or later.
- A working C/C++ toolchain is recommended bacause PySCF features compile on first use.
- The package depends on `numpy`, `tabulate`, and `pyscf`. \
  These are installed automatically when you install `gchsm`.

> **Tip for beginners**: if you are new to Python packaging, using a virtual
> environment keeps the installation isolated from the rest of your system.  A
> virtual environment is simply a private copy of Python and its packages.

## 2. Installation

The commands below assume that you are starting from your terminal (macOS/Linux) or PowerShell (Windows).

**2.1. Download the source code.**
   ```bash
   git clone https://github.com/Takuya-0606/gchsm.git
   cd gchsm
   ```

**2.2. Create and activate a dedicated Conda environment. (Recommended)**
  ```bash
   python3 -m venv .name
   # Activate it
   source .name/bin/activate          # macOS / Linux
   # .name\Scripts\Activate.ps1      # Windows PowerShell
  ```
  This environment contains Python itself, the scientific dependencies, and the tooling required to build the Conda package locally.

**2.3. Install the package in editable mode.**
  ```bash
  pip install --upgrade pip
  pip install -e .
  ```
  The installer downloads the required scientific libraries and places the `gchsm` command on your `PATH`.

**2.4. Verify the installation.**

   ```bash
   gchsm --help
   ```

   The program prints a short usage message.  If the command is not found,
   ensure that the virtual environment is activated and that the installation
   step completed without errors.

**2.5. Overwrite pyscf.**

```bash
cp code/pcm.py .name/lib/python3.12/site-packages/pyscf/solvent/pcm.py
cp code/hessian/pcm.py .name/lib/python3.12/site-packages/pyscf/solvent/hessian/pcm.py
cp code/thermo.py .name/lib/python3.12/site-packages/pyscf/solvent/hessian/thermo.py
```

The following program was implemented by appending to the pyscf program.
- numerical second derivative calculation with fixed PCM cavity
- Thermodynamics calculation


## 3. Running calculation
1. Prepare an input file that contains `%MAIN` and `%GEOMETRY` blocks.\
   An example named `input.inp` is shipped next to this README.

2. From the same directory as the input file, run:
   ```bash
   gchsm input.inp
   ```

3. The program writes a detailed report next to the input file (for the example above, the output is `input.out`).  When `CALCTYPE=freq` is specified, the program expects a `freq.out` file in the same folder that contains tabulated vibrational frequencies; the thermochemical tables are then generated from that data.

## 4. Updating or uninstalling
- To pick up the latest changes from this repository, update the sources (for
  example with `git pull`) and re-run `pip install -e .` inside the
  `gchsm` directory.
- To uninstall the package from the active Python environment, run:

  ```bash
  pip uninstall gchsm
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
