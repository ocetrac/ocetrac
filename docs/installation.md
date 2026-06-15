Installation
============
The easiest way to install Ocetrac is through **mamba** using the conda-forge package from the latest release. This method automatically installs all required dependencies for a complete Ocetrac setup and will work for most users. Ocetrac supports Python 3.9 and newer.

If you want to install the latest development version of Ocetrac, you can install directly from the repository hosted on GitHub. This method is recommended for users who want to contribute to the project or need access to the latest features and bug fixes that may not yet be included in a release.

The installation steps below apply to Linux, macOS, and Windows.

---

## Mamba (Recommended)

We recommend using [**mamba**](https://mamba.readthedocs.io/en/latest/) to install Ocetrac. Mamba is a fast, drop-in replacement for conda that resolves dependencies significantly faster and is especially helpful for packages with complex dependency trees like Ocetrac. If you already have conda installed but not mamba, you can install mamba into your base environment with:

```bash
conda install -n base -c conda-forge mamba
```

If you don't have conda or mamba installed, we recommend installing [Miniforge](https://github.com/conda-forge/miniforge), which ships with mamba by default. If you prefer to use conda instead of mamba, all of the commands below can be run with `conda` in place of `mamba`.

**Step 1:** Open a terminal on Linux or macOS, or the Anaconda Prompt on Windows. Activate the base environment, then create a new environment with Ocetrac and its core dependencies:

```bash
mamba activate base
mamba create -n ocetrac -c conda-forge ocetrac
```

**Step 2:** Activate the new Ocetrac environment:

```bash
mamba activate ocetrac
```

**Step 3:** Check that the installation was successful:

```bash
python -c "import ocetrac; print(ocetrac.__version__)"
```

**Step 4:** If you plan to run the examples, install a few extra packages:

```bash
mamba install -c conda-forge jupyter matplotlib cartopy
```

You can then download the Ocetrac tutorials from [the official repository](https://github.com/ocetrac/ocetrac).

---

## PyPI

You can also install Ocetrac with pip from PyPI, which is the Python package index. This method is recommended for users who want to quickly install Ocetrac without needing the latest development version.

To install Ocetrac from PyPI, run the following command in your terminal:

```bash
pip install ocetrac
```

---

## GitHub

To use the most up to date version of Ocetrac, you can install directly from the online repository hosted on GitHub.

1. Clone Ocetrac to your local machine:

```bash
git clone https://github.com/ocetrac/ocetrac
```

2. Change to the parent directory of Ocetrac.

3. Install Ocetrac with:

```bash
pip install -e ./ocetrac
```

This will allow changes you make locally to be reflected when you import the package in Python.

This method is recommended for users who want to contribute to the project or need access to the latest features and bug fixes that may not yet be included in a release.