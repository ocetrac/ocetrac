Installation
============
The easiest way to install ocetrac is Ocetrac is through **conda** using the **conda-forge** package from the latest release. This method automatically installs all required dependencies for a complete Ocetrac setup and will work
for most users. Ocetrac supports Python 3.9 and newer.

If you want to install the latest development version of Ocetrac, you can install directly from the  repository hosted on GitHub. This method is recommended for users who want to contribute to the project or need access to the latest features and bug fixes that may not yet be included in a release.

The installation steps below apply to Linux, macOS, and Windows.

###### Conda
**Step 1:** Install Miniconda by following the instructions at https://docs.anaconda.com/miniconda/. If you’re on Linux /macOS, the following assumes that you installed Miniconda to your home directory.

**Step 2:** Open a terminal on Linux or macOS, or the Anaconda Prompt on Windows. Activate the base Miniconda environment, then create a new environment with Ocetrac and its core dependencies:

```bash
conda activate base
conda create -n ocetrac -c conda-forge ocetrac
```
**Step 3:** Activate the new Ocetrac environment:

```bash
conda activate ocetrac
```

**Step 4:** Check that the installation was successful:

```bash
python -c "import ocetrac; print(ocetrac.__version__)"
```

**Step 5:** If you plan to run the examples, install a few extra packages:

```bash
conda install -c conda-forge jupyter matplotlib cartopy
``` 

You can then download the Ocetrac tutorials from [the official repository](https://github.com/ocetrac/ocetrac).

###### PyPI
You can also install ocetrac with pip from PyPI, which is the Python package index. This method is recommended for users who want to quickly install ocetrac without needing the latest development version.
To install ocetrac from PyPI, run the following command in your terminal:

```bash
pip install ocetrac
```

###### GitHub
To use the most up to date version of ocetrac, you can install directly from the online repository hosted on GitHub.
1. Clone ocetrac to your local machine: 

```bash
git clone https://github.com/ocetrac/ocetrac
```

2. Change to the parent directory of ocetrac
3. Install ocetrac with:

```bash
pip install -e ./ocetrac
```

This will allow changes you make locally, to be reflected when you import the package in Python.

This method is recommended for users who want to contribute to the project or need access to the latest features and bug fixes that may not yet be included in a release.
