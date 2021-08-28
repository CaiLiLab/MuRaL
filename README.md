## 1.Overview
**MuRaL**, short for **Mu**tation **Ra**te **L**earner, is a computational framework based on neural networks to generate single-nucleotide mutation rates across the genome.
## 2.Installation
MuRaL depends on several other packages and we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a virtual environment for installing MuRaL and its dependencies. Please refer to Miniconda's documentation for installation.

After installing Miniconda, download or clone the package source code from github and go into the folder code root folder.

First, use `conda` command from Miniconda to create an environment and install the dependencies:
```
$ conda env create --nanme mural -f environment.yml
```
If the command ends without errors, you will have a conda environment named 'mural' (or use another name). 

