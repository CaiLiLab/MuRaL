## 1.Overview
**MuRaL**, short for **Mu**tation **Ra**te **L**earner, is a computational framework based on neural networks to generate single-nucleotide mutation rates across the genome.
## 2.Installation
MuRaL depends on several other packages and we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a virtual environment for installing MuRaL and its dependencies. Please refer to Miniconda's documentation for installation.

After installing Miniconda, download or clone the MuRaL source code from github and go into the source code root folder.

First, use `conda` command from Miniconda to create an environment and install the dependencies. The dependencies are included in the files `environment.yml` and `environment_cpu.yml`. 

MuRaL supports training and prediction with or without CUDA GPUs. If you computing environment has CUDA GPUs, you may check the CUDA driver version (e.g. though `nvidia-smi`) and specify a compatible `cudatoolkit` version in the `environment.yml`. You can find the information about CUDA compatibility from [here] (https://docs.nvidia.com/deploy/cuda-compatibility/)

Then run one of the following to create a conda environment:
```
conda env create -n mural -f environment.yml # if your machine has GPUs

or 
conda env create -n mural -f environment_cpu.yml # if your machine has GPUs

```
If the command ends without errors, you will have a conda environment named 'mural' (or use another name). 

