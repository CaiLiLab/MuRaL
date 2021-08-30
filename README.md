## 1. Overview
**MuRaL**, short for **Mu**tation **Ra**te **L**earner, is a computational framework based on neural networks to generate single-nucleotide mutation rates across the genome.
## 2. Installation
MuRaL depends on several other packages and we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a virtual environment for installing MuRaL and its dependencies. Please refer to Miniconda's documentation for its installation.

After installing Miniconda, download or clone the MuRaL source code from github and go into the source code root folder.

MuRaL supports training and prediction with or without CUDA GPUs. If your computing environment has CUDA GPUs, you may check the CUDA driver version (e.g. though `nvidia-smi`) and specify a compatible `cudatoolkit` version in the `environment.yml` in the code folder. You can find the information about CUDA compatibility from [here] (https://docs.nvidia.com/deploy/cuda-compatibility/)

First, use `conda` command from Miniconda to create an environment and install the dependencies. The dependencies are included in the files `environment.yml` (GPU computing) and `environment_cpu.yml` (CPU-only computing). 

Then run one of the following to create a conda environment:
```
conda env create -n mural -f environment.yml # if your machine has GPUs
# or 
conda env create -n mural -f environment_cpu.yml # if your machine has GPUs

```
If the command ends without errors, you will have a conda environment named 'mural' (or use another name). Use the following command to activate the virtual environment:
```
conda activate mural
```
And install MuRaL using:
```
python setup.py install
```

If the installation is complete, three commands should be available from  the command line:
   * `mural_train`: This tool is for training mutation rate models from the begining.
   * `mural_train_TL`: This tool is for training transfer learning models, taking advantage of learned weights of a pre-trained model.
   * `mural_predict`: This tool is for predicting mutation rates of new sites using a trained model.

## 3. Examples
`mural_train` trains MuRaL models with training and validation mutation data and exports training results under the "./ray_results/" folder.
   * Input data
MuRaL requires input training and validation data files to be in BED format (more info about BED at https://genome.ucsc.edu/FAQ/FAQformat.html#format1). Some example lines of the input BED file are shown below.

```
chr1	2333436	2333437	.	0	+ \
chr1	2333446	2333447	.	2	- \
chr1	2333468	2333469	.	1	- \
chr1	2333510	2333511	.	3	- \
chr1	2333812	2333813	.	0	-   
```
   In the BED-formatted lines above, the 5th column is used to represent mutation status: usually, '0' means the non-mutated status and other numbers means specific mutation types (e.g. '1' for A>C, '2' for A>G, '3' for 'A>T'). You can specify a arbitrary order for a group of mutation types with incremental numbers starting from 1, but make sure that the same order is consistently used in training, validation and testing datasets.
    






