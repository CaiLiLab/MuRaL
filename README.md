## 1. Overview
**MuRaL**, short for **Mu**tation **Ra**te **L**earner, is a computational framework based on neural networks to generate single-nucleotide mutation rates across the genome. 

The MuRaL network architecture has two major modules (shown below), one is for learning signals from local genomic regions (<20bp from the focal nucleotide) of a focal nucleotide, the other for learning signals from expanded regions (up to 1kb from the focal nucleotide).

<img src="./images/model_schematic.jpg" alt="model schematic" width="700"/>

## 2. Installation
MuRaL depends on several other packages, and we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a conda environment for installing MuRaL and its dependencies. Please refer to Miniconda's documentation for its installation.

After installing Miniconda, download or clone the MuRaL source code from github and go into the **source code root folder**.

MuRaL supports training and prediction with or without CUDA GPUs. If your computing environment has CUDA GPUs, you may check the CUDA driver version (e.g. though `nvidia-smi`) and specify a compatible `cudatoolkit` version in the `environment.yml` file under the code root folder. You can find the information about CUDA compatibility from [here](https://docs.nvidia.com/deploy/cuda-compatibility/)

Before installing MuRaL, use `conda` command from Miniconda to create an environment and install the dependencies. The dependencies are included in `environment.yml` (if using GPUs) or `environment_cpu.yml` (if CPU-only computing). 

Then run one of the following to create a conda environment:
```
conda env create -n mural -f environment.yml # if your machine has GPUs
# or 
conda env create -n mural -f environment_cpu.yml # if your machine has only CPUs
```
If the command ends without errors, you will have a conda environment named 'mural' (or another name if you change the `-n mural` option). Use the following command to activate the conda environment:
```
conda activate mural
```
And then install MuRaL by typing:
```
python setup.py install
```

If the installation is complete, three commands are available from the command line:
   * `mural_train`: This tool is for training mutation rate models from the beginning.
   * `mural_train_TL`: This tool is for training transfer learning models, taking advantage of learned weights of a pre-trained model.
   * `mural_predict`: This tool is for predicting mutation rates of new sites using a trained model.

## 3. Usage Examples
### 3.1 Model training
`mural_train` trains MuRaL models with training and validation mutation data and exports training results under the "./ray_results/" folder.
   * Input data \
   MuRaL requires input training and validation data files to be in BED format (more info about BED at https://genome.ucsc.edu/FAQ/FAQformat.html#format1). Some example lines of the input BED file are shown below.
```
chr1	2333436	2333437	.	0	+ 
chr1	2333446	2333447	.	2	-
chr1	2333468	2333469	.	1	-
chr1	2333510	2333511	.	3	-
chr1	2333812	2333813	.	0	- 
```
   In the BED-formatted lines above, the 5th column is used to represent mutation status: usually, '0' means the non-mutated status and other numbers means specific mutation types (e.g. '1' for A>C, '2' for A>G, '3' for 'A>T'). You can specify a arbitrary order for a group of mutation types with incremental numbers starting from 1, but make sure that the same order is consistently used in training, validation and testing datasets. Importantly, the training and validation BED file MUST be SORTED by chromosome coordinates. You can sort BED files by `bedtools sort` or `sort -k1,1 -k2,2n`.

   * Output data \
   The checkpointed model files during training are saved under folders named like:
```
    ./ray_results/your_experiment_name/Train_xxx...xxx/checkpoint_x/
            - model
            - model.config.pkl
            - model.fdiri_cal.pkl
```
   In the above folder, the 'model' file contains the learned model parameters. The 'model.config.pkl' file contains configured hyperparameters of the model. The 'model.fdiri_cal.pkl' file (if exists) contains the calibration model learned with validation data, which can be used for calibrating predicted mutation rates. These files can be used in downstream analyses such as model prediction and transfer learning.
    
   * Example 1 \
   The following command will train a model by running two trials, using data in 'train.sorted.bed' for training. The training results will be saved under the folder './ray_results/example1/'. Default values will be used for other unspecified arguments. Note that, by default, 10% of the sites sampled from 'train.sorted.bed' is used as validation data (i.e., '--valid_ratio 0.1').
```
mural_train --ref_genome seq.fa --train_data train.sorted.bed \
        --n_trials 2 --experiment_name example1 > test1.out 2> test1.err
```
   * Example 2 \
   The following command will use data in 'train.sorted.bed' as training data and a separate 'validation.sorted.bed' as validation data. The option '--local_radius 10' means that length of the local sequence used for training is 10\*2+1 = 21 bp. '--distal_radius 100' means that length of the expanded sequence used for training is 100\*2+1 = 201 bp.
```
mural_train --ref_genome seq.fa --train_data train.sorted.bed \
        --validation_data validation.sorted.bed --n_trials 2 --local_radius 10 \
        --distal_radius 100 --experiment_name example2 > test2.out 2> test2.err
```

### 3.2 Model prediction
`mural_predict` predicts mutation rates for all sites in a BED file based on a trained model.
   * Input data \
   The input data for prediction includes a BED-formated file and a trained model. The BED file is organized in the same way as that for training. The 5th column can be set to '0' if no observed mutations for the sites in the prediction BED. The model-related files for input are 'model' and 'model.config.pkl', which are generated at the training step. The file 'model.fdiri_cal.pkl', which is for calibrating predicted mutation rates, is optional.
   * Output data \
   The output of `mural_predict` is a tab-separated file containing the chromosome positional information and the predicted probabilities for all possible mutation types. The 'prob0' column contains probalities for the non-mutated class and other 'probX' columns for mutated classes. 
   Some example lines of the prediction output file are shown below:
```
chrom   start   end    strand mut_type  prob0   prob1   prob2   prob3
chr1    10006   10007   -       0       0.9797  0.003134 0.01444 0.002724
chr1    10007   10008   +       0       0.9849  0.005517 0.00707 0.002520
chr1    10008   10009   +       0       0.9817  0.004801 0.01006 0.003399
chr1    10012   10013   -       0       0.9711  0.004898 0.02029 0.003746
```
   
   * Example 3 \
   The following command will predict mutation rates for all sites in 'testing.bed.gz' using model files under the 'checkpoint_6/' folder and save the prediction results into 'testing.ckpt6.fdiri.tsv.gz'.
```
mural_predict --ref_genome seq.fa --test_data testing.bed.gz \
        --model_path checkpoint_6/model --model_config_path checkpoint_6/model.config.pkl \
        --calibrator_path checkpoint_6/model.fdiri_cal.pkl --pred_file testing.ckpt6.fdiri.tsv.gz \
        > test.out 2> test.err
```



   
   
    






