## 1.Overview
**MuRaL**, short for **Mu**tation **Ra**te **L**earner, is a computational framework based on neural networks to generate single-nucleotide mutation rates across the genome.
## 2.Installation
MuRaL depends on several other packages and we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a virtual environment for installing MuRaL and its dependencies. Please refer to Miniconda's documentation for its installation.

After installing Miniconda, download or clone the MuRaL source code from github and go into the source code root folder.

MuRaL supports training and prediction with or without CUDA GPUs. If you computing environment has CUDA GPUs, you may check the CUDA driver version (e.g. though `nvidia-smi`) and specify a compatible `cudatoolkit` version in the `environment.yml` in the code folder. You can find the information about CUDA compatibility from [here] (https://docs.nvidia.com/deploy/cuda-compatibility/)

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

## 3. A brief example

```
usage: mural_train [-h] --ref_genome FILE --train_data FILE [--validation_data FILE]
                   [--valid_ratio FLOAT] [--split_seed INT] [--bw_paths FILE] [--seq_only]
                   [--save_valid_preds] [--model_no INT] [--n_class INT]
                   [--local_radius INT [INT ...]] [--local_order INT [INT ...]]
                   [--local_hidden1_size INT [INT ...]] [--local_hidden2_size INT [INT ...]]
                   [--distal_radius INT [INT ...]] [--distal_order INT]
                   [--emb_dropout FLOAT [FLOAT ...]] [--local_dropout FLOAT [FLOAT ...]]
                   [--CNN_kernel_size INT [INT ...]] [--CNN_out_channels INT [INT ...]]
                   [--distal_fc_dropout FLOAT [FLOAT ...]] [--batch_size INT [INT ...]]
                   [--optim STRING [STRING ...]] [--learning_rate FLOAT [FLOAT ...]]
                   [--weight_decay FLOAT [FLOAT ...]] [--LR_gamma FLOAT [FLOAT ...]]
                   [--experiment_name STRING] [--n_trials INT] [--epochs INT] [--grace_period INT]
                   [--ASHA_metric STRING] [--ray_ncpus INT] [--ray_ngpus INT] [--cpu_per_trial INT]
                   [--gpu_per_trial FLOAT] [--cuda_id STRING] [--rerun_failed]

    Overview
    --------
    
    This tool trains MuRaL models with training and validation mutation data
    and exports training results under the "./ray_results/" folder.
    
    * Input data
    MuRaL requires input training and validation data files to be in BED format
    (more info about BED at https://genome.ucsc.edu/FAQ/FAQformat.html#format1). 
    Some example lines of the input BED file are shown below.
    chr1        2333436 2333437 .       0       +
    chr1        2333446 2333447 .       2       -
    chr1        2333468 2333469 .       1       -
    chr1        2333510 2333511 .       3       -
    chr1        2333812 2333813 .       0       -
    
    In the BED-formatted lines above, the 5th column is used to represent mutation
    status: usually, '0' means the non-mutated status and other numbers means 
    specific mutation types (e.g. '1' for A>C, '2' for A>G, '3' for 'A>T'). You can
    specify a arbitrary order for a group of mutation types with incremental 
    numbers starting from 1, but make sure that the same order is consistently 
    used in training, validation and testing datasets. 
    
    Importantly, the training and validation BED file MUST be SORTED by chromosome
    coordinates. You can sort BED files by 'bedtools sort' or 'sort -k1,1 -k2,2n'.
    
    * Output data
    The checkpointed model files during training are saved under folders named like 
        ./ray_results/your_experiment_name/Train_xxx...xxx/checkpoint_x/
            - model
            - model.config.pkl
            - model.fdiri_cal.pkl
    
    In the above folder, the 'model' file contains the learned model parameters. 
    The 'model.config.pkl' file contains configured hyperparameters of the model.
    The 'model.fdiri_cal.pkl' file (if exists) contains the calibration model 
    learned with validation data, which can be used for calibrating predicted 
    mutation rates. These files can be used in downstream analyses such as
    model prediction and transfer learning.
    
    Command line examples
    ---------------------
    
    1. The following command will train a model by running two trials, using data in
    'train.sorted.bed' for training. The training results will be saved under the
    folder './ray_results/example1/'. Default values will be used for other
    unspecified arguments. Note that, by default, 10% of the sites sampled from 
    'train.sorted.bed' is used as validation data (i.e., '--valid_ratio 0.1').
    
        mural_train --ref_genome seq.fa --train_data train.sorted.bed \
        --n_trials 2 --experiment_name example1 > test1.out 2> test1.err
    
    2. The following command will use data in 'train.sorted.bed' as training
    data and a separate 'validation.sorted.bed' as validation data. The option
    '--local_radius 10' means that length of the local sequence used for training
    is 10*2+1 = 21 bp. '--distal_radius 100' means that length of the expanded 
    sequence used for training is 100*2+1 = 201 bp. 
    
        mural_train --ref_genome seq.fa --train_data train.sorted.bed \
        --validation_data validation.sorted.bed --n_trials 2 --local_radius 10 \ 
        --distal_radius 100 --experiment_name example2 > test2.out 2> test2.err
    
    3. If you don't have (or don't want to use) GPU resources, you can set options
    '--ray_ngpus 0 --gpu_per_trial 0' as below. Be aware that if training dataset 
    is large or the model is parameter-rich, CPU-only computing could take a very 
    long time!
    
        mural_train --ref_genome seq.fa --train_data train.sorted.bed \
        --n_trials 2 --ray_ngpus 0 --gpu_per_trial 0 --experiment_name example3 \ 
        > test3.out 2> test3.err
    
    Notes
    -----
    1. The training and validation BED file MUST BE SORTED by chromosome 
    coordinates. You can sort BED files by running 'bedtools sort' or 
    'sort -k1,1 -k2,2n'.
    
    2. By default, this tool generates a HDF5 file for each input BED
    file (training or validation file) based on the value of '--distal_radius' 
    and the tracks in '--bw_paths', if the corresponding HDF5 file doesn't 
    exist or is corrupted. Only one job is allowed to write to an HDF5 file,
    so don't run multiple jobs involving a same BED file when its HDF5 file 
    isn't generated yet. Otherwise, it may cause file permission errors.
    
    3. If it takes long to finish a job, you can check the information exported 
    to stdout (or redirected file) for the progress during running.
    
    

Required arguments:
  --ref_genome FILE     File path of the reference genome in FASTA format.
  --train_data FILE     File path of training data in a sorted BED format. If the options
                        --validation_data and --valid_ratio not specified, 10% of the
                        sites sampled from the training BED will be used as the
                        validation data.

Data-related arguments:
  --validation_data FILE
                        File path for validation data. If this option is set,
                        the value of --valid_ratio will be ignored. Default: None.
  --valid_ratio FLOAT   Ratio of validation data relative to the whole training data.
                        Default: 0.1.
  --split_seed INT      Seed for randomly splitting data into training and validation
                        sets. Default: a random number generated by the job.
  --bw_paths FILE       File path for a list of BigWig files for non-sequence 
                        features such as the coverage track. Default: None.
  --seq_only            If set, use only genomic sequences for the model and ignore
                        bigWig tracks. Default: False.
  --save_valid_preds    Save prediction results for validation data in the checkpoint
                        folders. Default: False.

Model-related arguments:
  --model_no INT        Which network architecture to be used: 
                        0 - 'local-only' model;
                        1 - 'expanded-only' model;
                        2 - 'local + expanded' model. 
                        Default: 2.
  --n_class INT         Number of mutation classes (or types), including the 
                        non-mutated class. Default: 4.
  --local_radius INT [INT ...]
                        Radius of the local sequence to be considered in the 
                        model. Length of the local sequence = local_radius*2+1 bp.
                        Default: 5.
  --local_order INT [INT ...]
                        Length of k-mer in the embedding layer. Default: 3.
  --local_hidden1_size INT [INT ...]
                        Size of 1st hidden layer for local module. Default: 150.
  --local_hidden2_size INT [INT ...]
                        Size of 2nd hidden layer for local module. 
                        Default: local_hidden1_size//2 .
  --distal_radius INT [INT ...]
                        Radius of the expanded sequence to be considered in the model. 
                        Length of the expanded sequence = distal_radius*2+1 bp.
                        Default: 50.
  --distal_order INT    Order of distal sequences to be considered. Kept for 
                        future development. Default: 1.
  --emb_dropout FLOAT [FLOAT ...]
                        Dropout rate for inputs of the k-mer embedding layer.
                        Default: 0.1.
  --local_dropout FLOAT [FLOAT ...]
                        Dropout rate for inputs of local hidden layers.  Default: 0.1.
  --CNN_kernel_size INT [INT ...]
                        Kernel size for CNN layers in the expanded module. Default: 3.
  --CNN_out_channels INT [INT ...]
                        Number of output channels for CNN layers. Default: 32.
  --distal_fc_dropout FLOAT [FLOAT ...]
                        Dropout rate for the FC layer of the expanded module.
                        Default: 0.25.

Learning-related arguments:
  --batch_size INT [INT ...]
                        Size of mini batches for model training. Default: 128.
  --optim STRING [STRING ...]
                        Optimization method for parameter learning.
                        Default: 'Adam'.
  --learning_rate FLOAT [FLOAT ...]
                        Learning rate for parameter learning, an argument for the 
                        optimization method.  Default: 0.005.
  --weight_decay FLOAT [FLOAT ...]
                        'weight_decay' argument (regularization) for the optimization 
                        method.  Default: 1e-5.
  --LR_gamma FLOAT [FLOAT ...]
                        'gamma' argument for the learning rate scheduler.
                         Default: 0.5.

RayTune-related arguments:
  --experiment_name STRING
                        Ray-Tune experiment name.  Default: 'my_experiment'.
  --n_trials INT        Number of trials for this training job.  Default: 3.
  --epochs INT          Maximum number of epochs for each trial.  Default: 10.
  --grace_period INT    'grace_period' parameter for early stopping. 
                         Default: 5.
  --ASHA_metric STRING  Metric for ASHA schedualing; the value can be 'loss' or 'score'.
                        Default: 'loss'.
  --ray_ncpus INT       Number of CPUs requested by Ray-Tune. Default: 6.
  --ray_ngpus INT       Number of GPUs requested by Ray-Tune. Default: 1.
  --cpu_per_trial INT   Number of CPUs used per trial. Default: 3.
  --gpu_per_trial FLOAT
                        Number of GPUs used per trial. Default: 0.19.
  --cuda_id STRING      Which GPU device to be used. Default: '0'.
  --rerun_failed        Rerun errored or incomplete trials. Default: False.

Other arguments:
  -h, --help            show this help message and exit
  
```




