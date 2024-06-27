Overview
--------

Germline mutation rates are important in genetics, genomics and
evolutionary biology. It is long known that mutation rates vary
substantially across the genome, but existing methods can only obtain
very rough estimates of local mutation rates and are difficult to be
applied to non-model species.

**MuRaL**, short for **Mu**\ tation **Ra**\ te **L**\ earner, is a
generalizable framework to estimate single-nucleotide mutation rates
based on deep learning. MuRaL has better predictive performance at
different scales than current state-of-the-art methods. Moreover, it can
generate genome-wide mutation rate maps with rare variants from a
moderate number of sequenced individuals (e.g. ~100 individuals), and
can leverage transfer learning to further reduce data and time
requirements. It can be applied to many sequenced species with
population polymorphism data.

The MuRaL network architecture has two main modules (shown below), one
is for learning signals from local genomic regions (e.g. 10bp on each
side of the focal nucleotide), the other for learning signals from
expanded regions (e.g. 1Kb on each side of the focal nucleotide).

.. image:: images/model_schematic.jpg
   :width: 830px

Below is an example showing that MuRaL-predicted rates (colored lines)
are highly correlated with observed mutation rates (grey shades) at
different scales on Chr3 of *A. thaliana*.

.. image:: images/regional_correlation_example.jpg
   :width: 500px

Installation
------------

You can install MuRaL with conda, or download pre-built Singularity
images if Singularity works in your system. More details are given
below.

Using Conda
~~~~~~~~~~~

MuRaL depends on several other packages, and we recommend using
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ (version
3 or newer) to create a conda environment for installing MuRaL and its
dependencies. Please refer to Miniconda's documentation for its
installation.

After installing Miniconda, download or clone the MuRaL source code from
github and go into the source code root folder 'MuRal-xxx/'.

MuRaL supports training and prediction with or without CUDA GPUs. Please
be aware that training tasks without GPUs could take a much longer time.
For most models, prediction tasks can be done with only CPUs.

Before installing MuRaL, use the ``conda`` command from Miniconda to create
an environment and install the dependencies. The dependencies are
included in ``environment.yml`` (if using GPUs) or
``environment_cpu.yml`` (if CPU-only computing). Run one of the
following commands to create a conda environment and install the
dependencies (this may take >30 minutes depending on your internet
speed):

::

    # if your machine has GPUs
    conda env create -n mural -f environment.yml 

    # if the above command is interupted because of internet issues or some dependencies 
    # in environment.yml are updated, try the following:
    conda env update -n mural -f environment.yml --prune


    # if your machine has only CPUs
    conda env create -n mural -f environment_cpu.yml 

If the command ends without errors, you will have a conda environment
named 'mural'. Use the following command to activate the conda
environment:

::

    conda activate mural

And then install MuRaL by typing:

::

    pip install .

If the installation is complete, you can type ``mural_train -v`` to get
the MuRaL version.

Using Singularity
~~~~~~~~~~~~~~~~~

Singularity is a popular container platform for scientific research. We
also built Singularity images for specific versions, which can be found
at this `OSF repo <https://osf.io/rd9k5/>`__. You can just download the
Singularity image ``mural_vx.x.x.sif`` from the OSF repo and don't need
to install the dependencies of MuRaL. Once Singularity is installed in
your system, you can try running the MuRaL commands with the
``mural_vx.x.x.sif`` file.

If your machine has GPUs and you want to use GPU resources for MuRaL
tools, please remember to set the ``--nv`` flag for Singularity commands.
See the following examples:

::

    singularity exec --nv /path/to/mural_vx.x.x.sif mural_train ...
    singularity exec --nv /path/to/mural_vx.x.x.sif mural_train_TL ...

For prediction tasks, it is recommended to use only CPUs so that you can
run many prediction tasks in parallel. See the example below:

::

   singularity exec /path/to/mural_vx.x.x.sif mural_predict ...

For more about Singularity, please refer to the `Singularity
documentation <https://docs.sylabs.io>`__.

Tools and examples
------------------

The following tools in MuRaL are available from the command line. Type 
a command with '-h' option to see the detailed help message. More specific 
examples are given in later sections.

**Main commands**: 

* ``mural_train``: This tool is for training mutation rate models from 
  the beginning. 
* ``mural_train_TL``: This tool is for training transfer learning models, 
  taking advantage of learned weights of a pre-trained model. 
* ``mural_predict``: This tool is for predicting mutation rates of new 
  sites with a trained model.

**Auxiliary commands**: 

* ``get_best_mural_models``: This tool is for finding the best model 
  per trial, given the 'progress.csv' files of trials. 
* ``calc_mu_scaling_factor``: This tool is for calculating
  scaling factors for generating per-generation mutation rates.
* ``scale_mu``: This tool is for scaling raw MuRaL-predicted mutation
  rates into per-generation rates given a scaling factor.
* ``calc_kmer_corr``: This tool is for calculating kmer mutation rate 
  correlations for evaluation.
* ``calc_region_corr``: This tool is for calculating regional mutation
  rate correlations for evaluation.

Model training
~~~~~~~~~~~~~~

``mural_train`` trains MuRaL models with training and validation
mutation data. It exports training results to different folders 
based on whether Ray is used for hyperparameter search. If Ray is used, 
the results are saved under the './ray_results/' folder. 
Otherwise, they are saved under the './results/' folder.

* Input data
   
Input data files include the reference sequence file (FASTA format,
required), a training data file (required) and a validation data file
(optional). If the validation data file isn't provided, a fraction of
the sites sampled from the training data file are used as validation
data.
Input training and validation data files are in BED format (more info
about BED format
`here <https://genome.ucsc.edu/FAQ/FAQformat.html#format1>`__). Some
example lines of an input BED file are shown below.

::

    chr1    2333436 2333437 .   0   + 
    chr1    2333446 2333447 .   2   -
    chr1    2333468 2333469 .   1   -
    chr1    2333510 2333511 .   3   -
    chr1    2333812 2333813 .   0   - 

In the BED-formatted lines above, the 5th column is used to represent
mutation status: usually, '0' means the non-mutated status and other
numbers for specific mutation types (e.g. '1' for 'A>C', '2' for 'A>G',
'3' for 'A>T'). You can specify an arbitrary order for a group of
mutation types with incremental numbers starting from 0, but make sure
that the same order is consistently used in training, validation and
testing datasets. Importantly, the training and validation BED file MUST
BE SORTED by chromosome coordinates. You can sort BED files by
``bedtools sort`` or ``sort -k1,1 -k2,2n``.

* Output data

``mural_train`` saves the model information at each checkpoint,
normally at the end of each training epoch of a trial. The checkpointed 
model files during training are saved under folders named like:
  
  ::
    
    # running trials without Ray (default)
    ./results/your_experiment_name/Train_xxx...xxx/checkpoint_x/
              - model
              - model.config.pkl
              - model.fdiri_cal.pkl

    # running trials using Ray 
    ./ray_results/your_experiment_name/Train_xxx...xxx/checkpoint_x/
              - model
              - model.config.pkl
              - model.fdiri_cal.pkl

In the above folder, the 'model' file contains the learned model
parameters. The 'model.config.pkl' file contains configured
hyperparameters of the model. The 'model.fdiri\_cal.pkl' file (if
exists) contains the calibration model learned with validation data,
which can be used for calibrating predicted mutation rates. These
files can be used in downstream analyses such as model prediction and
transfer learning. The 'progress.csv' files in 'Train\_xxx' folders
contain important information for each training epoch of trials
(e.g., validation loss, used time, etc.). You can use the command
``get_best_mural_models`` to find the best model per trial after
training.

  ::

    # running trials without Ray (default)
    get_best_mural_models ./results/your_experiment_name/Train_*/progress.csv

    # running trials using Ray 
    get_best_mural_models ./ray_results/your_experiment_name/Train_*/progress.csv

* Example 1

The following command will train a model by running two trials (default:``--n_trials 2``),
using data in 'data/training.sorted.bed' for training. The training
results will be saved under the folder './ray\_results/example1/'.
Default values will be used for other unspecified arguments. Note
that, by default, 10% of the sites sampled from 'training.sorted.bed'
is used as validation data (i.e. ``--valid_ratio 0.1``). You can run
this example under the 'examples/' folder in the package.

::
  
   # serially running two trials (default)
   mural_train --ref_genome data/seq.fa --train_data data/training.sorted.bed \
               --experiment_name example1 > test1.out 2> test1.err

   # parallel running two trials using Ray 
   mural_train --ref_genome data/seq.fa --train_data data/training.sorted.bed \
               --use_ray --experiment_name example1 > test1.out 2> test1.err
   
.. note::

  If your machine has sufficient resources to execute multiple trials in parallel, 
  it is recommended to add the ``--use_ray`` option. Using Ray allows for better resource 
  scheduling. If executing multiple trials serially or running only a single trial (set ``--n_trials 1``), 
  it is recommended not to use ``--use_ray``, which can improve the runtime speed by approximately 
  2-3 times for each trial.

* Example 2

The following command will use data in 'data/training.sorted.bed'
as training data and a separate 'data/validation.sorted.bed' as
validation data. The option ``--local_radius 7`` means that length of
the local sequence used for training is 7\*2+1 = 15 bp.
``--distal_radius 200`` means that length of the expanded sequence
used for training is 200\*2+1 = 401 bp. You can run this example
under the 'examples/' folder in the package.

::

  mural_train --ref_genome data/seq.fa \
              --train_data data/training.sorted.bed \
              --validation_data data/validation.sorted.bed \
              --n_trials 2 --local_radius 7 \
              --distal_radius 200 --experiment_name example2 \
              > test2.out 2> test2.err

* Example 3

If the length of the expanded sequence used for training is large (``distal_radius`` 
larger than 1000), data loading becomes a bottleneck in the training process. You can 
set the option ``--cpu_per_trial`` to specify how many CPUs each trial. The following 
command uses 3 extra CPUs to accelerate data loading. You can run this example 
under the 'examples/' folder in the package.

::
  
   mural_train --ref_genome data/seq.fa \
               --train_data data/training.sorted.bed \
               --cpu_per_trial 4 \
               --experiment_name example3 > test3.out 2> test3.err

* Example 4

If RAM memory or GPU memory limits the usage of ``mural_train`` (which may happen with large 
expanded sequences used for training), please refer to following suggestions.

Since v1.1.2, we split the genomes into segments of a spefic size to facilitate data preprocessing (see illustration in the panel A of the figure below). 

To reduce RAM memory, you can set smaller values for ``--segment_center`` and ``--sampled_segments``. 
The default value of ``--segment_center`` is 300,000 bp, meaning maximum encoding unit of the 
sequence is 300000+2*distal_radius bp (see illustration in the panel A of the figure below). It is the key parameter 
for trade-off between RAM memory usage and data preprocessing speed. You can reduce this 
(e.g., 50,000 bp) at the cost of an acceptable loss in data preprocessing speed. 
The second option is to set smaller value for ``--sampled_segments`` (default 10). If changing this, you should also 
check the performance of trained model, because ``--sampled_segments`` may also influnce model performance 
sometimes. The impact of the two parameters on RAM usage can be visualized in the following figure (panels B & C):

.. image:: images/preprocessAndRAM_memory_usage.jpg 
   :width: 830px

To reduce GPU memory, it is recommended to reduce ``--batch_size`` (default 128). You can set smaller values (e.g., 64, 32, 16, etc.).
In addition, you may also consider using smaller models (e.g., reducing ``--distal_radius``, ``--CNN_out_channels``, etc.), but that may affect model performance.

You can run the following commands under the 'examples/' folder in the package.

::
  
   # use less RAM memory
   mural_train --ref_genome data/seq.fa \
              --train_data data/training.sorted.bed \
              --validation_data data/validation.sorted.bed \
              --n_trials 2 --local_radius 7 \
              --distal_radius 64000 --segment_center 100000 \
              --sampled_segments 4 --experiment_name example4 \
              > test4.out 2> test4.err

   # use less GPU memory
   mural_train --ref_genome data/seq.fa \
              --train_data data/training.sorted.bed \
              --validation_data data/validation.sorted.bed \
              --n_trials 2 --local_radius 7 \
              --batch_size 64
              --distal_radius 64000 --experiment_name example4 \
              > test4.out 2> test4.err

.. note::

  The RAM memory usage is approximately proportional to 
  ``sampled_segments * segment_center * 4 * (2 * distal_radius + 1) * 4 / 2^30 + C`` (GB), 
  where ``C`` is a constant term ranging between 5 and 12 GB. If insufficient RAM memory, 
  using this formula to estimate RAM usage might help in finding proper parameters.

Model prediction
~~~~~~~~~~~~~~~~

``mural_predict`` predicts mutation rates for all sites in a BED file
based on a trained model. 

* Input data

The required input files for prediction include the reference FASTA
file, a BED-formated data file and a trained model. The BED file is
organized in the same way as that for training. The 5th column can be
set to '0' if no observed mutations for the sites in the prediction BED.
The model-related files for input are 'model' and 'model.config.pkl',
which are generated at the training step. The file
'model.fdiri\_cal.pkl', which is for calibrating predicted mutation
rates, is optional. 

* Output data

The output of ``mural_predict`` is a tab-separated file containing
the sequence coordinates (BED-formatted) and the predicted probabilities
for all possible mutation types. Usually, the 'prob0' column contains
probabilities for the non-mutated class and other 'probX' columns for
mutated classes. Some example lines of a prediction output file are
shown below.

::

    chrom   start   end    strand mut_type  prob0   prob1   prob2   prob3
    chr1    10006   10007   -       0       0.9797  0.003134 0.01444 0.002724
    chr1    10007   10008   +       0       0.9849  0.005517 0.00707 0.002520
    chr1    10008   10009   +       0       0.9817  0.004801 0.01006 0.003399
    chr1    10012   10013   -       0       0.9711  0.004898 0.02029 0.003746

* Example 5

The following command will predict mutation rates for all sites in
'data/testing.bed.gz' using model files under the
'models/checkpoint\_6/' folder and save prediction results into
'testing.ckpt6.fdiri.tsv.gz'. You can run this example under the
'examples/' folder in the package.

::

   mural_predict --ref_genome data/seq.fa --test_data data/testing.bed.gz \
                 --model_path models/checkpoint_6/model \
                 --model_config_path models/checkpoint_6/model.config.pkl \
                 --calibrator_path models/checkpoint_6/model.fdiri_cal.pkl \
                 --pred_file testing.ckpt6.fdiri.tsv.gz \
                 --cpu_only > test5.out 2> test5.err

Transfer learning
~~~~~~~~~~~~~~~~~

``mural_train_TL`` trains MuRaL models like ``mural_train`` but
initializes model parameters with learned weights from a pre-trained
model. Its training results are also saved under the './results/' or './ray\_results/'
folder. 

* Input data

The input files for ``mural_train_TL`` include the reference FASTA
file (required), a training data file (required), a validation data file
(optional), and model-related files of a trained model (required). The
required model-related files are 'model' and 'model.config.pkl' under a
specific checkpoint folder, normally generated by ``mural_train`` or
``mural_train_TL``. 

* Output data

Output data has the same structure as that of ``mural_train``.

* Example 6

The following command will train a transfer learning model using
training data in 'data/training\_TL.sorted.bed', the validation data
in 'data/validation.sorted.bed', and the model files under
'models/checkpoint\_6/'. You can run this example under the
'examples/' folder in the package.

::

 mural_train_TL --ref_genome data/seq.fa \
                --train_data data/training_TL.sorted.bed \
                --validation_data data/validation.sorted.bed \
                --model_path models/checkpoint_6/model \
                --model_config_path models/checkpoint_6/model.config.pkl \
                --train_all --init_fc_with_pretrained \
                --experiment_name example6 > test6.out 2> test6.err


Calculating k-mer and regional correlations for evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For model evaluation, since it is impossible to evaluate the accuracy 
of predicted mutation rates at the single-nucleotide level, we employ 
two metrics, k-mer correlation and regional correlation, to evaluate 
model performance at the higher (summarized) levels. More details about 
the two metrics can be found in the MuRaL paper. The k-mer and regional 
correlations can be calculated with the predicted tsv files generated 
by ``mural_predict``.

K-mer correlation analysis
..........................

The tool ``calc_kmer_corr`` is used for calculating k-mer correlations.

* Input data

The inputs for k-mer correlation analysis include the reference
FASTA file, a prediction tsv file and the length of k-mer. Note that for 
evaluation, we need to provide a specific set of observed mutations 
(e.g. all available rare variants), which are stored in the 5th column of 
the prediction tsv files. These observed mutations are used for
calculating observed mutation rates. We can change the content in the 5th 
column to evaluate model performance in different observed mutation data.
   
* Output data

The outputs include a file ('\*-mer.mut\_rates.tsv') storing predicted and 
observed k-mer rates of all possible mutation subtypes, and a file ('\*-mer.corr.txt')
storing the k-mer correlations (Pearson's r and p-value) of three mutation
types in a specific order (e.g., for A/T sites, prob1, prob2 and prob3 are
for A>C, A>G and A>T, respectively).

::

 # example of '*-mer.mut_rates.tsv'
 type	avg_obs_rate1	avg_obs_rate2	avg_obs_rate3	avg_pred_prob1	avg_pred_prob2	avg_pred_prob3	number_of_mut1	number_of_mut2	number_of_mut3	number_of_all
 TAG	0.006806776385512125	0.010141979926438501	0.012039461380213204	0.012744358544122413	0.01817057941563919	0.021860978496512425	3494	5206	6180	513312
 TAA	0.007517292690907348	0.011278023120833133	0.01318808653952362	0.013600087566977897	0.019697007577734515	0.024266536859123104	7214	10823	12656	959654
 AAA	0.0068964404639771226	0.010705555691654661	0.009617493130148654	0.012599749576515839	0.020442895433664586	0.01646869397956817	11542	17917	16096	1673617
 
 # example of '*-mer.corr.txt'
 3-mer	prob1	0.9569216831654604	6.585788162834682e-09 # r and p for prob1
 3-mer	prob2	0.9326211281771537	1.4129640985193586e-07 # r and p for prob2
 3-mer	prob3	0.947146892265788	2.6848989196451608e-08 # r and p for prob3


* Example 7 

The following commands use the prediction file 'testing.ckpt4.fdiri.tsv.gz' 
to calculate 3-mer, 5-mer and 7-mer correlations:

::

 calc_kmer_corr --pred_file testing.ckpt4.fdiri.tsv.gz --ref_genome data/seq.fa --kmer_length 3 --out_prefix test
 calc_kmer_corr --pred_file testing.ckpt4.fdiri.tsv.gz --ref_genome data/seq.fa --kmer_length 5 --out_prefix test
 calc_kmer_corr --pred_file testing.ckpt4.fdiri.tsv.gz --ref_genome data/seq.fa --kmer_length 7 --out_prefix test

Regional correlation analysis
.............................

The tool ``calc_region_corr`` is used for calculating regional correlations.

* Input data

The inputs for regional correlation analysis include a prediction tsv 
file and the window size. Like the k-mer correlation analysis, we need to 
provide a specific set of observed mutations in the 5th column of the prediction
tsv files. These observed mutations are used for calculating observed 
regional mutation rates. 

* Output data

There are multiple output files. The files storing regional rates 
('\*.regional\_rates.tsv') have seven columns: chromosome name, the end
position of the window, number of valid sites in the window, number of 
observed mutations in the window, average observed mutation rate, average 
predicted mutation rate in the window and the 'used_or_deprecated' label. 
The windows labeled 'deprecated' are not used in correlation analysis due 
to too few valid sites. The regional correlation (Pearson's r and p-value)
of the considered mutation type is given in the '\*.corr.txt'.

::

 # example of '*.regional_rates.tsv'
 chrom	end	sites_count	mut_type_total	mut_type_avg	avg_pred	used_or_deprecated
 chr3	100000	61492	576	0.009367072139465296	0.020374342255903233	used
 chr3	200000	60680	531	0.008750823994726434	0.02025859070533955	used
 chr3	300000	59005	499	0.00845691043131938	0.01882644280993153	used
 ...
 
 # example of '*.corr.txt'
 100Kb	prob3	0.4999	6.040983e-16 


* Example 8

The following command will calculate the regional correlation for 100Kb windows and 
'prob2' mutation type. 

::

 calc_regional_corr --pred_file testing.ckpt4.fdiri.tsv.gz --window 100000 --model prob2 --out_prefix test_region_corr

Visualization of correlation results
....................................

You can run the commands like below to extract k-mer correlations and corresponding 
p-values for further visualization:

::

 cat test.{3,5,7}-mer.corr.txt | awk 'BEGIN{print "k-mer\tmut_type\tcorrelation\tp-value"}{print;}' > kmer_correlations.tsv

The resulting 'kmer_correlations.tsv' file is tab-delimited, looking like:

::

 k-mer	mut_type	correlation	p-value
 3-mer	A>C			0.8527		2.7049e-05
 3-mer	A>G			0.8453		3.7235e-05
 ...

The following python code can be used for generating bar plots for k-mer 
correlations:

::

 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns

 df = pd.read_table('kmer_correlations.tsv')
 plt.figure(figsize=(6,4))
 sns.catplot(x="mut_type", y="correlation", kind="bar", hue="k-mer", data=df, palette="Blues_r")
 plt.title('Bar plots of k-mer correlations')
 plt.savefig('kmer_correlations.jpg', bbox_inches='tight')

The plot looks like below:

.. image:: images/kmer_correlations.jpg

Similarly, one can generate bar plots for regional correlations for 
evaluation.

In addition, based on the output of ``calc_region_corr`` above, we can 
visualize how predicted rates fit observed rates for windows across 
a chromosome or a specific region. First, we should standardize the 
observed rates and the predicted rates for all windows by using z-score 
transformation. Then we select some regions to generate the plots. Below 
we use the results for 100Kb windows and A>G mutation type, and the region 
selected is from 15Mb to 23.6Mb. The solid line indicates average 
predicted mutation rates and the shade for average observed mutation 
rates:

::

 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns
 from sklearn import preprocessing
 from scipy.stats import pearsonr

 df = pd.read_table('test2.100Kb.prob2.regional_rates.tsv')
 df = df[df['used_or_deprecated'] == 'used']

 #z-score preprocessing
 avg_obs = preprocessing.scale(df['avg_obs'])
 avg_pred = preprocessing.scale(df['avg_pred'])
 data = {'window_end':df['window_end'],'avg_obs':list(avg_obs),'avg_pred':list(avg_pred)}
 df1 = pd.DataFrame(data)

 #select the region
 df2 = df1[143:229]
 corr = pearsonr(df2['avg_obs'],df2['avg_pred'])
 print("Correlation of the selected regions is %f, p-value is %f" %(corr[0],corr[1]))

 #plot
 fig, ax = plt.subplots(1, figsize=(10, 2))
 ax.set_xlabel("Chr3(Mb)")
 ax.fill_between(df2['window_end']/1000000,df2['avg_obs'], alpha=0.3, color = 'Grey')
 ax.plot(df2['window_end']/1000000,df2['avg_pred'], label="avg_pred", linewidth = 1.5)
 plt.ylabel('average mutation rate (Z-score)')
 
 plt.savefig('regional_rates.jpg', bbox_inches = 'tight')

The plot looks like below:

.. image:: images/regional_rates.jpg

Scaling MuRaL-predicted mutation rates to per base per generation rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The raw MuRaL-predicted mutation rates are not mutation rates per bp per
generation. To obtain a mutation rate per bp per generation for each
nucleotide, one can scale the MuRaL-predicted rates using reported
genome-wide de novo mutation rate and spectrum per generation. First, use
the command ``calc_mu_scaling_factor`` to calculate scaling factors for
specific groups of sites (e.g. A/T sites, C/G sites). Then use the
scaling factors to scale mutation rates in prediction files via the
command ``scale_mu``.

Note that we cannot directly compare or add up raw predicted rates from
different MuRaL models (e.g. A/T model and C/G model), but we can do
that with scaled mutation rates. The accuray of genome-wide mutation rate
per generation does not affect within-genome comparison but can affect
between-species comparison. Whether to do or not do scaling does not affect
the calculation of k-mer and regional mutation rate correlations.

* Example 9

Here is an example for scaling mutation rates for A/T sites. Suppose that we 
have the following proportions of different mutation types and proportions
of different site groups in a genome. In addition, suppose we already know from 
previous research that the genome-wide mutation rate per generation of the 
species is 5e−9 per base per generation. If the per generation mutation rate 
is not available for the studied species, one may use the estimate from a 
closely related species.

::

 #mutation_type  proportion
 AT_mutations       0.355
 nonCpG_mutations   0.423
 CpG_mutations      0.222
 
 #site_group    proportion
 AT_sites           0.475
 nonCpG_sites       0.391
 CpG_sites          0.134

To calculate the scaling factor, we need to have the predicted mutation rates for
a set of representative sites based on a trained model. It is recommended to use
the validation sites at the training step whose size is relatively small and 
representative enough. For instance, the following command is for obtaining 
predicted mutation rates for validation sites of the A/T model.

::
 
 mural_predict --ref_genome data/seq.fa --test_data data/AT_validation.sorted.bed --model_path 
 models/checkpoint_6/model --model_config_path models/checkpoint_6/model.config.pkl  --calibrator_path  
 models/checkpoint_6/model.fdiri_cal.pkl --pred_file AT_validation.ckpt6.fdiri.tsv.gz --without_h5 --cpu_only > 
 test.out 2> test.err

Next, the command ``calc_mu_scaling_factor`` will be used to compute the scaling
factor based on the predicted rates, the proportions of A/T mutation types and 
proportions of A/T sites in the genome, and the genome-wide per generation mutation
rate.

:: 

 calc_mu_scaling_factor --pred_files AT_validation.ckpt6.fdiri.tsv.gz --genomewide_mu 5e-9 
 --m_proportions 0.355 --g_proportions 0.475 > scaling_factor.out
 
 # Output file 'scaling_factor.out' may look like the following:
 pred_file: AT_validation.ckpt6.fdiri.tsv.gz
 genomewide_mu: 5e-09
 n_sites: 84000
 g_proportion: 0.475
 m_proportion: 0.355
 prob_sum: 4.000e+03
 scaling factor: 7.848e-08
 
Finally, the obtained scaling factor ``7.848e-08`` is used to scale all the 
predicted rates of all A/T sites using ``scale_mu``. You can run  ``scale_mu`` 
separately for each chromosome.

::
 
 scale_mu --pred_file AT_chr1.tsv.gz --scale_factor 7.848e-08 --out_file AT_chr1.scaled.tsv.gz

Similarly, you can generate the scaled mutation rates for non-CpG and CpG sites like
the above example. More details can be found in the MuRaL paper.

Trained models and predicted mutation rate maps of multiple species
-----------------------------------------------------------------------

Trained models (by MuRaL v1.0.0) for four species - *Homo sapiens*, *Macaca mulatta*, 
*Arabidopsis thaliana* and *Drosophila melanogaster* are provided in 
the 'models/' folder of the package. One can use these model files 
for prediction or transfer learning.

Predicted single-nucleotide mutation rate maps (by MuRaL v1.0.0) for these genomes are
available at `ScienceDB <https://www.doi.org/10.11922/sciencedb.01173>`__.

Citation
--------

Fang Y, Deng S, Li C. A generalizable deep learning framework for inferring 
fine-scale germline mutation rate maps. *Nature Machine Intelligence* (2022)
`doi:10.1038/s42256-022-00574-5 <https://doi.org/10.1038/s42256-022-00574-5>`__

Contact
-------

For reporting issues or requests related to the package, please use GitHub Issues
or write to mural-project@outlook.com.
