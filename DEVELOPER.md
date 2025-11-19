# mural_indel/mural_snv train

##  func: use h5 output distal encoding

`--with_h5` :
Enables HDF5-based distal encoding output to accelerate data loading in specific sparse mutation scenarios.
When variant extensions in upstream/downstream regions (distal_radius) show minimal overlap, using this parameter accelerates data preprocessing. For example, it provides faster processing in human 1in2000 dataset with less than 4k distal_radius values, while potentially slowing down processing with larger distal_radius values.

`--h5f_path` :
Specify the folder to generate HDF5. Default: Folder containing the BED file.

`--n_h5_files` :
Number of HDF5 files for each BED file. When the BED file has many positions and the distal radius is large, increasing the value for `--n_h5_files` files can reduce the time for generating HDF5 files. Default: 1.

`--custom_dataloader` :
Use a custom data loader. This data loader is not supported parallelizing
data loading. For '--cpu-per-trial 1' and without HD5, the speed of loading
data is faster than default dataloader. Default: False.

`--poisson_calib`:
    Enable Poisson-based calibration.

    For indel models:
        - Calibration is automatically enabled unless explicitly disabled.

    For SNV models:
        - Calibration is disabled by default.
        - Use this flag to enable it.

    Default: False

# mural_indel/mural_snv predict

##  func: use h5 output distal encoding

`--with_h5` :
same as `train` command

`--h5f_path` :
same as `train` command

`--n_h5_files` :
same as `train` command

`--poisson_calib`:
same as `train` command

# mural_indel/mural_snv calc_scaling_factor

##  func: scale after calc_scaling_factor

`--do_scaling` :
Save scaled mutation rates for input pred files. Default: False.

# mural_indel evaluate

`--motif_only` :
Evaluate motif correlation(specify motif length by --motif_length) for indel. Default: False.

`motif_length` :
Length of motif used for evaluation. Default: 6.




