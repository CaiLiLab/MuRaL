# Example 1 (training): 
# The following command will train a model by running two trials, using data in 'data/training.sorted.bed' for training. The training results will be saved under the folder './ray_results/example1/'. Default values will be used for other unspecified arguments. Note that, by default, 10% of the sites sampled from 'training.sorted.bed' is used as validation data (i.e. '--valid_ratio 0.1').
mural_train --ref_genome data/seq.fa --train_data data/training.sorted.bed --experiment_name example1 > test1.out 2> test1.err

# after training, one can run the following command to get the best model per trial
get_best_mural_models ./ray_results/example1/Train_*/progress.csv

# Example 2 (training):
# The following command will use data in 'training.sorted.bed' as training data and a separate 'validation.sorted.bed' as validation data. The option '--local_radius 7' means that length of the local sequence used for training is 7*2+1 = 15 bp. '--distal_radius 200' means that length of the expanded sequence used for training is 200*2+1 = 401 bp. '--n_trials 3' means that three trials will be run.
mural_train --ref_genome data/seq.fa --train_data data/training.sorted.bed --validation_data data/validation.sorted.bed  --local_radius 7 --distal_radius 200 --n_trials 3 --experiment_name example2 > test2.out 2> test2.err

# Example 3 (prediction): 
#The following command will predict mutation rates for all sites in 'data/testing.bed.gz' using model files under the 'checkpoint_6/' folder and save prediction results into 'testing.ckpt6.fdiri.tsv.gz'.
mural_predict --ref_genome data/seq.fa --test_data data/testing.bed.gz --model_path models/checkpoint_6/model --model_config_path models/checkpoint_6/model.config.pkl  --calibrator_path models/checkpoint_6/model.fdiri_cal.pkl --pred_file testing.ckpt6.fdiri.tsv.gz --without_h5 --cpu_only > test3.out 2> test3.err

# Example 4 (transfer learning):
# The following command will train a transfer learning model using training data in 'data/training_TL.sorted.bed', the validation data in 'data/validation.sorted.bed', and the model files under 'models/checkpoint_6/'.
mural_train_TL --ref_genome data/seq.fa --train_data data/training_TL.sorted.bed --validation_data data/validation.sorted.bed --model_path models/checkpoint_6/model --model_config_path models/checkpoint_6/model.config.pkl --train_all  --init_fc_with_pretrained --experiment_name example4 > test4.out 2> test4.err
