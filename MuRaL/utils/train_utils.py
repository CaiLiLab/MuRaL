import random
import string
import os
import sys
import re

import multiprocessing
from multiprocessing import Process

def get_trialid(count, local_dir):
    global unique_id
    if count == 0:
        unique_id = generate_unique_id()
    trial_id = generate_trialid(unique_id, count)
    
    if check_trial_id(trial_id, local_dir):
        return trial_id
    else:
        return get_trialid(count, local_dir)


def generate_unique_id(unique_length=5):
    unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=unique_length))
    return unique_id

def generate_trialid(unique_id, count, prefix="Train"):
    # Generate a random string of specified length
    count_str = f"{count:05d}"
    trial_id = f"{prefix}_{unique_id}_{count_str}"
    return trial_id
    

def check_trial_id(trial_id, local_dir):
    if trial_id in os.listdir(local_dir):
        return 0
    else:
        return 1
    
def wrap_args(args, trial_dir):
    # output dir of trial
    args.trial_dir = trial_dir
    # change tmp_out_file name
    args.trial_training_log = os.path.join(trial_dir, 'stdlog')
    args.progress_log = os.path.join(trial_dir, 'progress.csv')
    return args

def run_standalong_training(train, n_trials, config, args, para=False):
    
    experiment_dir = args.experiment_name
    local_dir = os.path.join('./results', experiment_dir)
    trial_dir_list = [] 
    # to do: memory control
    if para:
        multiprocessing.set_start_method('spawn')
        processes = []
        for count in range(n_trials):
            trial_id = get_trialid(count, local_dir)
            trial_dir = os.path.join(local_dir, trial_id)
            os.makedirs(trial_dir, exist_ok=True)
            trial_dir_list.append(trial_dir)
            args = wrap_args(args, trial_dir)
        
            p = Process(target=train, args=(config, args))
            processes.append(p)
            p.start()
    
        for p in processes:
            p.join()
    else:
        for count in range(n_trials):
            trial_id = get_trialid(count, local_dir)
            trial_dir = os.path.join(local_dir, trial_id)
            os.makedirs(trial_dir, exist_ok=True)
            trial_dir_list.append(trial_dir)
            args = wrap_args(args, trial_dir)

            train(config, args)
    
    # out best model in n_trials
    output_best_mural_model(trial_dir_list, args.tmp_log_file)

    return 0

def output_best_mural_model(trial_dir_list, file):
    """
    Find best model form mult trials
    """
    best_loss = float('inf')
    best_checkpoint = None
    
    for trial_dir in trial_dir_list:
        trial_checkpoint, trial_loss = get_best_model_from_trial(trial_dir)
        if trial_loss < best_loss:
            best_loss = trial_loss
            best_checkpoint = trial_checkpoint

    with open(file, 'a+') as f:
        f.write('\t'.join(['best model: ', best_checkpoint, str(best_loss)]) + '\n')   
    #return best_checkpoint, best_loss


def get_best_model_from_trial(trial_dir, minor='loss'):
    """
    Find best model from given trials
    """
    best_loss = float('inf')
    best_checkpoint  = None
    
    checkpoint_prefix = 'check_point'
    checkpoint_num = max([extract_number_from_checkpoint(name) \
                          for name in os.listdir(trial_dir) if name.startswith(checkpoint_prefix)])
    progress_message = ''
    for num in range(checkpoint_num + 1):
        checkpoint_dir = checkpoint_prefix + str(num)
        checkpoint_dir = os.path.join(trial_dir, checkpoint_dir)
        # check 
        if not os.path.isdir(checkpoint_dir):
            sys.exit(f"Error: {checkpoint_dir} should be dir!")
    
        try:
            metric_path = find_metric(checkpoint_dir)
        except FileNotFoundError:
            continue
        
        loss_checkpoint, fdiri_loss_checkpoint = get_loss_from_metric(metric_path)
        
        # save loss
        progress_message += '\t'.join([str(loss_checkpoint), str(fdiri_loss_checkpoint)]) + '\n'

        if minor == 'loss':
            current_loss = loss_checkpoint
        elif minor == 'fdiri_loss':
            current_loss = fdiri_loss_checkpoint
        else:
            sys.exit(f"Error: minor should be 'loss' or 'fdiri_loss', but input {minor}")

        if current_loss < best_loss:
            best_loss = current_loss
            best_checkpoint = checkpoint_dir
    
    # output loss
    progress_log = os.path.join(trial_dir, 'progress.csv')            
    trial_progress_out(progress_message, progress_log)

    return best_checkpoint, best_loss

def extract_number_from_checkpoint(checkpoint_name):
    match = re.search(r'\d+', checkpoint_name)
    if match:
        return int(match.group())
    sys.exit(f"Error: checkpoint_name should be check_pointx, but input {checkpoint_name}!")

def find_metric(checkpoint_dir):
    for name in os.listdir(checkpoint_dir):
        if name.endswith('metrics.txt'):
            metric_abs_path = os.path.join(checkpoint_dir, name)
            return metric_abs_path

    print(f'Warning: metrics file not in {checkpoint_dir}, check if training is finished!')
    raise FileNotFoundError(f'Metrics file not found in {checkpoint_dir}')

def get_loss_from_metric(metric_path):
    with open(metric_path) as f:
        for line in f:
            if line.startswith('loss'):
                loss_checkpoint = float(line.strip().split(':')[-1])
            if line.startswith('fdiri_loss'):
                fdiri_loss_checkpoint = float(line.strip().split(':')[-1])
        return loss_checkpoint, fdiri_loss_checkpoint

def trial_progress_out(progress_message, progress_log):
    with open(progress_log, 'w') as f:
        f.write(progress_message)
    return 0
