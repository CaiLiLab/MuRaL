import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


from pybedtools import BedTool

import sys
import argparse
import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data import random_split

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import os
import time
import datetime
import random
import inspect

from utils.printer_utils import get_printer
from model.nn_utils import *
from evaluation.evaluation import *
from data.custom_dataloader import MyDataLoader
from data.preprocessing import *
from model.UNet_models import UNet_Small
from model.nn_models import *



#from torchsampler import ImbalancedDatasetSampler
def train(config, args, model_type, checkpoint_dir=None):
    """
    Training funtion.
    
    Args:
        config: configuration of hyperparameters
        args: input args from the command line
        checkpoint_dir: checkpoint dir
    """
    
    # Ensure output can be viewed in real-time in a distributed environment
    if args.use_ray:
        print = get_printer(args.use_ray, None)
    else:
        print = get_printer(args.use_ray, args.trial_training_log)
        trial_dir = args.trial_dir


    print("torch._C._cuda_getDeviceCount():", torch._C._cuda_getDeviceCount())
    print("torch.cuda.device_count(): ", torch.cuda.device_count())

    # Get parameters from the command line
    train_file = args.train_data # Ray requires absolute paths
    valid_file = args.validation_data
    ref_genome= args.ref_genome
    n_h5_files = args.n_h5_files

    local_radius = args.local_radius
    local_order = args.local_order
    distal_radius = args.distal_radius  
    distal_order = args.distal_order
    batch_size = args.batch_size
    ####
    sample_weights = args.sample_weights
    #ImbSampler = args.ImbSampler
    local_dropout = args.local_dropout
    CNN_kernel_size = args.CNN_kernel_size   
    CNN_out_channels = args.CNN_out_channels
    distal_fc_dropout = args.distal_fc_dropout
    model_no = args.model_no   
    #pred_file = args.pred_file   
    optim = args.optim
    learning_rate = args.learning_rate   
    weight_decay = args.weight_decay
    weight_decay_auto = args.weight_decay_auto
    LR_gamma = args.LR_gamma 
    restart_lr = args.restart_lr
    min_lr = args.min_lr
    epochs = args.epochs
    n_class = args.n_class  
    cuda_id = args.cuda_id
    valid_ratio = args.valid_ratio
    seq_only = args.seq_only
    cudnn_benchmark_false = args.cudnn_benchmark_false
    with_h5 = args.with_h5
    split_seed = args.split_seed
    gpu_per_trial = args.gpu_per_trial
    cpu_per_trial = args.cpu_per_trial
    save_valid_preds = args.save_valid_preds
    grace_period = args.grace_period

    # unet specify para
    if model_type == 'indel':
        if not hasattr(args, 'down_list'):
            sys.exit('Error: The indel model requires the --down-list parameter. '
                    'Please provide a down_list file path.')
        config['down_list'] = down_list = args.down_list

    bw_paths = args.bw_paths
    without_bw_distal = args.without_bw_distal
    bw_files = []
    bw_names = []
    bw_radii = []
    h5f_path=args.h5f_path
    use_ray = args.use_ray
    custom_dataloader = args.custom_dataloader
    segment_workers = cpu_per_trial - 1
    print("Extral cpu used dataloadr: ", segment_workers)


    start_time = time.time()

    if cudnn_benchmark_false:
        torch.backends.cudnn.benchmark = False
        print('NOTE: setting torch.backends.cudnn.benchmark = False')
    
    if bw_paths:
        try:
            bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
            bw_files = list(bw_list[0])
            bw_names = list(bw_list[1])
            if bw_list.shape[1]>2:
                bw_radii = list(bw_list[2].astype(int))
            else:
                bw_radii = [config['local_radius']]*len(bw_files)
            
            print("bw_radii:", bw_radii)
        except pd.errors.EmptyDataError:
            print('Warnings: no bigWig files provided in', bw_paths)
    else:
        print('NOTE: no bigWig files provided.')

    # Read BED files
    train_bed = BedTool(train_file)
    
    if not with_h5:
        print('using numpy/pandas for distal_seq ...')
        step_stime = time.time()
        dataset = prepare_dataset_np(train_bed, ref_genome, bw_files, bw_names, bw_radii, \
                                     config['segment_center'], config['local_radius'], config['local_order'], \
                                        config['distal_radius'], distal_order, seq_only=seq_only, model_type=model_type)
        if not dataset.distal_info:
            dataset.get_distal_encoding_infomation()
        print("training set preprocess without H5 used time:", (time.time() - step_stime))
    else:
        step_stime = time.time()
        dataset = prepare_dataset_h5(train_bed, ref_genome, bw_paths, bw_files, bw_names, bw_radii, \
                                     config['segment_center'], config['local_radius'], config['local_order'], \
                                        config['distal_radius'], distal_order, h5f_path=h5f_path, chunk_size=5000, \
                                            seq_only=seq_only, n_h5_files=n_h5_files, without_bw_distal=without_bw_distal, model_type=model_type)
        print("training set preprocess with H5 used time:", time.time() - step_stime)

    data_local = dataset.data_local
    categorical_features = dataset.cat_cols
    n_cont = len(dataset.cont_cols)
    
    #config['n_cont'] = n_cont
    config['n_class'] = n_class
    config['model_no'] = model_no
    #config['bw_paths'] = bw_paths
    config['without_bw_distal'] = without_bw_distal
    config['seq_only'] = seq_only
    config['restart_lr'] = restart_lr
    config['min_lr'] = min_lr
    #print('n_cont: ', n_cont)
    
    ################
    if valid_file:
        print('using given validation file:', valid_file)
        valid_bed = BedTool(valid_file)
        if not with_h5:
            step_time = time.time()
            dataset_valid = prepare_dataset_np(valid_bed, ref_genome, bw_files, bw_names, bw_radii, \
                                               config['segment_center'], config['local_radius'], config['local_order'], config['distal_radius'], distal_order, seq_only=seq_only, model_type=model_type)
            if not dataset_valid.distal_info:
                dataset_valid.get_distal_encoding_infomation()
            print("validation set preprocess time without H5 used time:", (time.time() - step_time))
        else:
            step_stime = time.time()
            dataset_valid = prepare_dataset_h5(valid_bed, ref_genome, bw_paths, bw_files, bw_names, bw_radii, \
                                         config['segment_center'], config['local_radius'], config['local_order'], \
                                            config['distal_radius'], distal_order, h5f_path=h5f_path, chunk_size=5000, \
                                                seq_only=seq_only, n_h5_files=n_h5_files, without_bw_distal=without_bw_distal, model_type=model_type)
            print("validation set preprocess with H5 used time:", time.time() - step_stime)

    
        data_local_valid = dataset_valid.data_local.reset_index(drop=True)
        
    ################
    
    device = torch.device('cpu')
    if gpu_per_trial > 0:
        # assert CUDA is available
        if not torch.cuda.is_available():
            print('Warning: You requested GPU computing, but CUDA is not available! If you want to run without GPU, please set "--ray_ngpus 0 --gpu_per_trial 0"', file=sys.stderr)
        # set the device
        if not use_ray:
            device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
            torch.cuda.set_device(f'cuda:{cuda_id}')
        else:    
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #valid_size = int(len(dataset)*valid_ratio)
    #train_size = len(dataset) - valid_size
    #print('train_size, valid_size:', train_size, valid_size)

    if not valid_file:
        # split by segment
        valid_segment_size = int(len(dataset)*valid_ratio)
        train_segment_size = len(dataset) - valid_segment_size
    
        # Split the data into two parts - training data and validation data
        dataset_train, dataset_valid = random_split(dataset, [train_segment_size, valid_segment_size], torch.Generator().manual_seed(split_seed))
        dataset_valid.indices.sort()
        data_local = dataset.data_local.loc[dataset_train.indices, ].reset_index(drop=True)
        data_local_valid = dataset.data_local.loc[dataset_valid.indices, ].reset_index(drop=True)
    else:
        dataset_train = dataset
    train_size = len(data_local)
    valid_size = len(data_local_valid)
    
    print('train_size, valid_size:', train_size, valid_size)
    # Dataloader for training
    #if not ImbSampler: 
    if sample_weights:
        print("Warning: sample_weights be dropped, the program will run with sample_weights=None!")
    if custom_dataloader:
        # Dataloader for train and validation
        dataloader_train = MyDataLoader(dataset_train, config['sampled_segments'], config['batch_size'], shuffle=True, shuffle2=True, num_workers=0, pin_memory=False)
        dataloader_valid = MyDataLoader(dataset_valid, config['sampled_segments'], config['batch_size'], shuffle=False, shuffle2=False, num_workers=0, pin_memory=False)
    else:
        segmentLoader_train = DataLoader(dataset_train, 1, shuffle=True, num_workers=segment_workers, pin_memory=False)
        dataloader_train = generate_data_batches(segmentLoader_train, config['sampled_segments'], config['batch_size'], shuffle=True)
        
        segmentLoader_valid = DataLoader(dataset_valid, 1, shuffle=False, num_workers=segment_workers, pin_memory=False)
        dataloader_valid = generate_data_batches(segmentLoader_valid, config['sampled_segments'], config['batch_size'], shuffle=False)

    if config['transfer_learning']:
        emb_dims = config['emb_dims']
    else:
        # Number of categorical features
        cat_dims = dataset.cat_dims

        # Set embedding dimensions for categorical features
        # According to https://stackoverflow.com/questions/48479915/what-is-the-preferred-ratio-between-the-vocabulary-size-and-embedding-dimension
        emb_dims = [(x, min(16, int(x**0.25))) for x in cat_dims] 
        config['emb_dims'] = emb_dims    
    #####
    if without_bw_distal:
        in_channels = 4**distal_order
    else:
        in_channels = 4**distal_order+n_cont
    #####
    
    # Choose the network model for training
    common_model_config = {
        'emb_dims': emb_dims,
        'n_cont': n_cont,
        'n_class': n_class,
        'distal_order': distal_order,
        'in_channels': in_channels,
        }
    model = model_choice(model_no, config, common_model_config, model_type)
    
    model.to(device)
    # Count the parameters in the model
    total_params = count_parameters(model)
    print('model:')
    print(model)
    
    '''
    writer = SummaryWriter('runs/test')
    dataiter = iter(dataloader_train)
    y, cont_x, cat_x, distal_x = dataiter.next()
    cat_x = cat_x.to(device)
    cont_x = cont_x.to(device)
    distal_x = distal_x.to(device)
    writer.add_graph(model, ((cont_x, cat_x), distal_x))
    writer.close()
    '''
    if config['transfer_learning']:
        #model_state = torch.load(args.model_path, map_location=device)
        model_state = torch.load(args.model_path, map_location='cpu')
        model.to(torch.device('cpu'))# if loaded into GPU, it will double the GPU memory!
        model.load_state_dict(model_state)
        model.to(device)
        
        del model_state
        torch.cuda.empty_cache() 

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        if config['train_all']:
            # Train all parameters
            for param in model.parameters():
                param.requires_grad = True
        else:
            # Train only the final fc layers
            for param in model.parameters():
                param.requires_grad = False
            model.local_fc[-1].weight.requires_grad = True
            model.local_fc[-1].bias.requires_grad = True
            model.distal_fc[-1].weight.requires_grad = True
            model.distal_fc[-1].bias.requires_grad = True

        if not config['init_fc_with_pretrained']:
            # Re-initialize fc layers
            model.local_fc[-1].apply(weights_init)
            model.distal_fc[-1].apply(weights_init)    
    else:
        # Initiating weights of the models;
        model.apply(weights_init)
    
    # Set loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    #weights = torch.tensor([0.00515898, 0.44976093, 0.23657462, 0.30850547]).to(device)
    #weights = torch.tensor([0.00378079, 0.42361806, 0.11523816, 0.45736299]).to(device)
    #criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='sum')
    
    #criterion = PCSoftmaxCrossEntropyV1(lb_proportion=[0.952381, 0.0140095, 0.0198, 0.0138095], reduction='sum')
    #print("using Focal Loss ...")
    #criterion = FocalLoss(gamma=2, size_average=False)
    #criterion = CBLoss(samples_per_cls=[400000, 5884, 8316, 5800], no_of_classes=4, loss_type="sigmoid", beta=0.999999, gamma=1)
    #criterion = CBLoss(samples_per_cls=[400000, 5884, 8316, 5800], no_of_classes=4, loss_type="focal", beta=0.999999, gamma=1)
    
    if weight_decay_auto != None and weight_decay_auto > 0:
        #print("rewriting config['weight_decay'] ...")
        if weight_decay_auto >=1:
            print('Please set a value smaller than 1 for --weight_decay_auto.', file=sys.stderr)
            sys.exit()
        config['weight_decay'] = 1- weight_decay_auto **(config['batch_size']/(epochs*train_size))
        print("NOTE: rewriting config['weight_decay'], new weight_decay: ", config['weight_decay'])
    
    # Set Optimizer
    if config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    elif config['optim'] == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'], amsgrad=True)

    elif config['optim'] == 'AdamW2':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'], amsgrad=True)
        
    elif config['optim'] == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=0.98, nesterov=True)
     
    else:
        print('Error: unsupported optimization method', config['optim'])
        sys.exit()


    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['LR_gamma'])
    if config['lr_scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(5000*128)//config['batch_size'], gamma=config['LR_gamma'])
    elif config['lr_scheduler'] == 'StepLR2':
        gamma = (config['min_lr']/config['restart_lr'])**(1/(train_size//config['batch_size']))
        print('learning rate gamma:', gamma)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    elif config['lr_scheduler'] == 'ROP':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.2, patience=1, threshold=0.0001, min_lr=1e-7)
        print("using lr_scheduler.ReduceLROnPlateau ...")

    print('optimizer:', optimizer)
    print('scheduler:', scheduler)
    sys.stdout.flush()

    prob_names = ['prob'+str(i) for i in range(n_class)]
    
    min_loss = 0
    min_loss_epoch = 0
    after_min_loss = 0
    if not use_ray:
        early_stopping = EarlyStopping(patience=grace_period, verbose=True)
    # Training loop
    for epoch in range(epochs):
        
        step_time = time.time()
        epoch_time = time.time()

        model.train()
        total_loss = 0
        batch_count = 0
        
        if epoch > 0 and config['lr_scheduler'] == 'StepLR2':
            for g in optimizer.param_groups:
                g['lr'] = config['restart_lr']            
        ### get batch time view ###########
        time_per_batch_fetch = 0
        time_per_batch_training = 0
        get_batch_time = time.time()
        ############################
        for y, cont_x, cat_x, distal_x in dataloader_train:
            time_per_batch_fetch += time.time() - get_batch_time
            batch_count += 1
            ### get batch time view ##############
            if batch_count % 1000 == 0:
                print("get 1000 batch used time: ", time_per_batch_fetch)
                time_per_batch_fetch = 0
            #############################
            
            batch_train_time = time.time() #
            
            if y.shape[0] == 1:
                continue

            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)

            # Forward Pass
            preds = model.forward((cont_x, cat_x), distal_x)
            loss = criterion(preds, y.long().squeeze())
            optimizer.zero_grad()
            loss.backward()
            # time view
            if batch_count % 1000 == 0:
                print(f"Batch Number: {batch_count}; Mean Time of 1000 batch: {(time.time()-step_time) / 60} min")
                step_time = time.time()
            
            #Clips gradient norm to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, error_if_nonfinite=False)
            
            optimizer.step()
            total_loss += loss.item()
            ### get train time view ##############
            time_per_batch_training += time.time() - batch_train_time
            if batch_count % 1000 == 0:
                print("training 1000 batch used time: ", time_per_batch_training)
                time_per_batch_training = 0
            ##################  
            if config['lr_scheduler'] != 'ROP':
                scheduler.step() 
                # avoid very small learning rates
                if optimizer.param_groups[0]['lr'] < config['min_lr']:
                    print("optimizer.param_groups[0]:", optimizer.param_groups[0]['lr'])
                    for g in optimizer.param_groups:
                        g['lr'] = config['restart_lr']
                        #scheduler.step(1)
            get_batch_time = time.time()
        # Flush StdOut buffer
        sys.stdout.flush()
        
        print('optimizer learning rate:', optimizer.param_groups[0]['lr'])
        # Update learning rate
        #scheduler.step()
        
        model.eval()
        with torch.no_grad():
            #if model_no != 0:
                #print('F.softmax(model.models_w):', F.softmax(model.models_w, dim=0))
                #print('model.conv1[0].weight.grad:', model.conv1[0].weight.grad)
            #print('model.conv1.0.weight.grad:', model.conv1.0.weight)
            save_path = get_save_path(args.use_ray, args.trial_dir, epoch)

            valid_pred_y, valid_total_loss = model_predict_m(model, dataloader_valid, criterion, device, n_class, distal=True)

            valid_y_prob = pd.DataFrame(data=to_np(F.softmax(valid_pred_y, dim=1)), columns=prob_names)
            
            valid_data_and_prob = pd.concat([data_local_valid, valid_y_prob], axis=1)
            
            valid_y = valid_data_and_prob['mut_type'].to_numpy().squeeze()
            
            # Train the calibrator using the validataion data
            #valid_y_prob = valid_y_prob.reset_index() #### for "ValueError: Input contains NaN"
            fdiri_cal, fdiri_nll = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='FullDiri')
            #fdirio_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='FullDiriODIR')
            #vec_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='VectS')
            #tmp_cal, _ = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='TempS')

            prob_cal = fdiri_cal.predict_proba(valid_y_prob.to_numpy())  # calibrate
            #Evaluation
            evaluator_before_calibra = Evaluator(data_local_valid, valid_y_prob, n_class, printer=print)
            evaluator_after_calibra = Evaluator(data_local_valid, prob_cal, n_class, calibra="FullDiri", printer=print)

            kmer_list = [2, 4, 6] if model_type == 'indel' else [3, 5, 7]

            evaluator_before_calibra.evaluate_kmer(kmer_list)
            evaluator_after_calibra.evaluate_kmer(kmer_list)

            print ('Training Loss: ', total_loss/train_size)
            print ('Validation Loss: ', valid_total_loss/valid_size)
            print ('Validation Loss (after fdiri_cal): ', fdiri_nll)
            
            # Calculate a custom score by looking obs/pred 3/5-mer correlations in binned windows
            evaluator_before_calibra.evaluate_regional_score(valid_size, kmer_list[:2])
            evaluator_after_calibra.evaluate_regional_score(valid_size, kmer_list[:2])

            # Output genomic positions and predicted probabilities
            if not valid_file:
                chr_pos = get_position_info_by_trainset(train_bed, config['segment_center'])
                chr_pos = chr_pos.loc[dataset_valid.indices].reset_index(drop=True)
            else:
                chr_pos = get_position_info(valid_bed, config['segment_center'])

            evaluator_before_calibra.evaluate_regional_corr(chr_pos, save_valid_preds=args.save_valid_preds, save_path=save_path)
            evaluator_after_calibra.evaluate_regional_corr(chr_pos)

            # Save model data for each checkpoint
            save_model(model, fdiri_cal, config, save_path)

                    
            current_loss = valid_total_loss/valid_size
            if epoch == 0 or current_loss < min_loss:
                min_loss = current_loss
                min_loss_epoch = epoch
                after_min_loss = 0
            else:
                after_min_loss = epoch - min_loss_epoch
            
            if use_ray:
                tune.report(loss=current_loss, fdiri_loss=fdiri_nll, after_min_loss=after_min_loss, score=score, total_params=total_params)
            else:
                early_stopping(current_loss, model)
                metrics = {
                    'loss': current_loss,
                    'fdiri_loss': fdiri_nll,
                    'after_min_loss': after_min_loss,
                    'score': evaluator_before_calibra.metrics['score'],
                    'total_params': total_params
                    }
                non_ray_checkpoint_dir = f'{trial_dir}/checkpoint_{epoch}'
                report_path = os.path.join(non_ray_checkpoint_dir, f'epoch_{epoch}_metrics.txt')
                report_metrics(metrics, report_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            #####
            if config['lr_scheduler'] == 'ROP':
                scheduler.step(current_loss)
            torch.cuda.empty_cache() 
            
        print(f"Epoch {epoch} used time:{time.time()-epoch_time} seconds!")
        epoch_time = time.time()
        sys.stdout.flush()

        
        if not custom_dataloader:
            dataloader_train = generate_data_batches(segmentLoader_train, config['sampled_segments'], config['batch_size'], shuffle=True)
            dataloader_valid = generate_data_batches(segmentLoader_valid, config['sampled_segments'], config['batch_size'], shuffle=False)


    print(f"dital_radius: {distal_radius} training finish, {epoch} epochs total time:{time.time()-start_time} min!")
    best_epoch = epoch - early_stopping.counter
    print(f"Best Epoch: {best_epoch}")

def save_model(model, fdiri_cal, config, save_path):
    """Save model state, fdiri_cal, config and validation predictions to the specified path."""
    torch.save(model.state_dict(), save_path)

    with open(save_path + '.fdiri_cal.pkl', 'wb') as pkl_file:
        pickle.dump(fdiri_cal, pkl_file)

    with open(save_path + '.config.pkl', 'wb') as fp:
        pickle.dump(config, fp)


def report_metrics(metrics, report_path=None):
    """Report metrics by saving them to a file or printing them."""
    if report_path:
        with open(report_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    else:
        for key, value in metrics.items():
            print(f"{key}: {value}")


def get_save_path(use_ray, trial_dir, epoch):
    if use_ray:
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'model')
    else:
        # Define a directory to save the files when not using Ray
        non_ray_checkpoint_dir = f'{trial_dir}/checkpoint_{epoch}'
        os.makedirs(non_ray_checkpoint_dir, exist_ok=True)
        path = os.path.join(non_ray_checkpoint_dir, 'model')
    return path