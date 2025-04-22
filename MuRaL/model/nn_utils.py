import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys


def weights_init(m):
    """Initialize network layers"""
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight)
        
        if m.bias is not None:
            #nn.init.normal_(m.bias)
            nn.init.constant_(m.bias, 0)
        
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
            
        if m.bias is not None:
            #nn.init.normal_(m.bias)
            nn.init.constant_(m.bias, 0)
        
    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(m.__getattr__(p))

def model_predict_m(model, dataloader, criterion, device, n_class, distal=True):
    """Do model prediction using dataloader"""
    import time
    model.to(device)
    model.eval()
    
    pred_y = torch.empty(0, n_class).to(device)        
    total_loss = 0
    batch_count = 0
    step_time = time.time()
    with torch.no_grad():
        for y, cont_x, cat_x, distal_x in dataloader:
            batch_count += 1
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)
        
            if distal:
                preds = model.forward((cont_x, cat_x), distal_x)
            else:
                preds = model.forward(cont_x, cat_x)
            pred_y = torch.cat((pred_y, preds), dim=0)
                
            loss = criterion(preds, y.long().squeeze(1))
            total_loss += loss.item()
            
            if device == torch.device('cpu'):
                if  np.random.uniform(0,1) < 0.0001:
                    #print('in the model_predict_m:', device)
                    sys.stdout.flush()
            
    # time view
    print(f"Batch Number: {batch_count}; prediction Time of {batch_count} batch: {(time.time()-step_time) / 60} min")
    sys.stdout.flush()

    return pred_y, total_loss



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss









def run_time_view_model_predict_m(model, dataloader, criterion, device, n_class, distal=True):
    """Do model prediction using dataloader"""
    import time
    model.to(device)
    model.eval()
    
    pred_y = torch.empty(0, n_class).to(device)        
    total_loss = 0
    batch_count = 0
    get_batch_time_recode = 0
    get_batch_predict_recode = 0
    step_time = time.time()
    get_batch_time = time.time()

    with torch.no_grad():
        for y, cont_x, cat_x, distal_x in dataloader:
            get_batch_time_recode += time.time() - get_batch_time
            batch_count += 1
            
            if batch_count % 500 == 0:
                print("get 500 batch used time: ", get_batch_time_recode)
                get_batch_time_recode = 0
            
            batch_predict_time = time.time()
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)
        
            if distal:
                preds = model.forward((cont_x, cat_x), distal_x)
            else:
                preds = model.forward(cont_x, cat_x)
            pred_y = torch.cat((pred_y, preds), dim=0)
                
            loss = criterion(preds, y.long().squeeze(1))
            total_loss += loss.item()

            if batch_count % 500 == 0:
                print(f"Batch Number: {batch_count}; Mean Time of 500 batch: {(time.time()-step_time)}")
                step_time = time.time()
 
            get_batch_predict_recode += time.time() - batch_predict_time
            if batch_count % 500 == 0:
                print("training 500 batch used time:", get_batch_predict_recode)
                get_batch_predict_recode = 0
            get_batch_time = time.time()

            if device == torch.device('cpu'):
                if  np.random.uniform(0,1) < 0.0001:
                    #print('in the model_predict_m:', device)
                    sys.stdout.flush()
            
    # time view
    print(f"Batch Number: {batch_count}; prediction Time of {batch_count} batch: {(time.time()-step_time)}")
    sys.stdout.flush()

    return pred_y, total_loss

