import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def weights_init(m):
    """Initialize layers"""
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight)
        
        if m.bias is not None:
            nn.init.normal_(m.bias)
        
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
            
        if m.bias is not None:
            nn.init.normal_(m.bias)
        
    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(m.__getattr__(p))

def model_predict_m(model, dataloader, criterion, device, n_class, distal=True):
    """Do model prediction using dataloader"""
    model = model.to(device)
    model.eval()
    
    pred_y = torch.empty(0, n_class).to(device)        
    total_loss = 0
    
    with torch.no_grad():
        for y, cont_x, cat_x, distal_x in dataloader:
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

    return pred_y, total_loss