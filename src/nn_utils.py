import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
        #nn.init.normal_(m.weight, 0.0, 1e-06)
        nn.init.xavier_uniform_(m.weight)
        #nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias)
        
        #print(m.weight.shape)
        
    elif classname.find('Linear') != -1:
        #nn.init.normal_(m.weight, 0, 0.004)
        #nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_normal_(m.weight)
            
        if m.bias is not None:
            nn.init.normal_(m.bias)
        #print(m.weight.shape)
        
    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(m.__getattr__(p))
                 
def two_model_predict(model, model2, dataloader, criterion, device):
 
    model = model.to(device)
    model2 = model2.to(device)
    model.eval()
    model2.eval()
    
    pred_y = torch.empty(0, 1).to(device)
    pred_y2 = torch.empty(0, 1).to(device)
        
    total_loss = 0
    total_loss2 = 0

    with torch.no_grad():
        for y, cont_x, cat_x, distal_x in dataloader:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)
        
            preds = model.forward((cont_x, cat_x), distal_x)
            pred_y = torch.cat((pred_y, preds), dim=0)
                
            loss = criterion(preds, y)
            total_loss += loss.item()
            
            preds = model2.forward(cont_x, cat_x)
            pred_y2 = torch.cat((pred_y2, preds), dim=0)
            loss2 = criterion(preds, y)
            total_loss2 += loss2.item()

    return pred_y, total_loss, pred_y2, total_loss2 

def two_model_predict_m(model, model2, dataloader, criterion, device, n_class):
 
    model = model.to(device)
    model2 = model2.to(device)
    model.eval()
    model2.eval()
    
    pred_y = torch.empty(0, n_class).to(device)
    pred_y2 = torch.empty(0, n_class).to(device)
        
    total_loss = 0
    total_loss2 = 0
    
    #print("in two_model_predict_m, current CUDA:", torch.cuda.current_device())
    
    with torch.no_grad():
        for y, cont_x, cat_x, distal_x in dataloader:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)
        
            preds = model.forward((cont_x, cat_x), distal_x)
            pred_y = torch.cat((pred_y, preds), dim=0)
            #print('pred_y:', pred_y[1:10])
            #print('y:', y[1:10])
            #print('pred_y.shape, preds.shape, y.shape, distal_x.shape', pred_y.shape, preds.shape, y.shape, distal_x.shape)
                
            loss = criterion(preds, y.long().squeeze(1))
            total_loss += loss.item()
            
            preds = model2.forward(cont_x, cat_x)
            #print('pred_y2.shape, preds.shape', pred_y2.shape, preds.shape)
            pred_y2 = torch.cat((pred_y2, preds), dim=0)
            loss2 = criterion(preds, y.long().squeeze(1))
            total_loss2 += loss2.item()

    return pred_y, total_loss, pred_y2, total_loss2 

def model_predict_m(model, dataloader, criterion, device, n_class, distal=True):
 
    model = model.to(device)
    model.eval()
    
    pred_y = torch.empty(0, n_class).to(device)        
    total_loss = 0
    
    #print("in two_model_predict_m, current CUDA:", torch.cuda.current_device())
    
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
            #print('pred_y:', pred_y[1:10])
            #print('y:', y[1:10])
            #print('pred_y.shape, preds.shape, y.shape, distal_x.shape', pred_y.shape, preds.shape, y.shape, distal_x.shape)
                
            loss = criterion(preds, y.long().squeeze(1))
            total_loss += loss.item()

    return pred_y, total_loss