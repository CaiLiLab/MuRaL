import sys
import math
import random
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, calibration

from torchsummary import summary


class FeedForwardNN(nn.Module):

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts):

        """
        Parameters
        ----------

        emb_dims: List of two element tuples
            This list will contain a two element tuple for each
            categorical feature. The first element of a tuple will
            denote the number of unique values of the categorical
            feature. The second element will denote the embedding
            dimension to be used for that feature.

        no_of_cont: Integer
            The number of continuous features in the data.

        lin_layer_sizes: List of integers.
            The size of each linear layer. The length will be equal
            to the total number
            of linear layers in the network.

        output_size: Integer
            The size of the final output.

        emb_dropout: Float
            The dropout to be used after the embedding layers.

        lin_layer_dropouts: List of floats
            The dropouts to be used after each linear layer.
        """

        super(FeedForwardNN, self).__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers =\
         nn.ModuleList([first_lin_layer] +\
                    [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                     for i in range(len(lin_layer_sizes) - 1)])
        
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], 1)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):

        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            ###
        #print(len(x)) #x is a list, x[i].shape: batch_size * emb_size  
        #print(x)
            ###
        x = torch.cat(x, 1) #x.shape: batch_size * sum(emb_size)
        x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1) 
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in\
                zip(self.lin_layers, self.droput_layers, self.bn_layers):
            
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        #x = self.output_layer(x) #oringial output
        
        x = torch.sigmoid(self.output_layer(x)) # for logistic regression

        return x

    def batch_predict(self, dataloader, criterion, device):
        
        self.eval()
        
        pred_y = torch.empty(0, 1).to(device)
        total_loss = 0

        with torch.no_grad():
            for y, cont_x, cat_x, _ in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                y = y.to(device)
        
                preds = self.forward(cont_x, cat_x)
                pred_y = torch.cat((pred_y, preds), dim=0)
                
                loss = criterion(preds, y)
                total_loss += loss.item()

        return pred_y, total_loss

class Network(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order):
        
        super(Network, self).__init__()
        #self.cnn = nn.Conv1d(in_channels, out_channels, kernel_size)
        #self.maxpool =  nn.MaxPool1d(kernel_size, stride)
        
        # FeedForward layers for local input
        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])
        
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])

        
        self.kernel_size = kernel_size
        self.RNN_hidden_size = RNN_hidden_size
        self.RNN_layers = RNN_layers
        self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        # CNN layers for distal input
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels), #this is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool1d(4, 2), # kernel_size, stride
            #nn.Conv1d(out_channels*2, out_channels, kernel_size),
            #nn.ReLU(),
            #nn.MaxPool1d(4, 2)
        )
        
        # RNN layers
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            self.rnn = nn.LSTM(out_channels, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            fc_in_size = RNN_hidden_size*2 + lin_layer_sizes[-1]
        else:
            fc_in_size = out_channels + lin_layer_sizes[-1]
        
        # FC layers      
        self.fc = nn.Sequential(
            nn.BatchNorm1d(fc_in_size),
            nn.Dropout(0.1),
            nn.Linear(fc_in_size, last_lin_size), 
            nn.ReLU(),
            nn.Dropout(0.2), #dropout prob
            #nn.Linear(out_channels*2, 1),
            nn.Linear(last_lin_size, 1),
            #nn.ReLU(), #NOTE: addting this makes two early convergence
            #nn.Dropout(0.1)
        )
        
        #nn.init.kaiming_normal_(self.fc.weight.data) #
    
    def forward(self, local_input, distal_input):
        
        #FeedForward layers for local input
        cont_data, cat_data = local_input
        if self.no_of_embs != 0:
            local_out = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            
        #print(len(x)) #x is a list, x[i].shape: batch_size * emb_size  
        #print('cont_data.shape:')
        #print(cont_data.shape)
        
        local_out = torch.cat(local_out, dim = 1) #x.shape: batch_size * sum(emb_size)
        local_out = self.emb_dropout_layer(local_out)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                local_out = torch.cat([local_out, normalized_cont_data], dim = 1) 
            else:
                local_out = normalized_cont_data
        
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            local_out = F.relu(lin_layer(local_out))
            local_out = bn_layer(local_out)
            local_out = dropout_layer(local_out)
        
        
        # CNN layers for distal_input
        #input data shape: batch_size, in_channels, L_in (lenth of sequence)
        distal_out = self.conv(distal_input) #out_shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)
        #out, _ = torch.max(out, dim=2)
        #print("out.shape")
        #print(out.shape)

        #RNN after CNN
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            distal_out = distal_out.permute(2,0,1)
            distal_out, _ = self.rnn(distal_out) # output of shape (seq_len, batch, num_directions * hidden_size)
            Fwd_RNN=distal_out[-1, :, :self.RNN_hidden_size] # output of last position
            Rev_RNN=distal_out[0, :, self.RNN_hidden_size:] # output of last position
            distal_out = torch.cat([Fwd_RNN, Rev_RNN], dim=1)
        else:
            distal_out, _ = torch.max(distal_out, dim=2)
        #print("RNN out.shape")
        #print(distal_out.shape)        

        out = torch.cat([local_out, distal_out], dim=1)
        
        out = self.fc(out)
        
        return torch.sigmoid(out)
    
    # do prediction using batches in DataLoader to save memory 
    def batch_predict(self, dataloader, criterion, device):
 
        self.eval()
        pred_y = torch.empty(0, 1).to(device)
        
        total_loss = 0

        with torch.no_grad():
            for y, cont_x, cat_x, distal_x in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                distal_x = distal_x.to(device)
                y  = y.to(device)
        
                preds = self.forward((cont_x, cat_x), distal_x)
                pred_y = torch.cat((pred_y, preds), dim=0)
                
                loss = criterion(preds, y)
                total_loss += loss.item()

        return pred_y, total_loss

class Network2(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order):
        
        super(Network2, self).__init__()
        #self.cnn = nn.Conv1d(in_channels, out_channels, kernel_size)
        #self.maxpool =  nn.MaxPool1d(kernel_size, stride)
        
        # FeedForward layers for local input
        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])
        
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])

        
        self.kernel_size = kernel_size
        self.RNN_hidden_size = RNN_hidden_size
        self.RNN_layers = RNN_layers
        self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        # CNN layers for distal input
        maxpool_kernel_size = 10
        maxpool_stride = 10
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels*2, kernel_size), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.MaxPool1d(maxpool_kernel_size, maxpool_stride), # kernel_size, stride
            
            nn.Conv1d(out_channels*2, out_channels, kernel_size//3),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.MaxPool1d(2, 2)
        )
        
        
        # RNN layers
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            self.rnn = nn.LSTM(out_channels, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            fc_in_size = RNN_hidden_size*2 + lin_layer_sizes[-1]
            crnn_fc_in_size = RNN_hidden_size*2
        else:
            fc_in_size = out_channels + lin_layer_sizes[-1]
            crnn_fc_in_size = out_channels
            
            ##### use the flattened output of CNN instead of torch.max
            last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
            last_seq_len = (last_seq_len - (kernel_size//2 -1) )//2 #for the 2nd conv1d
            
            #crnn_fc_in_size = out_channels*last_seq_len
        
        #=== separate FC layers for distal and local ====
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(crnn_fc_in_size),
            nn.Dropout(0.3),
            nn.Linear(crnn_fc_in_size, 50), 
            nn.ReLU(),
            nn.Dropout(0.2), #dropout prob
            
            #nn.Linear(crnn_fc_in_size, 1),
            #nn.Linear(out_channels*2, 1),
            nn.Linear(50, 1),
            #nn.Dropout(0.1)
        )
        
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            nn.Linear(lin_layer_sizes[-1], 1), 
        )       
        
        self.fc2to1 = nn.Linear(2, 1)
        self.w_ld = torch.nn.Parameter(torch.Tensor([0]))

        #====================================
        
        
        # FC layers      
        self.fc = nn.Sequential(
            nn.BatchNorm1d(fc_in_size),
            nn.Linear(fc_in_size, last_lin_size), 
            nn.ReLU(),
            nn.Dropout(0.1), #dropout prob
            #nn.Linear(out_channels*2, 1),
            nn.Linear(last_lin_size, 1),
            #nn.ReLU(), #NOTE: addting this makes two early convergence
            #nn.Dropout(0.1)
        )
        
        #nn.init.kaiming_normal_(self.fc.weight.data) #
    
    def forward(self, local_input, distal_input):
        
        #FeedForward layers for local input
        cont_data, cat_data = local_input
        if self.no_of_embs != 0:
            local_out = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            
        #print(len(x)) #x is a list, x[i].shape: batch_size * emb_size  
        #print('cont_data.shape:')
        #print(cont_data.shape)
        
        local_out = torch.cat(local_out, dim = 1) #x.shape: batch_size * sum(emb_size)
        local_out = self.emb_dropout_layer(local_out)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                local_out = torch.cat([local_out, normalized_cont_data], dim = 1) 
            else:
                local_out = normalized_cont_data
        
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            local_out = F.relu(lin_layer(local_out))
            local_out = bn_layer(local_out)
            local_out = dropout_layer(local_out)
        
        # CNN layers for distal_input
        #input data shape: batch_size, in_channels, L_in (lenth of sequence)
        distal_out = self.conv(distal_input) #out_shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)
        #out, _ = torch.max(out, dim=2)
        #print("out.shape")
        #print(out.shape)

        #RNN after CNN
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            distal_out = distal_out.permute(2,0,1)
            distal_out, _ = self.rnn(distal_out) # output of shape (seq_len, batch, num_directions * hidden_size)
            Fwd_RNN=distal_out[-1, :, :self.RNN_hidden_size] # output of last position
            Rev_RNN=distal_out[0, :, self.RNN_hidden_size:] # output of last position
            distal_out = torch.cat([Fwd_RNN, Rev_RNN], dim=1)
        else:
            
            distal_out, _ = torch.max(distal_out, dim=2)
            
            #use flattened layer instead of torchmax
            #distal_out = distal_out.view(distal_out.shape[0], -1)
        #print("RNN out.shape")
        #print(distal_out.shape)        
        
        #=========separate FC layers ===========
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        if np.random.uniform(0,1) < 0.01 and self.training == False:
            print('local_out:', torch.min(local_out).item(), torch.max(local_out).item(), torch.var(local_out).item(), torch.var(torch.sigmoid(local_out)).item())
            print('distal_out:', torch.min(distal_out).item(), torch.max(distal_out).item(),torch.var(distal_out).item(), torch.var(torch.sigmoid(distal_out)).item())
        
        #out = local_out * torch.sigmoid(distal_out)
        #out = local_out * distal_out # NO
        #out = local_out * torch.exp(distal_out)
        #out = local_out + distal_out

        #=======================================
        
        #out = torch.cat([local_out, distal_out], dim=1)
        #out = self.fc(out)
        
        #out = torch.sigmoid(local_out + distal_out)
        #out = (torch.sigmoid(local_out) + torch.sigmoid(distal_out))/2
        #out = torch.sigmoid(local_out) * torch.sigmoid(distal_out) # OK for large data?
        #out = torch.sigmoid(out)
        #out = torch.sigmoid(local_out)
        out = torch.sigmoid(local_out) * torch.sigmoid(self.w_ld) + torch.sigmoid(distal_out)*(1-torch.sigmoid(self.w_ld)) #set the weight as a Parameter when adding local and distal
        
        return out
    
    # do prediction using batches in DataLoader to save memory 
    def batch_predict(self, dataloader, criterion, device):
 
        self.eval()
        pred_y = torch.empty(0, 1).to(device)
        
        total_loss = 0

        with torch.no_grad():
            for y, cont_x, cat_x, distal_x in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                distal_x = distal_x.to(device)
                y  = y.to(device)
        
                preds = self.forward((cont_x, cat_x), distal_x)
                pred_y = torch.cat((pred_y, preds), dim=0)
                
                loss = criterion(preds, y)
                total_loss += loss.item()

        return pred_y, total_loss
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        #nn.init.normal_(m.weight, 0.0, 1e-06)
        nn.init.xavier_uniform_(m.weight)
        nn.init.normal_(m.bias)

        
        print(m.weight.shape)
    elif classname.find('Linear') != -1:
        #nn.init.normal_(m.weight, 0, 0.004)
        #nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_normal_(m.weight.data)
            
        nn.init.normal_(m.bias)
        print(m.weight.shape)
    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(m.__getattr__(p))


