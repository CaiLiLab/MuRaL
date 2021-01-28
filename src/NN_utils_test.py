import sys
import math
import random
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, calibration
from scipy.special import lambertw

from evaluation import *

#from torchsummary import summary


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
        
        # Do the embedding for the categrical features
        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]

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
        
        out = torch.sigmoid(self.output_layer(x))

        return out
    
    # Do prediction using batches in DataLoader to save memory
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

class FeedForwardNNm(nn.Module):

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, n_class, emb_padding_idx=None):

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

        super(FeedForwardNNm, self).__init__()
        
        self.n_class = n_class
        # Embedding layers
        #self.emb_layers = nn.ModuleList([nn.Embedding(x, y, padding_idx = emb_padding_idx) for x, y in emb_dims])
        self.emb_layers = nn.ModuleList([nn.Embedding(emb_padding_idx+1, y, padding_idx = emb_padding_idx) for x, y in emb_dims])

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
        self.output_layer = nn.Linear(lin_layer_sizes[-1], n_class)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):
        
        # Do the embedding for the categrical features
        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]

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
        
        #out = torch.sigmoid(self.output_layer(x))
        #out = F.log_softmax(self.output_layer(x), dim=1)
        out = self.output_layer(x)

        return out
    
    # Do prediction using batches in DataLoader to save memory
    def batch_predict(self, dataloader, criterion, device):
        
        self.eval()
        
        pred_y = torch.empty(0, self.n_class).to(device)
        total_loss = 0

        with torch.no_grad():
            for y, cont_x, cat_x, _ in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                y = y.to(device)
        
                preds = self.forward(cont_x, cat_x)
                pred_y = torch.cat((pred_y, preds), dim=0)
                
                loss = criterion(preds, y.long().squeeze())
                total_loss += loss.item()

        return pred_y, total_loss

    
# Hybrid network with feedforward (local) and CNN/RNN (distal) layers, followed by a FC layer for combined output 
class Network(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order):
        
        super(Network, self).__init__()
        
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
            nn.BatchNorm1d(in_channels), #This is important!
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
            #nn.ReLU(), # NOTE: adding this makes too early convergence
            #nn.Dropout(0.1)
        )
    
    def forward(self, local_input, distal_input):
        
        # FeedForward layers for local input
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
        # Input data shape: batch_size, in_channels, L_in (lenth of sequence)
        distal_out = self.conv(distal_input) #out_shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)
        #out, _ = torch.max(out, dim=2)
        #print("out.shape")
        #print(out.shape)

        # Add RNN after CNN
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            distal_out = distal_out.permute(2,0,1)
            distal_out, _ = self.rnn(distal_out) # output of shape (seq_len, batch, num_directions * hidden_size)
            Fwd_RNN=distal_out[-1, :, :self.RNN_hidden_size] # output of last position
            Rev_RNN=distal_out[0, :, self.RNN_hidden_size:] # output of last position
            distal_out = torch.cat([Fwd_RNN, Rev_RNN], dim=1)
        else:
            # Flatten the sequence dimension
            distal_out, _ = torch.max(distal_out, dim=2)
        #print("RNN out.shape")
        #print(distal_out.shape)        

        out = torch.cat([local_out, distal_out], dim=1)
        
        out = self.fc(out)
        
        return torch.sigmoid(out)
    
    # Do prediction using batches in DataLoader to save memory 
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

# Hybrid network with feedforward and CNN/RNN layers; the FC layers of local and distal data are separated.
class Network2(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order):
        
        super(Network2, self).__init__()

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
        second_kernel_size = kernel_size//2
        third_kernel_size = kernel_size//3
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            
            #ResBlock(out_channels, kernel_size=11, stride=1, padding=(11-1)//2, dilation=1),
            #ResBlock(out_channels, kernel_size=11, stride=1, padding=(11-1)//2, dilation=1),
            #ResBlock(out_channels, kernel_size=11, stride=1, padding=(11-1)//2, dilation=1),
            #ResBlock(out_channels, kernel_size=11, stride=1, padding=(11-1)//2, dilation=1),
            nn.MaxPool1d(maxpool_kernel_size, maxpool_stride), # kernel_size, stride
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels*2, second_kernel_size),
            nn.ReLU(),
            
            #ResBlock(out_channels*2, kernel_size=7, stride=1, padding=(7-1)//2, dilation=1),
            #ResBlock(out_channels*2, kernel_size=7, stride=1, padding=(7-1)//2, dilation=1),
            #ResBlock(out_channels*2, kernel_size=7, stride=1, padding=(7-1)//2, dilation=1),
            #ResBlock(out_channels*2, kernel_size=7, stride=1, padding=(7-1)//2, dilation=1),
            nn.MaxPool1d(4, 4),
            nn.BatchNorm1d(out_channels*2),
            nn.Conv1d(out_channels*2, out_channels*3, third_kernel_size),
            nn.ReLU(),
            #nn.MaxPool1d(2, 2),
        )
        
        
        # RNN layers
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            self.rnn = nn.LSTM(out_channels*2, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            #fc_in_size = RNN_hidden_size*2 + lin_layer_sizes[-1]
            crnn_fc_in_size = RNN_hidden_size*2
        else:
            #fc_in_size = out_channels + lin_layer_sizes[-1]
            #crnn_fc_in_size = out_channels*2
            crnn_fc_in_size = out_channels*3
            
            # Use the flattened output of CNN instead of torch.max
            last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
            last_seq_len = (last_seq_len - (second_kernel_size-1) )//2 # For the 2nd conv1d
            
            #crnn_fc_in_size = out_channels*last_seq_len
        
        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(crnn_fc_in_size),
            nn.Dropout(0.25), #control overfitting
            nn.Linear(crnn_fc_in_size, 1), 
            #nn.ReLU(),
            
            #nn.Linear(crnn_fc_in_size, 1),
            #nn.Linear(out_channels*2, 1),
            #nn.BatchNorm1d(30),
            #nn.Dropout(0.25), #dropout prob
            #nn.Linear(30, 1),
            #nn.Dropout(0.1)
        )
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            nn.Linear(lin_layer_sizes[-1], 1), 
        )       
        

        # Learn the weight parameter 
        self.w_ld = torch.nn.Parameter(torch.Tensor([0]))
        
    
    def forward(self, local_input, distal_input):
        
        # FeedForward layers for local input
        cont_data, cat_data = local_input
        if self.no_of_embs != 0:
            local_out = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            
        
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
        # Input data shape: batch_size, in_channels, L_in (lenth of sequence)
        distal_out = self.conv(distal_input) #out_shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)
        #out, _ = torch.max(out, dim=2)
        #print("out.shape")
        #print(out.shape)

        # RNN after CNN
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            distal_out = distal_out.permute(2,0,1)
            distal_out, _ = self.rnn(distal_out) # output of shape (seq_len, batch, num_directions * hidden_size)
            Fwd_RNN=distal_out[-1, :, :self.RNN_hidden_size] # output of last position
            Rev_RNN=distal_out[0, :, self.RNN_hidden_size:] # output of last position
            distal_out = torch.cat([Fwd_RNN, Rev_RNN], dim=1)
        else:
            
            distal_out, _ = torch.max(distal_out, dim=2)
            
            # Use flattened layer instead of torchmax
            #distal_out = distal_out.view(distal_out.shape[0], -1)
        #print("RNN out.shape")
        #print(distal_out.shape)        
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        if np.random.uniform(0,1) < 0.01 and self.training == False:
            print('local_out:', torch.min(local_out).item(), torch.max(local_out).item(), torch.var(local_out).item(), torch.var(torch.sigmoid(local_out)).item())
            print('distal_out:', torch.min(distal_out).item(), torch.max(distal_out).item(),torch.var(distal_out).item(), torch.var(torch.sigmoid(distal_out)).item())
        
        #out = local_out * torch.sigmoid(distal_out)
        #out = local_out * torch.exp(distal_out)
        #out = local_out + distal_out
        
        #out = torch.cat([local_out, distal_out], dim=1)
        
        #out = torch.sigmoid(local_out + distal_out)
        out = (torch.sigmoid(local_out) + torch.sigmoid(distal_out))/2
        #out = torch.sigmoid(out)
        #out = torch.sigmoid(local_out)
        #out = torch.sigmoid(distal_out)
        
        # Set the weight as a Parameter when adding local and distal
        #out = torch.sigmoid(local_out) * torch.sigmoid(self.w_ld) + torch.sigmoid(distal_out)*(1-torch.sigmoid(self.w_ld)) 
        
        return out
    
    # Do prediction using batches in DataLoader to save memory 
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
    
# Hybrid network with feedforward and ResNet layers; the FC layers of local and distal data are separated.
class Network3(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order):
        
        super(Network3, self).__init__()

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
        second_kernel_size = kernel_size
        third_kernel_size = kernel_size
        rb1_kernel_size = 3
        rb2_kernel_size = 5
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            #nn.ReLU(),
        )
        
        self.RBs1 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb1_kernel_size, stride=1, padding=(rb1_kernel_size-1)//2, dilation=1) for x in range(4)])
            

        self.maxpool1 = nn.MaxPool1d(maxpool_kernel_size, maxpool_stride) # kernel_size, stride
        
        self.conv2 = nn.Sequential(    
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, second_kernel_size),
            #nn.ReLU(),
        )
        
        self.RBs2 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb2_kernel_size, stride=1, padding=(rb2_kernel_size-1)//2, dilation=1) for x in range(4)])

        self.maxpool2 = nn.MaxPool1d(4, 4)
    
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, third_kernel_size),
            nn.ReLU(),
        )
        
        
        # RNN layers
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            #self.rnn = nn.LSTM(out_channels*2, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            self.rnn = nn.LSTM(out_channels, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            #fc_in_size = RNN_hidden_size*2 + lin_layer_sizes[-1]
            crnn_fc_in_size = RNN_hidden_size*2
        else:
            #fc_in_size = out_channels + lin_layer_sizes[-1]
            #crnn_fc_in_size = out_channels*2
            crnn_fc_in_size = out_channels
            
            # Use the flattened output of CNN instead of torch.max
            last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
            last_seq_len = (last_seq_len - (second_kernel_size-1) )//2 # For the 2nd conv1d
            
            #crnn_fc_in_size = out_channels*last_seq_len
        
        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(crnn_fc_in_size),
            nn.Dropout(0.25), #control overfitting
            nn.Linear(crnn_fc_in_size, 1), 
            #nn.ReLU(),
            
            #nn.Linear(crnn_fc_in_size, 1),
            #nn.Linear(out_channels*2, 1),
            #nn.BatchNorm1d(30),
            #nn.Dropout(0.25), #dropout prob
            #nn.Linear(30, 1),
            #nn.Dropout(0.1)
        )
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], 1), 
        )       
        

        # Learn the weight parameter 
        self.w_ld = torch.nn.Parameter(torch.Tensor([0]))
        
    
    def forward(self, local_input, distal_input):
        
        # FeedForward layers for local input
        cont_data, cat_data = local_input
        if self.no_of_embs != 0:
            local_out = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            
        
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
        # Input data shape: batch_size, in_channels, L_in (lenth of sequence)
        #distal_out = self.conv1(distal_input) #out_shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)
        
        jump_input = distal_out = self.conv1(distal_input)
        distal_out = self.RBs1(distal_out)    
        assert(jump_input.shape[2] >= distal_out.shape[2])
        distal_out = distal_out + jump_input[:,:,0:distal_out.shape[2]]
        distal_out = self.maxpool1(distal_out)
        
        jump_input = distal_out = self.conv2(distal_out)
        distal_out = self.RBs2(distal_out)
        assert(jump_input.shape[2] >= distal_out.shape[2])
        distal_out = distal_out + jump_input[:,:,0:distal_out.shape[2]]
        distal_out = self.maxpool2(distal_out)
        
        distal_out = self.conv3(distal_out)
        #d = x.shape[2] - out.shape[2]
        #out = x[:,:,0:x.shape[2]-d] + out
        
        #out, _ = torch.max(out, dim=2)
        #print("out.shape")
        #print(out.shape)

        # RNN after CNN
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            distal_out = distal_out.permute(2,0,1)
            distal_out, _ = self.rnn(distal_out) # output of shape (seq_len, batch, num_directions * hidden_size)
            Fwd_RNN=distal_out[-1, :, :self.RNN_hidden_size] # output of last position
            Rev_RNN=distal_out[0, :, self.RNN_hidden_size:] # output of last position
            distal_out = torch.cat([Fwd_RNN, Rev_RNN], dim=1)
        else:
            
            distal_out, _ = torch.max(distal_out, dim=2)
            
            # Use flattened layer instead of torchmax
            #distal_out = distal_out.view(distal_out.shape[0], -1)
        #print("RNN out.shape")
        #print(distal_out.shape)        
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        if np.random.uniform(0,1) < 0.00005*local_out.shape[0] and self.training == False:
            print('local_out:', torch.min(local_out).item(), torch.max(local_out).item(), torch.var(local_out).item(), torch.var(torch.sigmoid(local_out)).item())
            print('distal_out:', torch.min(distal_out).item(), torch.max(distal_out).item(),torch.var(distal_out).item(), torch.var(torch.sigmoid(distal_out)).item())
        
        #out = local_out * torch.sigmoid(distal_out)
        #out = local_out * torch.exp(distal_out)
        #out = local_out + distal_out
        
        #out = torch.cat([local_out, distal_out], dim=1)
        
        #out = torch.sigmoid(local_out + distal_out)
        out = (torch.sigmoid(local_out) + torch.sigmoid(distal_out))/2
        #out = torch.sigmoid(out)
        #out = torch.sigmoid(local_out)
        #out = torch.sigmoid(distal_out)
        
        # Set the weight as a Parameter when adding local and distal
        #out = torch.sigmoid(local_out) * torch.sigmoid(self.w_ld) + torch.sigmoid(distal_out)*(1-torch.sigmoid(self.w_ld)) 
        
        return out
    
    # Do prediction using batches in DataLoader to save memory 
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

# Hybrid network with feedforward and ResNet layers; the FC layers of local and distal data are separated.
class Network3m(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order, n_class, emb_padding_idx=None):
        
        super(Network3m, self).__init__()
        
        self.n_class = n_class
        # FeedForward layers for local input
        # Embedding layers
        print('emb_dims: ', emb_dims)
        print('emb_padding_idx: ', emb_padding_idx)
        
        #self.emb_layers = nn.ModuleList([nn.Embedding(x, y, padding_idx = emb_padding_idx) for x, y in emb_dims])
        self.emb_layers = nn.ModuleList([nn.Embedding(emb_padding_idx+1, y, padding_idx = emb_padding_idx) for x, y in emb_dims])

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
        second_kernel_size = kernel_size
        third_kernel_size = kernel_size
        rb1_kernel_size = 3
        rb2_kernel_size = 5
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            #nn.ReLU(),
        )
        
        self.RBs1 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb1_kernel_size, stride=1, padding=(rb1_kernel_size-1)//2, dilation=1) for x in range(4)])
            

        self.maxpool1 = nn.MaxPool1d(maxpool_kernel_size, maxpool_stride) # kernel_size, stride
        
        self.conv2 = nn.Sequential(    
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, second_kernel_size),
            #nn.ReLU(),
        )
        
        self.RBs2 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb2_kernel_size, stride=1, padding=(rb2_kernel_size-1)//2, dilation=1) for x in range(4)])

        self.maxpool2 = nn.MaxPool1d(4, 4)
    
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, third_kernel_size),
            nn.ReLU(),
        )
        
        
        # RNN layers
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            #self.rnn = nn.LSTM(out_channels*2, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            self.rnn = nn.LSTM(out_channels, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            #fc_in_size = RNN_hidden_size*2 + lin_layer_sizes[-1]
            crnn_fc_in_size = RNN_hidden_size*2
        else:
            #fc_in_size = out_channels + lin_layer_sizes[-1]
            #crnn_fc_in_size = out_channels*2
            crnn_fc_in_size = out_channels
            
            # Use the flattened output of CNN instead of torch.max
            last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
            last_seq_len = (last_seq_len - (second_kernel_size-1) )//2 # For the 2nd conv1d
            
            #crnn_fc_in_size = out_channels*last_seq_len
        
        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(crnn_fc_in_size),
            nn.Dropout(0.25), #control overfitting
            nn.Linear(crnn_fc_in_size, n_class), 
            #nn.ReLU(),
            
            #nn.Linear(crnn_fc_in_size, 1),
            #nn.Linear(out_channels*2, 1),
            #nn.BatchNorm1d(30),
            #nn.Dropout(0.25), #dropout prob
            #nn.Linear(30, 1),
            #nn.Dropout(0.1)
        )
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], n_class), 
        )       
        

        # Learn the weight parameter 
        self.w_ld = torch.nn.Parameter(torch.Tensor([0]))
        
    
    def forward(self, local_input, distal_input):
        
        # FeedForward layers for local input
        cont_data, cat_data = local_input
        if self.no_of_embs != 0:
            local_out = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            
        
        local_out = torch.cat(local_out, dim = 1) #x.shape: batch_size * sum(emb_size)
        local_out = self.emb_dropout_layer(local_out)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                local_out = torch.cat([local_out, normalized_cont_data], dim = 1) 
            else:
                local_out = normalized_cont_data
        
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            #print('local_out.shape: ', local_out.shape)
            #print('local_out: ', local_out[0, 0:10])
            local_out = F.relu(lin_layer(local_out))
            local_out = bn_layer(local_out)
            local_out = dropout_layer(local_out)
        
        # CNN layers for distal_input
        # Input data shape: batch_size, in_channels, L_in (lenth of sequence)
        #distal_out = self.conv1(distal_input) #out_shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)
        
        jump_input = distal_out = self.conv1(distal_input)
        distal_out = self.RBs1(distal_out)    
        assert(jump_input.shape[2] >= distal_out.shape[2])
        distal_out = distal_out + jump_input[:,:,0:distal_out.shape[2]]
        distal_out = self.maxpool1(distal_out)
        
        jump_input = distal_out = self.conv2(distal_out)
        distal_out = self.RBs2(distal_out)
        assert(jump_input.shape[2] >= distal_out.shape[2])
        distal_out = distal_out + jump_input[:,:,0:distal_out.shape[2]]
        distal_out = self.maxpool2(distal_out)
        
        distal_out = self.conv3(distal_out)
        #d = x.shape[2] - out.shape[2]
        #out = x[:,:,0:x.shape[2]-d] + out
        
        #out, _ = torch.max(out, dim=2)
        #print("out.shape")
        #print(out.shape)

        # RNN after CNN
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            distal_out = distal_out.permute(2,0,1)
            distal_out, _ = self.rnn(distal_out) # output of shape (seq_len, batch, num_directions * hidden_size)
            Fwd_RNN=distal_out[-1, :, :self.RNN_hidden_size] # output of last position
            Rev_RNN=distal_out[0, :, self.RNN_hidden_size:] # output of last position
            distal_out = torch.cat([Fwd_RNN, Rev_RNN], dim=1)
        else:
            
            distal_out, _ = torch.max(distal_out, dim=2)
            
            # Use flattened layer instead of torchmax
            #distal_out = distal_out.view(distal_out.shape[0], -1)
        #print("RNN out.shape")
        #print(distal_out.shape)        
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00005*local_out.shape[0] and self.training == False:
            print('local_out1:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            print('local_out2:', torch.min(local_out[:,2]).item(), torch.max(local_out[:,2]).item(), torch.var(local_out[:,2]).item(), torch.var(F.softmax(local_out, dim=1)[:,2]).item())
            print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())
        
        #out = local_out * torch.sigmoid(distal_out)
        #out = local_out * torch.exp(distal_out)
        #out = local_out + distal_out
        
        #out = torch.cat([local_out, distal_out], dim=1)
        
        #out = torch.sigmoid(local_out + distal_out)
        
        #out = (torch.sigmoid(local_out) + torch.sigmoid(distal_out))/2
        
        out = torch.log((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2)
        
        #out = torch.sigmoid(out)
        #out = torch.sigmoid(local_out)
        #out = torch.sigmoid(distal_out)
        
        # Set the weight as a Parameter when adding local and distal
        #out = torch.sigmoid(local_out) * torch.sigmoid(self.w_ld) + torch.sigmoid(distal_out)*(1-torch.sigmoid(self.w_ld)) 
        
        return out
    
    # Do prediction using batches in DataLoader to save memory 
    def batch_predict(self, dataloader, criterion, device):
 
        self.eval()
        pred_y = torch.empty(0, self.n_class).to(device)
        
        total_loss = 0

        with torch.no_grad():
            for y, cont_x, cat_x, distal_x in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                distal_x = distal_x.to(device)
                y  = y.to(device)
        
                preds = self.forward((cont_x, cat_x), distal_x)
                pred_y = torch.cat((pred_y, preds), dim=0)
                
                loss = criterion(preds, y.long().squeeze())
                total_loss += loss.item()

        return pred_y, total_loss

#====
class Network4m(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order, n_class, emb_padding_idx=None):
        
        super(Network4m, self).__init__()
        
        self.n_class = n_class
        # FeedForward layers for local input
        # Embedding layers
        print('emb_dims: ', emb_dims)
        print('emb_padding_idx: ', emb_padding_idx)
        
        #self.emb_layers = nn.ModuleList([nn.Embedding(x, y, padding_idx = emb_padding_idx) for x, y in emb_dims])
        self.emb_layers = nn.ModuleList([nn.Embedding(emb_padding_idx+1, y, padding_idx = emb_padding_idx) for x, y in emb_dims])

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
        second_kernel_size = kernel_size
        third_kernel_size = kernel_size
        rb1_kernel_size = 3
        rb2_kernel_size = 5
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            #nn.ReLU(),
        )
        
        self.RBs1 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb1_kernel_size, stride=1, padding=(rb1_kernel_size-1)//2, dilation=1) for x in range(4)])
            

        self.maxpool1 = nn.MaxPool1d(maxpool_kernel_size, maxpool_stride) # kernel_size, stride
        
        self.conv2 = nn.Sequential(    
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, second_kernel_size),
            #nn.ReLU(),
        )
        
        self.RBs2 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb2_kernel_size, stride=1, padding=(rb2_kernel_size-1)//2, dilation=1) for x in range(4)])

        self.maxpool2 = nn.MaxPool1d(4, 4)
    
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, third_kernel_size),
            nn.ReLU(),
        )
        
        
        # RNN layers
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            #self.rnn = nn.LSTM(out_channels*2, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            self.rnn = nn.LSTM(out_channels, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            #fc_in_size = RNN_hidden_size*2 + lin_layer_sizes[-1]
            crnn_fc_in_size = RNN_hidden_size*2
        else:
            #fc_in_size = out_channels + lin_layer_sizes[-1]
            #crnn_fc_in_size = out_channels*2
            crnn_fc_in_size = out_channels
            
            # Use the flattened output of CNN instead of torch.max
            last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
            last_seq_len = (last_seq_len - (second_kernel_size-1) )//2 # For the 2nd conv1d
            
            #crnn_fc_in_size = out_channels*last_seq_len
        
        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(crnn_fc_in_size),
            nn.Dropout(0.25), #control overfitting
            nn.Linear(crnn_fc_in_size, n_class), 
            #nn.ReLU(),
            
            #nn.Linear(crnn_fc_in_size, 1),
            #nn.Linear(out_channels*2, 1),
            #nn.BatchNorm1d(30),
            #nn.Dropout(0.25), #dropout prob
            #nn.Linear(30, 1),
            #nn.Dropout(0.1)
        )
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], n_class), 
        )       

        self.combined_fc = nn.Sequential(
            nn.BatchNorm1d(n_class*2),
            #nn.ReLU(),
            nn.Linear(n_class, n_class), 
        )

        # Learn the weight parameter 
        self.w_ld = torch.nn.Parameter(torch.Tensor([0]))
        
    
    def forward(self, local_input, distal_input):
        
        # FeedForward layers for local input
        cont_data, cat_data = local_input
        if self.no_of_embs != 0:
            local_out = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            
        
        local_out = torch.cat(local_out, dim = 1) #x.shape: batch_size * sum(emb_size)
        local_out = self.emb_dropout_layer(local_out)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                local_out = torch.cat([local_out, normalized_cont_data], dim = 1) 
            else:
                local_out = normalized_cont_data
        
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            #print('local_out.shape: ', local_out.shape)
            #print('local_out: ', local_out[0, 0:10])
            local_out = F.relu(lin_layer(local_out))
            local_out = bn_layer(local_out)
            local_out = dropout_layer(local_out)
        
        # CNN layers for distal_input
        # Input data shape: batch_size, in_channels, L_in (lenth of sequence)
        #distal_out = self.conv1(distal_input) #out_shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)
        
        jump_input = distal_out = self.conv1(distal_input)
        distal_out = self.RBs1(distal_out)    
        assert(jump_input.shape[2] >= distal_out.shape[2])
        distal_out = distal_out + jump_input[:,:,0:distal_out.shape[2]]
        distal_out = self.maxpool1(distal_out)
        
        jump_input = distal_out = self.conv2(distal_out)
        distal_out = self.RBs2(distal_out)
        assert(jump_input.shape[2] >= distal_out.shape[2])
        distal_out = distal_out + jump_input[:,:,0:distal_out.shape[2]]
        distal_out = self.maxpool2(distal_out)
        
        distal_out = self.conv3(distal_out)
        #d = x.shape[2] - out.shape[2]
        #out = x[:,:,0:x.shape[2]-d] + out
        
        #out, _ = torch.max(out, dim=2)
        #print("out.shape")
        #print(out.shape)

        # RNN after CNN
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            distal_out = distal_out.permute(2,0,1)
            distal_out, _ = self.rnn(distal_out) # output of shape (seq_len, batch, num_directions * hidden_size)
            Fwd_RNN=distal_out[-1, :, :self.RNN_hidden_size] # output of last position
            Rev_RNN=distal_out[0, :, self.RNN_hidden_size:] # output of last position
            distal_out = torch.cat([Fwd_RNN, Rev_RNN], dim=1)
        else:
            
            distal_out, _ = torch.max(distal_out, dim=2)
            
            # Use flattened layer instead of torchmax
            #distal_out = distal_out.view(distal_out.shape[0], -1)
        #print("RNN out.shape")
        #print(distal_out.shape)        
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00005*local_out.shape[0] and self.training == False:
            print('local_out1:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            print('local_out2:', torch.min(local_out[:,2]).item(), torch.max(local_out[:,2]).item(), torch.var(local_out[:,2]).item(), torch.var(F.softmax(local_out, dim=1)[:,2]).item())
            print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())
        
        #out = local_out * torch.sigmoid(distal_out)
        #out = local_out * torch.exp(distal_out)
        #out = local_out + distal_out
        
        #out = torch.cat([local_out, distal_out], dim=1)
        
        #out = torch.sigmoid(local_out + distal_out)
        
        #out = (torch.sigmoid(local_out) + torch.sigmoid(distal_out))/2
        
        #out = self.combined_fc(torch.cat([F.softmax(local_out, dim=1), F.softmax(distal_out, dim=1)], dim=1))
        
        #For cross entropy loss
        out = self.combined_fc((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2)
        
        #out = F.log_softmax(out, dim=1)
        
        #out = torch.log((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2)
        
        #out = torch.sigmoid(out)
        #out = torch.sigmoid(local_out)
        #out = torch.sigmoid(distal_out)
        
        # Set the weight as a Parameter when adding local and distal
        #out = torch.sigmoid(local_out) * torch.sigmoid(self.w_ld) + torch.sigmoid(distal_out)*(1-torch.sigmoid(self.w_ld)) 
        
        return out
    
    # Do prediction using batches in DataLoader to save memory 
    def batch_predict(self, dataloader, criterion, device):
 
        self.eval()
        pred_y = torch.empty(0, self.n_class).to(device)
        
        total_loss = 0

        with torch.no_grad():
            for y, cont_x, cat_x, distal_x in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                distal_x = distal_x.to(device)
                y  = y.to(device)
        
                preds = self.forward((cont_x, cat_x), distal_x)
                pred_y = torch.cat((pred_y, preds), dim=0)
                
                loss = criterion(preds, y.long().squeeze())
                total_loss += loss.item()

        return pred_y, total_loss

    
    
class AttentionModule_stage1(nn.Module):

    def __init__(self, in_channels, out_channels, seq_len=101):
        super(AttentionModule_stage1, self).__init__()
        self.seq_len = seq_len
        
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            #ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 8*8
        
        #new_seq_len = (seq_len + 2*padding - kernel_size + stride)//stride
        seq_len1 = (self.seq_len + 2 * 1 - (3 - 2))//2

        self.down_residual_blocks1 = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 4*4
        seq_len2 = (seq_len1 + 2 * 1 - (3 - 2))//2

        self.middle_2r_blocks = nn.Sequential(
            #ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation1 = nn.Upsample(size=seq_len1)  # 8*8

        self.up_residual_blocks1 = ResidualBlock(in_channels, out_channels)

        self.interpolation2 = nn.Upsample(size=self.seq_len)  # 16*16

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            #nn.BatchNorm1d(out_channels),
            #nn.ReLU(inplace=True),
            #nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_down_residual_blocks1 = self.down_residual_blocks1(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_down_residual_blocks1)
        out_mpool2 = self.mpool2(out_down_residual_blocks1)
        out_middle_2r_blocks = self.middle_2r_blocks(out_mpool2)
        #
        out_interp = self.interpolation1(out_middle_2r_blocks) + out_down_residual_blocks1
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp + out_skip1_connection
        out_up_residual_blocks1 = self.up_residual_blocks1(out)
        #out_interp2 = self.interpolation2(out_up_residual_blocks1) + out_trunk # not needed?
        out_interp2 = self.interpolation2(out_up_residual_blocks1)
        out_conv1_1_blocks = self.conv1_1_blocks(out_interp2)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.last_blocks(out)

        return out_last

class AttentionModule_stage2(nn.Module):
    # input size is 8*8
    def __init__(self, in_channels, out_channels, seq_len):
        super(AttentionModule_stage2, self).__init__()
        
        self.seq_len = seq_len
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            #ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        #new_seq_len = (seq_len + 2*padding - kernel_size + stride)//stride
        seq_len1 = (self.seq_len + 2 * 1 - (3 - 2))//2

        self.middle_2r_blocks = nn.Sequential(
            #ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation1 = nn.Upsample(size=self.seq_len)

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            #nn.BatchNorm1d(out_channels),
            #nn.ReLU(inplace=True),
            #nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_middle_2r_blocks = self.middle_2r_blocks(out_mpool1)
        #
        out_interp = self.interpolation1(out_middle_2r_blocks) + out_trunk
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out_conv1_1_blocks = self.conv1_1_blocks(out_interp)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.last_blocks(out)

        return out_last    

    
class AttentionModule_stage3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentionModule_stage3, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            #ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.middle_2r_blocks = nn.Sequential(
            #ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            #nn.BatchNorm1d(out_channels),
            #nn.ReLU(inplace=True),
            #nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_middle_2r_blocks = self.middle_2r_blocks(x)
        #
        out_conv1_1_blocks = self.conv1_1_blocks(out_middle_2r_blocks)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.last_blocks(out)

        return out_last
    
class ResidualAttionNetwork3m(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order, n_class):
        
        super(ResidualAttionNetwork3m, self).__init__()
        
        self.n_class = n_class
        self.seq_len = distal_radius*2+1 - (distal_order-1)
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
        #self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        #===========================================================
        # CNN layers for distal input
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        #self.mpool1 = nn.MaxPool1d(kernel_size=5, stride=3, padding=2)  
        self.mpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        #new_seq_len = (seq_len + 2*padding - kernel_size + stride)//stride
        seq_len1 = (self.seq_len + 2 * 1 - (3 - 2))//2
        
        self.residual_block1 = ResidualBlock(out_channels, out_channels)
        self.attention_module1 = AttentionModule_stage1(out_channels, out_channels, seq_len1)  
        self.residual_block2 = ResidualBlock(out_channels, out_channels)  
        #seq_len2 = (seq_len1 + 3)//4
        self.mpool2 = nn.MaxPool1d(kernel_size=5, stride=3, padding=1)
        seq_len2 = (seq_len1 + 2 * 1 - (5 - 3))//3
        
        self.attention_module2 = AttentionModule_stage2(out_channels, out_channels, seq_len2) 
        self.attention_module2_2 = AttentionModule_stage2(out_channels, out_channels, seq_len2) 
        self.residual_block3 = ResidualBlock(out_channels, out_channels)   
        #seq_len3 = (seq_len2 + 3)//4
        self.mpool3 = nn.MaxPool1d(kernel_size=7, stride=5, padding=1)
        seq_len3 = (seq_len2 + 2 * 1 - (7 - 5))//5
        
        self.attention_module3 = AttentionModule_stage3(out_channels, out_channels)
        self.attention_module3_2 = AttentionModule_stage3(out_channels, out_channels) 
        self.attention_module3_3 = AttentionModule_stage3(out_channels, out_channels) 
        self.residual_block4 = ResidualBlock(out_channels, out_channels) 
        #self.residual_block5 = ResidualBlock(out_channels*16, out_channels*16)
        #self.residual_block6 = ResidualBlock(out_channels*16, out_channels*16)  
        self.avgpool = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=7, stride=3)
        )
        
        self.distal_fc = nn.Linear(out_channels, n_class)
        #=========================================================
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], n_class), 
        )       
        

        # Learn the weight parameter 
        self.w_ld = torch.nn.Parameter(torch.Tensor([0]))
        
    
    def forward(self, local_input, distal_input):
        
        # FeedForward layers for local input
        cont_data, cat_data = local_input
        if self.no_of_embs != 0:
            local_out = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            
        
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
        #====================================================================
        distal_out = self.conv1(distal_input)
        distal_out = self.mpool1(distal_out)
        # print(out.data)
        distal_out = self.residual_block1(distal_out)
        distal_out = self.attention_module1(distal_out)
        distal_out = self.residual_block2(distal_out)
        distal_out = self.mpool2(distal_out)
        distal_out = self.attention_module2(distal_out)
        #distal_out = self.attention_module2_2(distal_out)
        distal_out = self.residual_block3(distal_out)
        ##distal_out = self.mpool3(distal_out)
        # print(out.data)
        ##distal_out = self.attention_module3(distal_out)
        #distal_out = self.attention_module3_2(distal_out)
        #distal_out = self.attention_module3_3(distal_out)
        ##distal_out = self.residual_block4(distal_out)
        #distal_out = self.residual_block5(distal_out)
        #distal_out = self.residual_block6(distal_out)
        distal_out = self.avgpool(distal_out)
        #print('1. distal_out.shape:', distal_out.shape)
        #distal_out = distal_out.view(distal_out.size(0), -1)
        distal_out, _ = torch.max(distal_out, dim=2)
        #print('2. distal_out.shape:', distal_out.shape)   
        
        distal_out = self.distal_fc(distal_out)
        #====================================================================
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        #distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00005*local_out.shape[0] and self.training == False:
            print('local_out:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
        
        
        out = torch.log((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2)
        
        #out = torch.log(F.softmax(distal_out, dim=1))
        
        # Set the weight as a Parameter when adding local and distal
        #out = torch.sigmoid(local_out) * torch.sigmoid(self.w_ld) + torch.sigmoid(distal_out)*(1-torch.sigmoid(self.w_ld)) 
        
        return out
    
    # Do prediction using batches in DataLoader to save memory 
    def batch_predict(self, dataloader, criterion, device):
 
        self.eval()
        pred_y = torch.empty(0, self.n_class).to(device)
        
        total_loss = 0

        with torch.no_grad():
            for y, cont_x, cat_x, distal_x in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                distal_x = distal_x.to(device)
                y  = y.to(device)
        
                preds = self.forward((cont_x, cat_x), distal_x)
                pred_y = torch.cat((pred_y, preds), dim=0)
                
                loss = criterion(preds, y.long().squeeze())
                total_loss += loss.item()

        return pred_y, total_loss

class ResidualAttionNetwork4m(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order, n_class):
        
        super(ResidualAttionNetwork4m, self).__init__()
        
        self.n_class = n_class
        self.seq_len = distal_radius*2+1 - (distal_order-1)
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
        #self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        #===========================================================
        # CNN layers for distal input
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        #self.mpool1 = nn.MaxPool1d(kernel_size=5, stride=3, padding=2)  
        self.mpool1 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        #new_seq_len = (seq_len + 2*padding - kernel_size + stride)//stride
        seq_len1 = (self.seq_len + 2 * 1 - (4 - 2))//2
        
        self.residual_block1 = ResidualBlock(out_channels, out_channels*4)
        self.attention_module1 = AttentionModule_stage1(out_channels*4, out_channels*4, seq_len1)  
        self.residual_block2 = ResidualBlock(out_channels*4, out_channels*8, 4)  
        seq_len2 = (seq_len1 + 3)//4
        self.attention_module2 = AttentionModule_stage2(out_channels*8, out_channels*8, seq_len2) 
        self.attention_module2_2 = AttentionModule_stage2(out_channels*8, out_channels*8, seq_len2) 
        self.residual_block3 = ResidualBlock(out_channels*8, out_channels*16, 4)
        
        seq_len3 = (seq_len2 + 3)//4
        self.attention_module3 = AttentionModule_stage3(out_channels*16, out_channels*16)
        self.attention_module3_2 = AttentionModule_stage3(out_channels*16, out_channels*16) 
        self.attention_module3_3 = AttentionModule_stage3(out_channels*16, out_channels*16) 
        self.residual_block4 = ResidualBlock(out_channels*16, out_channels*32) 
        #self.residual_block5 = ResidualBlock(out_channels*16, out_channels*16)
        #self.residual_block6 = ResidualBlock(out_channels*16, out_channels*16)  
        self.mpool2 = nn.Sequential(
            nn.BatchNorm1d(out_channels*32),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=4, stride=2)
        )
        
        self.distal_fc = nn.Linear(out_channels*32, n_class)
        #=========================================================
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], n_class), 
        )       
        

        # Learn the weight parameter 
        self.w_ld = torch.nn.Parameter(torch.Tensor([0]))
        
    
    def forward(self, local_input, distal_input):
        
        # FeedForward layers for local input
        cont_data, cat_data = local_input
        if self.no_of_embs != 0:
            local_out = [emb_layer(cat_data[:, i]) for i,emb_layer in enumerate(self.emb_layers)]
            
        
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
        #====================================================================
        distal_out = self.conv1(distal_input)
        distal_out = self.mpool1(distal_out)
        # print(out.data)
        distal_out = self.residual_block1(distal_out)
        distal_out = self.attention_module1(distal_out)
        distal_out = self.residual_block2(distal_out)
        distal_out = self.attention_module2(distal_out)
        #distal_out = self.attention_module2_2(distal_out)
        distal_out = self.residual_block3(distal_out)
        # print(out.data)
        distal_out = self.attention_module3(distal_out)
        #distal_out = self.attention_module3_2(distal_out)
        #distal_out = self.attention_module3_3(distal_out)
        distal_out = self.residual_block4(distal_out)
        #distal_out = self.residual_block5(distal_out)
        #distal_out = self.residual_block6(distal_out)
        distal_out = self.mpool2(distal_out)
        #print('1. distal_out.shape:', distal_out.shape)
        #distal_out = distal_out.view(distal_out.size(0), -1)
        distal_out, _ = torch.max(distal_out, dim=2)
        #print('2. distal_out.shape:', distal_out.shape)   
        
        distal_out = self.distal_fc(distal_out)
        #====================================================================
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        #distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00005*local_out.shape[0] and self.training == False:
            print('local_out:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
        
        
        out = torch.log((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2)
        
        #out = torch.log(F.softmax(distal_out, dim=1))
        
        # Set the weight as a Parameter when adding local and distal
        #out = torch.sigmoid(local_out) * torch.sigmoid(self.w_ld) + torch.sigmoid(distal_out)*(1-torch.sigmoid(self.w_ld)) 
        
        return out
    
    # Do prediction using batches in DataLoader to save memory 
    def batch_predict(self, dataloader, criterion, device):
 
        self.eval()
        pred_y = torch.empty(0, self.n_class).to(device)
        
        total_loss = 0

        with torch.no_grad():
            for y, cont_x, cat_x, distal_x in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                distal_x = distal_x.to(device)
                y  = y.to(device)
        
                preds = self.forward((cont_x, cat_x), distal_x)
                pred_y = torch.cat((pred_y, preds), dim=0)
                
                loss = criterion(preds, y.long().squeeze())
                total_loss += loss.item()

        return pred_y, total_loss

# Residual block (according to Jaganathan et al. 2019 Cell)
class ResBlock(nn.Module):

    def __init__(self, in_channels=32, kernel_size=3, stride=1, padding=0, dilation=1):
        super(ResBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        #self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)        
        
        self.layer = nn.Sequential(nn.ReLU(),self.bn1, self.conv1, nn.ReLU(), self.bn2, self.conv2)

    def forward(self, x):
        out = self.layer(x)
       #print('out.shape, x.shape:', out.shape, x.shape)
        d = x.shape[2] - out.shape[2]
        out = x[:,:,0:x.shape[2]-d] + out
        
        #out += x
        
        return out

# Residual block ('bottleneck' version)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels//4, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm1d(out_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels//4, out_channels//4, 3, stride, padding = 1, bias = False)
        #new_seq_len = (seq_len + 2*padding - kernel_size + stride)//stride
        #seq_len1 = (self.seq_len + 2 * 1 - (3 - stride))//stride
        self.bn3 = nn.BatchNorm1d(out_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(out_channels//4, out_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv1d(in_channels, out_channels, 1, stride, bias = False)
        #new_seq_len = (seq_len + 2*padding - kernel_size + stride)//stride
        #seq_len1 = (self.seq_len + 2 * 0 - (1 - stride))//stride
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.in_channels != self.out_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out   

class BrierScore(nn.Module):
    def __init__(self):
        super(BrierScore, self).__init__()

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = F.softmax(input)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff) / float(input.shape[0])
        return loss
    
# Hybrid loss function: BCELoss + the difference between observed and predicted mutation probabilities in bins
class HybridLoss(nn.Module):
    
    def __init__(self, n_bin):
        super(HybridLoss,self).__init__()
        
        # Set the bin number for splitting the input samples
        self.n_bin = n_bin
        
    def forward(self, y_hat, y):
        
        loss_f1 = nn.BCELoss()
        loss1 = loss_f1(y_hat, y)
        
        split_size = int(np.ceil(y.shape[0]/self.n_bin))
        
        y_hat_means = []
        y_means = []
        
        # Calculate observed/predicted mutation probabilities in each bin
        for el1, el2 in zip(torch.split(y_hat, split_size), torch.split(y, split_size)):
            if el1.shape[0] == split_size and el2.shape[0] == split_size:
                y_hat_means.append(torch.mean(el1))
                y_means.append(torch.mean(el2))
        
        #y_hat_means = torch.tensor(y_hat_means, dtype=torch.float32, requires_grad=True)
        #y_means = torch.tensor(y_means, dtype=torch.float32, requires_grad=True)
        y_hat_means2 = torch.stack(y_hat_means)
        y_means2 = torch.stack(y_means)
        
        loss2 = 1 - pearsonr(y_hat_means2, y_means2)**2 # 1- R^2
        
        #print data for debugging
        if np.random.uniform(0,1) < 0.01:
            print('loss1, loss2:', loss1, loss2)
            print('y_hat_means2:', y_hat_means2)
        
        return loss1 + loss2
    
# Caculate pearsonr correlation coefficient between two pytorch Tensors
def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    
    return r_val

'''
Implementation of Focal Loss.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def get_gamma(p=0.2):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
    gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
    gamma = np.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma

'''
ps = [0.2, 0.5]
gammas = [5.0, 3.0]
i = 0
gamma_dic = {}
for p in ps:
    gamma_dic[p] = gammas[i]
    i += 1
'''

class FocalLossAdaptive(nn.Module):
    def __init__(self, gamma=0, size_average=False, device=None):
        super(FocalLossAdaptive, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.device = device

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        gamma_dic = {0.2: 5.0, 0.5: 3.0}
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        gamma = self.get_gamma_list(pt)
        loss = -1 * (1-pt)**gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


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

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.1)

    def forward(self, local_x, distal_x):
        #logits = self.model(input)
        #cont_x, cat_x = local_x
        logits = self.model.forward(local_x, distal_x)
        
        #logits = torch.cat((1-preds,preds),1)
        #logits = torch.log(logits/(1-logits))
                
        #return F.log_softmax(self.temperature_scale(logits), dim=1)
        return self.temperature_scale(logits)
        #return torch.sigmoid(self.temperature_scale(logits))[:,1].unsqueeze(1)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, dataloader, device):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        #nll_criterion = torch.nn.NLLLoss(reduction='mean').to(device)
        ece_criterion = ECELoss(n_bins=25).to(device)
        c_ece_criterion = ClasswiseECELoss(n_bins=25).to(device)
        ada_ece_criterion = AdaptiveECELoss(n_bins=15).to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            #for input, label in valid_loader:
            for y, cont_x, cat_x, distal_x in dataloader:
                #input = input.to(device)
                #logits = self.model(input)
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                distal_x = distal_x.to(device)
                y  = y.to(device)
        
                logits = self.model.forward((cont_x, cat_x), distal_x)
                #logits = torch.cat((1-preds,preds),1)
                #logits = torch.log(logits/(1-logits))
                
                logits_list.append(logits)
                labels_list.append(y.long())
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).squeeze().to(device)
        print('logits.shape:', logits.shape, labels.shape)
        print(logits, labels)
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        before_temperature_c_ece = c_ece_criterion(logits, labels).item()
        before_temperature_ada_ece = ada_ece_criterion(logits, labels).item()
        
        #print('Before temperature - NLL:', before_temperature_nll)
        print('Before temperature - NLL: %.5f, ECE: %.5f, ClassECE: %.5f, AdaECE: %.5f,' % (before_temperature_nll, before_temperature_ece, before_temperature_c_ece, before_temperature_ada_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=10000)
        print('temp optimizer:', optimizer)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_c_ece = c_ece_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ada_ece = ada_ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.5f' % self.temperature.item())
        #print('After temperature - NLL:', after_temperature_nll)
        print('After temperature - NLL: %.5f, ECE: %.5f, ClassECE: %.5f, AdaECE: %.5f,' % (after_temperature_nll, after_temperature_ece, after_temperature_c_ece, after_temperature_ada_ece))

        return self
    
    # Do prediction using batches in DataLoader to save memory 
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

'''
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
'''