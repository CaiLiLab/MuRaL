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
        second_kernel_size = kernel_size//2
        third_kernel_size = kernel_size//3
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            
            ResBlock(out_channels, kernel_size=11, stride=1, padding=(11-1)//2, dilation=1),
            ResBlock(out_channels, kernel_size=11, stride=1, padding=(11-1)//2, dilation=1),
            #ResBlock(out_channels, kernel_size=11, stride=1, padding=(11-1)//2, dilation=1),
            #ResBlock(out_channels, kernel_size=11, stride=1, padding=(11-1)//2, dilation=1),
            nn.MaxPool1d(maxpool_kernel_size, maxpool_stride), # kernel_size, stride
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels*2, second_kernel_size),
            nn.ReLU(),
            
            ResBlock(out_channels*2, kernel_size=7, stride=1, padding=(7-1)//2, dilation=1),
            ResBlock(out_channels*2, kernel_size=7, stride=1, padding=(7-1)//2, dilation=1),
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
            #self.rnn = nn.LSTM(out_channels*2, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            self.rnn = nn.LSTM(out_channels*3, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
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
class Network4(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size, distal_radius, distal_order):
        
        super(Network4, self).__init__()

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

        # These variables are not used currently
        self.kernel_size = kernel_size
        self.RNN_hidden_size = RNN_hidden_size
        self.RNN_layers = RNN_layers
        self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        # CNN ResNet layers for distal input
        # Set prelayer before the ResNet Blocks
        
        #pre_chan_sizes = [48, 64]
        #res_chan_sizes = [64, 128, 200, 200]
        pre_chan_sizes = [16, 32]
        res_chan_sizes = [32, 32]
        res_kernel_sizes = [11, 7]
        
        assert(pre_chan_sizes[-1] == res_chan_sizes[0])
        in_chan_size = 4**distal_order + self.no_of_cont
        self.bn1 = nn.BatchNorm1d(in_chan_size)
        self.conv1 = nn.Conv1d(in_chan_size, pre_chan_sizes[0], kernel_size=1, stride=1)
        
        self.bn2 = nn.BatchNorm1d(pre_chan_sizes[0])
        self.conv2 = nn.Conv1d(pre_chan_sizes[0], pre_chan_sizes[1], kernel_size=1, stride=1)
        #self.bn2 = nn.BatchNorm1d(pre_chan_sizes[1])
        self.relu = nn.ReLU()
        
        self.prelayer = nn.Sequential(self.bn1, self.conv1, self.relu, self.bn2, self.conv2)
        #self.prelayer = nn.Sequential(self.bn1, self.conv1)

        # ResNet Blocks
        self.RBs1 = nn.Sequential(*[ResBlock(res_chan_sizes[0], kernel_size=res_kernel_sizes[0], stride=1, padding=(res_kernel_sizes[0]-1)//2, dilation=1) for x in range(4)])
        
        self.RBs2 = nn.Sequential(*[ResBlock(res_chan_sizes[1], kernel_size=res_kernel_sizes[1], stride=1, padding=(res_kernel_sizes[1]-1)//2, dilation=2) for x in range(4)])

        
        self.bn3 = nn.BatchNorm1d(res_chan_sizes[0])
        self.conv3 = nn.Conv1d(res_chan_sizes[0], res_chan_sizes[0], kernel_size=1, stride=1)
        
        self.bn4 = nn.BatchNorm1d(res_chan_sizes[1])
        self.conv4 = nn.Conv1d(res_chan_sizes[1], res_chan_sizes[1], kernel_size=1, stride=1)
        
        self.bn5 = nn.BatchNorm1d(res_chan_sizes[1])
        self.conv5 = nn.Conv1d(res_chan_sizes[1], res_chan_sizes[1], kernel_size=1, stride=1)
        
        self.bn6 = nn.BatchNorm1d(res_chan_sizes[1])
        self.conv6 = nn.Conv1d(res_chan_sizes[1], 1, kernel_size=1, stride=1)
        
        
        #self.conv_out_len = (((distal_radius*2+1)//3)//4 + 2)//4; reasoning: maxpooling layers reduce seq len
        #self.resnet_fc_in_size = (((distal_radius*2+1)//3 + 6)//4 + 2)//4 * res_chan_sizes[3]
 
        
        # FC layers after ResNet
        '''
        self.resnet_distal_fc = nn.Sequential(nn.BatchNorm1d(self.resnet_fc_in_size),
                                              nn.Dropout(0.1), 
                                              nn.Linear(self.resnet_fc_in_size, 1), 
                                              #nn.ReLU(), 
                                              #nn.BatchNorm1d(50),
                                              #nn.Dropout(0.1),
                                              #nn.Linear(50, 1)
                                             )
        '''
        # RNN layers
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            self.rnn = nn.LSTM(out_channels*2, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            #fc_in_size = RNN_hiddens_size*2 + lin_layer_sizes[-1]
            crnn_fc_in_size = RNN_hidden_size*2
        else:
            #fc_in_size = out_channels + lin_layer_sizes[-1]
            crnn_fc_in_size = out_channels*2
            
            # Use the flattened output of CNN instead of torch.max
            #last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
            #last_seq_len = (last_seq_len - (second_kernel_size-1) )//2 #for the 2nd conv1d
            
            #crnn_fc_in_size = out_channels*last_seq_len
        
        # Separate FC layers for distal and local 
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
            
        
        local_out = torch.cat(local_out, dim = 1) #shape: batch_size * sum(emb_size)
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
        # Input data shape: batch_size, in_channels, L_in(lenth of sequence)
        distal_out = self.prelayer(distal_input)
        
        distal_out1 = self.RBs1(distal_out)
        
        d = distal_out.shape[2] - distal_out1.shape[2]
        #print('distal_out.shape[2] - distal_out1.shape[2]:', d)
        
        #add BN and ReLU
        distal_out = self.conv3(self.bn3(F.relu(distal_out)))
        jump_out = distal_out[:,:,0:distal_out.shape[2]-d] + self.conv4(self.bn4(F.relu(distal_out1)))
        
        
        distal_out1 = self.RBs2(distal_out1)
        
        distal_out1 = self.conv5(self.bn5(F.relu(distal_out1)))
        
        d = jump_out.shape[2] - distal_out1.shape[2]
        #print('jump_out.shape[2] - distal_out1.shape[2]:', d)
        
        distal_out = self.conv6(self.bn6(F.relu(distal_out1 + jump_out[:,:,0:jump_out.shape[2]-d])))
         
        
        #distal_out = self.conv6(self.bn6(F.relu(distal_out)))

        #RNN after CNN
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            distal_out = distal_out.permute(2,0,1)
            distal_out, _ = self.rnn(distal_out) # output of shape (seq_len, batch, num_directions * hidden_size)
            Fwd_RNN=distal_out[-1, :, :self.RNN_hidden_size] # output of last position
            Rev_RNN=distal_out[0, :, self.RNN_hidden_size:] # output of last position
            distal_out = torch.cat([Fwd_RNN, Rev_RNN], dim=1)
        else:
            
            distal_out, _ = torch.max(distal_out, dim=2)
            #distal_out = torch.mean(distal_out, dim=2)
            
            #print('distal_out.shape:', distal_out.shape)
            #distal_out = distal_out.view(-1, self.resnet_fc_in_size)
                  
        
        #Separate FC layers 
        local_out = self.local_fc(local_out)
        
        #distal_out = self.distal_fc(distal_out)
        #distal_out = self.resnet_distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.01 and self.training == False:
            print('local_out:', torch.min(local_out).item(), torch.max(local_out).item(), torch.var(local_out).item(), torch.var(torch.sigmoid(local_out)).item())
            print('distal_out:', torch.min(distal_out).item(), torch.max(distal_out).item(),torch.var(distal_out).item(), torch.var(torch.sigmoid(distal_out)).item())
        
        #out = local_out * torch.sigmoid(distal_out)
        #out = local_out * torch.exp(distal_out)
        #out = local_out + distal_out
        
        #out = torch.cat([local_out, distal_out], dim=1)
        #out = self.fc(out)
        
        #out = torch.sigmoid(distal_out)
        #out = torch.sigmoid(local_out + distal_out)
        out = (torch.sigmoid(local_out) + torch.sigmoid(distal_out))/2
        #out = torch.sigmoid(out)

        
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

# Residual block
class L1Block(nn.Module):

    def __init__(self, channels=64):
        super(L1Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out

# Residual block
class L2Block(nn.Module):

    def __init__(self, channels=128):
        super(L2Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(channels, channels, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out

# Residual block
class L3Block(nn.Module):

    def __init__(self, channels=200):
        super(L3Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(channels, channels, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(channels, channels, (3, 1), stride=(1, 1), padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)

        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                   self.conv2, self.bn2, nn.ReLU(inplace=True),
                                   self.conv3, self.bn3)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out

# Residual block
class L4Block(nn.Module):

    def __init__(self, channels=200):
        super(L4Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                   self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
        #nn.init.normal_(m.weight, 0.0, 1e-06)
        nn.init.xavier_uniform_(m.weight)
        #nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(m.bias)
        
        print(m.weight.shape)
        
    elif classname.find('Linear') != -1:
        #nn.init.normal_(m.weight, 0, 0.004)
        #nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_normal_(m.weight)
            
        nn.init.normal_(m.bias)
        print(m.weight.shape)
        
    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(m.__getattr__(p))