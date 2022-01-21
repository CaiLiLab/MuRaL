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

from MuRaL.nn_utils import *
from MuRaL.evaluation import *

class FeedForwardNN(nn.Module):
    """Feedforward only model with local data"""

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, n_class, emb_padding_idx=None):
        """  
        Args:
            emb_dims: embedding dimensions
            no_of_cont: number of continuous features
            lin_layer_sizes: sizes of linear layers
            emb_dropout: dropout for the embedding layer
            lin_layer_dropouts: dropouts for linear layers
            n_class: number of classes (labels)
            emb_padding_idx: number to be used for padding in embeddings
        """
        super(FeedForwardNN, self).__init__()
        
        self.n_class = n_class
        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(emb_padding_idx+1, y, padding_idx = emb_padding_idx) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])
        
        #for lin_layer in self.lin_layers:
        #    nn.init.kaiming_normal_(lin_layer.weight.data)
        #nn.init.kaiming_normal_(m.weight)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], n_class)
        #nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):
        """
        Forward pass
        
        Args:
            cont_data: continuous data
            cat_data: categorical seq data
        """
        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]

        x = torch.cat(x, 1) #x.shape: batch_size * sum(emb_size)
        x = self.emb_dropout_layer(x)
        
        # Add the continuous features
        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1) 
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        #out = F.log_softmax(self.output_layer(x), dim=1)
        out = self.output_layer(x)

        return out
    
    def batch_predict(self, dataloader, criterion, device):
        """Do prediction using batches in DataLoader to save memory"""
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


class Network0(nn.Module):
    """Wrapper for Feedforward only model with local data"""
    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, n_class, emb_padding_idx=None):
        
        super(Network0, self).__init__()
        self.model = FeedForwardNN(emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, n_class, emb_padding_idx)
    
    def forward(self, local_input, distal_input=None):
        """Write this for using the same functional interface when doing forward pass"""
        cont_data, cat_data = local_input
        
        return self.model.forward(cont_data, cat_data)
        
       
class Network1(nn.Module):
    """ResNet-only model"""
    def __init__(self, in_channels, out_channels, kernel_size, distal_radius, distal_order, distal_fc_dropout, n_class):
        """  
        Args:
            in_channels: number of input channels
            out_channels: number of output channels after first covolution layer
            kernel_size: kernel size of first covolution layer
            distal_radius: distal radius of a focal site to be considered
            distal_order: sequece order for distal sequences
            n_class: number of classes (labels)
        """
        super(Network1, self).__init__()
        
        self.n_class = n_class     
        
        self.kernel_size = kernel_size
        self.seq_len = distal_radius * 2 + 1 - (distal_order - 1)
        
        # CNN layers for distal input
        maxpool_kernel_size = 10
        maxpool_stride = 10

        rb1_kernel_size = 3
        rb2_kernel_size = 5
        
        # 1st conv layer
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            #nn.ReLU(),
        )
        
        # 1st set of residual blocks
        self.RBs1 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb1_kernel_size, stride=1, padding=(rb1_kernel_size-1)//2, dilation=1) for x in range(4)])    

        self.maxpool1 = nn.MaxPool1d(maxpool_kernel_size, maxpool_stride) # kernel_size, stride    
        self.conv2 = nn.Sequential(    
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            #nn.ReLU(),
        )
        
        # 2nd set of residual blocks
        self.RBs2 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb2_kernel_size, stride=1, padding=(rb2_kernel_size-1)//2, dilation=1) for x in range(4)])

        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.ReLU(),
        )
        
        cnn_fc_in_size = out_channels

        # Calculate the current seq len
        last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
        last_seq_len = (last_seq_len - (kernel_size-1) )//2 # For the 2nd conv1d
            
        
        # last FC layers for distal data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(cnn_fc_in_size),
            nn.Dropout(distal_fc_dropout), 
            nn.Linear(cnn_fc_in_size, n_class), 
            #nn.ReLU(),
      
        )     
        
    
    def forward(self, local_input, distal_input):
        """
        Forward pass
        
        Args:
            local_input: local input
            distal_input: distal input
        """
            
        # CNN layers for distal_input
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
        distal_out, _ = torch.max(distal_out, dim=2)
 
        distal_out = self.distal_fc(distal_out)
        
        # Print some data for debugging
        if np.random.uniform(0,1) < 0.00001*distal_out.shape[0] and self.training == False:
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())
        
        
        return distal_out
 
class Network2(nn.Module):
    """Combined model with FeedForward and ResNet componets"""
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, distal_radius, distal_order, distal_fc_dropout, n_class, emb_padding_idx=None):
        """  
        Args:
            emb_dims: embedding dimensions
            no_of_cont: number of continuous features
            lin_layer_sizes: sizes of linear layers
            emb_dropout: dropout for the embedding layer
            lin_layer_dropouts: dropouts for linear layers            
            in_channels: number of input channels
            out_channels: number of output channels after first covolution layer
            kernel_size: kernel size of first covolution layer
            distal_radius: distal radius of a focal site to be considered
            distal_order: sequece order for distal sequences
            distal_fc_dropout: dropout for distal fc layer
            n_class: number of classes (labels)
            emb_padding_idx: number to be used for padding in embeddings
        """
        
        super(Network2, self).__init__()
        
        self.n_class = n_class
        
        # FeedForward layers for local input
        # Embedding layers
        print('emb_dims: ', emb_dims)
        print('emb_padding_idx: ', emb_padding_idx)
        
        self.emb_layers = nn.ModuleList([nn.Embedding(emb_padding_idx+1, y, padding_idx = emb_padding_idx) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])
        
        self.kernel_size = kernel_size
        self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        # CNN layers for distal input
        maxpool_kernel_size = 10
        maxpool_stride = 10

        rb1_kernel_size = 3
        rb2_kernel_size = 5
        
        # 1st conv layer
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            #nn.ReLU(),
        )
        
        # 1st set of residual blocks
        self.RBs1 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb1_kernel_size, stride=1, padding=(rb1_kernel_size-1)//2, dilation=1) for x in range(4)])
            

        self.maxpool1 = nn.MaxPool1d(maxpool_kernel_size, maxpool_stride) # kernel_size, stride  
        self.conv2 = nn.Sequential(    
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            #nn.ReLU(),
        )
        
        # 2nd set of residual blocks
        self.RBs2 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb2_kernel_size, stride=1, padding=(rb2_kernel_size-1)//2, dilation=1) for x in range(4)])

        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.ReLU(),
        )
        
        cnn_fc_in_size = out_channels

        # calculate the current sequence len
        last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
        last_seq_len = (last_seq_len - (kernel_size-1) )//2 # For the 2nd conv1d

        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(cnn_fc_in_size),
            nn.Dropout(distal_fc_dropout), 
            nn.Linear(cnn_fc_in_size, n_class), 
            #nn.ReLU(),
            
        )
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], n_class), 
        )       
    
    def forward(self, local_input, distal_input):
        """
        Forward pass
        
        Args:
            local_input: local input
            distal_input: distal input
        """
        
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
        jump_input = distal_out = self.conv1(distal_input) #output shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)   
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
        distal_out, _ = torch.max(distal_out, dim=2)    
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00001*local_out.shape[0] and self.training == False:
            print('local_out1:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            #print('local_out2:', torch.min(local_out[:,2]).item(), torch.max(local_out[:,2]).item(), torch.var(local_out[:,2]).item(), torch.var(F.softmax(local_out, dim=1)[:,2]).item())
            #print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())

        
        out = torch.log((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2) 
        
        return out
     
    def batch_predict(self, dataloader, criterion, device):
        """Do prediction using batches in DataLoader to save memory"""
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

class Network3(nn.Module):
    """Combined model with FeedForward and ResNet componets"""
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, distal_radius, distal_order, distal_fc_dropout, n_class, emb_padding_idx=None):
        """  
        Args:
            emb_dims: embedding dimensions
            no_of_cont: number of continuous features
            lin_layer_sizes: sizes of linear layers
            emb_dropout: dropout for the embedding layer
            lin_layer_dropouts: dropouts for linear layers            
            in_channels: number of input channels
            out_channels: number of output channels after first covolution layer
            kernel_size: kernel size of first covolution layer
            distal_radius: distal radius of a focal site to be considered
            distal_order: sequece order for distal sequences
            distal_fc_dropout: dropout for distal fc layer
            n_class: number of classes (labels)
            emb_padding_idx: number to be used for padding in embeddings
        """
        
        super(Network3, self).__init__()
        
        self.n_class = n_class
        
        # FeedForward layers for local input
        # Embedding layers
        print('emb_dims: ', emb_dims)
        print('emb_padding_idx: ', emb_padding_idx)
        
        self.emb_layers = nn.ModuleList([nn.Embedding(emb_padding_idx+1, y, padding_idx = emb_padding_idx) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])
        
        self.kernel_size = kernel_size
        self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        # CNN layers for distal input
        maxpool_kernel_size = 10
        maxpool_stride = 10

        rb1_kernel_size = 3
        rb2_kernel_size = 5
        
        # 1st conv layer
        self.conv1 = nn.Sequential(
            #nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        
        # 1st set of residual blocks
        self.RBs1 = nn.Sequential(*[ResBlock2(out_channels, kernel_size=rb1_kernel_size, stride=1, padding=(rb1_kernel_size-1)//2, dilation=1) for x in range(3)])
            

        self.maxpool1 = nn.MaxPool1d(maxpool_kernel_size, maxpool_stride) # kernel_size, stride  
        self.conv2 = nn.Sequential(    
            #nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        
        # 2nd set of residual blocks
        self.RBs2 = nn.Sequential(*[ResBlock2(out_channels, kernel_size=rb2_kernel_size, stride=1, padding=(rb2_kernel_size-1)//2, dilation=1) for x in range(3)])

        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.conv3 = nn.Sequential(
            #nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        
        cnn_fc_in_size = out_channels

        # calculate the current sequence len
        last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
        last_seq_len = (last_seq_len - (kernel_size-1) )//2 # For the 2nd conv1d

        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(cnn_fc_in_size),
            nn.Dropout(distal_fc_dropout), 
            nn.Linear(cnn_fc_in_size, n_class), 
            #nn.ReLU(),
            
        )
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], n_class), 
        )  
        
        self.out_fc = nn.Sequential(
            nn.BatchNorm1d(2*n_class),
            #nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(2*n_class, n_class), 
        ) 
    
    def forward(self, local_input, distal_input):
        """
        Forward pass
        
        Args:
            local_input: local input
            distal_input: distal input
        """
        
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
            local_out = F.relu(bn_layer(lin_layer(local_out)))
            #local_out = bn_layer(local_out)
            local_out = dropout_layer(local_out)
        
        # CNN layers for distal_input
        # Input data shape: batch_size, in_channels, L_in (lenth of sequence)
        jump_input = distal_out = self.conv1(distal_input) #output shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)   
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
        distal_out, _ = torch.max(distal_out, dim=2)    
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00001*local_out.shape[0] and self.training == False:
            print('local_out1:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            #print('local_out2:', torch.min(local_out[:,2]).item(), torch.max(local_out[:,2]).item(), torch.var(local_out[:,2]).item(), torch.var(F.softmax(local_out, dim=1)[:,2]).item())
            #print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())

        out = torch.cat((local_out, distal_out), dim=1)
        out = self.out_fc(out)
        
        #out = torch.log((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2) 
        
        return out
     
    def batch_predict(self, dataloader, criterion, device):
        """Do prediction using batches in DataLoader to save memory"""
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
    """Residual block unit"""
    def __init__(self, in_channels=32, kernel_size=3, stride=1, padding=0, dilation=1):
        super(ResBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)  
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)        
        
        self.layer = nn.Sequential(nn.ReLU(),self.bn1, self.conv1, nn.ReLU(), self.bn2, self.conv2)

    def forward(self, x):
        out = self.layer(x)
        #print('out.shape, x.shape:', out.shape, x.shape)
        d = x.shape[2] - out.shape[2]
        out = x[:,:,0:x.shape[2]-d] + out
        
        return out

class ResBlock2(nn.Module):
    """Residual block unit"""
    def __init__(self, in_channels=32, kernel_size=3, stride=1, padding=0, dilation=1):
        super(ResBlock2, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)  
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)        
        
        self.layer = nn.Sequential(self.bn1, nn.ReLU(),self.conv1, self.bn2, nn.ReLU(), self.conv2)

    def forward(self, x):
        out = self.layer(x)
        #print('out.shape, x.shape:', out.shape, x.shape)
        d = x.shape[2] - out.shape[2]
        out = x[:,:,0:x.shape[2]-d] + out
        
        return out
    
# Residual block ('bottleneck' version)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """Residual block ('bottleneck' version)"""
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


class MuTransformer(nn.Module):
    """ResNet-only model"""
    def __init__(self, in_channels, out_channels, kernel_size, distal_radius, distal_order, distal_fc_dropout, n_class, nhead, dim_feedforward, trans_dropout, num_layers):
        """  
        Args:
            in_channels: number of input channels
            out_channels: number of output channels after first covolution layer
            kernel_size: kernel size of first covolution layer
            distal_radius: distal radius of a focal site to be considered
            distal_order: sequece order for distal sequences
            n_class: number of classes (labels)
        """
        super(MuTransformer, self).__init__()
        
        print("Using Transformer ...")
        
        self.n_class = n_class     
        
        self.kernel_size = kernel_size
        self.seq_len = distal_radius * 2 + 1 - (distal_order - 1)
        
        # 1st conv layer
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2), # in_channels, out_channels, kernel_size
            #nn.ReLU(),
        )
        
        self.pos_encoder = PositionalEncoding(
            d_model=out_channels,
            dropout=trans_dropout,
            max_len=self.seq_len,
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=trans_dropout,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        ) 
        
        # Separate FC layers for distal and local data
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Dropout(distal_fc_dropout), 
            nn.Linear(out_channels, n_class), 
            #nn.ReLU(),
            
        )
        
        self.d_model = out_channels
        
    
    def forward(self, local_input, distal_input):
        """
        Forward pass
        
        Args:
            local_input: local input
            distal_input: distal input
        """
        
        x = self.conv1(distal_input) #output shape: batch_size, out_channels, L_out
        x = x.permute(2, 0, 1)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        
        return x
        
'''
class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        print("x.shape, pe.shape", x.shape, self.pe.shape)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
'''

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Network5(nn.Module):
    """Combined model with FeedForward and ResNet componets"""
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, distal_radius, distal_order, distal_fc_dropout, n_class,  nhead, dim_feedforward, trans_dropout, num_layers, emb_padding_idx=None):
        """  
        Args:
            emb_dims: embedding dimensions
            no_of_cont: number of continuous features
            lin_layer_sizes: sizes of linear layers
            emb_dropout: dropout for the embedding layer
            lin_layer_dropouts: dropouts for linear layers            
            in_channels: number of input channels
            out_channels: number of output channels after first covolution layer
            kernel_size: kernel size of first covolution layer
            distal_radius: distal radius of a focal site to be considered
            distal_order: sequece order for distal sequences
            distal_fc_dropout: dropout for distal fc layer
            n_class: number of classes (labels)
            emb_padding_idx: number to be used for padding in embeddings
        """
        
        super(Network5, self).__init__()
        
        self.n_class = n_class
        
        # FeedForward layers for local input
        # Embedding layers
        print('emb_dims: ', emb_dims)
        print('emb_padding_idx: ', emb_padding_idx)
        
        self.emb_layers = nn.ModuleList([nn.Embedding(emb_padding_idx+1, y, padding_idx = emb_padding_idx) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])
        
        self.kernel_size = kernel_size
        self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        # CNN layers for distal input
        maxpool_kernel_size = 10
        maxpool_stride = 10

        rb1_kernel_size = 3
        rb2_kernel_size = 5
        
        # 1st conv layer
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            #nn.ReLU(),
        )
        
        # 1st set of residual blocks
        self.RBs1 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb1_kernel_size, stride=1, padding=(rb1_kernel_size-1)//2, dilation=1) for x in range(4)])
            

        self.maxpool1 = nn.MaxPool1d(maxpool_kernel_size, maxpool_stride) # kernel_size, stride  
        self.conv2 = nn.Sequential(    
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            #nn.ReLU(),
        )
        
        # 2nd set of residual blocks
        self.RBs2 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb2_kernel_size, stride=1, padding=(rb2_kernel_size-1)//2, dilation=1) for x in range(4)])

        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.ReLU(),
        )
        
        cnn_fc_in_size = out_channels

        # calculate the current sequence len
        last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
        last_seq_len = last_seq_len - (kernel_size-1) #For the 2nd conv1d
        #last_seq_len = (last_seq_len -(4-4))//4 # For maxpool2
        #last_seq_len = last_seq_len - (kernel_size-1) #For the 3rd conv1d
        
        print('last_seq_len:', last_seq_len)

        ########
        self.pos_encoder = PositionalEncoding(
            d_model=out_channels,
            dropout=trans_dropout,
            max_len=last_seq_len,
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=trans_dropout,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        ) 
        
        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Dropout(distal_fc_dropout), 
            nn.Linear(out_channels, n_class), 
            #nn.ReLU(),
            
        )
        
        self.distal_fc2 = nn.Sequential(
            nn.BatchNorm1d(out_channels*last_seq_len),
            nn.Dropout(distal_fc_dropout), 
            nn.Linear(out_channels*last_seq_len, n_class), 
            #nn.ReLU(),
            
        )
        
        self.d_model = out_channels
        ########
        '''
        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(cnn_fc_in_size),
            nn.Dropout(distal_fc_dropout), 
            nn.Linear(cnn_fc_in_size, n_class), 
            #nn.ReLU(),
            
        )
        '''
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], n_class), 
        )
        
        self.out_fc = nn.Sequential(
            nn.BatchNorm1d(2*n_class),
            #nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(2*n_class, n_class), 
        ) 
    
    def forward(self, local_input, distal_input):
        """
        Forward pass
        
        Args:
            local_input: local input
            distal_input: distal input
        """
        
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
        jump_input = distal_out = self.conv1(distal_input) #output shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)   
        distal_out = self.RBs1(distal_out)    
        assert(jump_input.shape[2] >= distal_out.shape[2])
        distal_out = distal_out + jump_input[:,:,0:distal_out.shape[2]]    
        distal_out = self.maxpool1(distal_out)
        
        #print('distal_out.shape1:', distal_out.shape)
        
        jump_input = distal_out = self.conv2(distal_out)    
        distal_out = self.RBs2(distal_out)
        assert(jump_input.shape[2] >= distal_out.shape[2])
        distal_out = distal_out + jump_input[:,:,0:distal_out.shape[2]]
        #print('distal_out.shape2:', distal_out.shape)
        
        #distal_out = self.maxpool2(distal_out)
        #distal_out = self.conv3(distal_out)
        
        #distal_out, _ = torch.max(distal_out, dim=2)
        
        ##############
        #print('distal_out.shape3:', distal_out.shape)
        distal_out = distal_out.permute(2, 0, 1)
        distal_out = distal_out * math.sqrt(self.d_model)
        distal_out = self.pos_encoder(distal_out)
        distal_out = self.transformer_encoder(distal_out)
        #distal_out = distal_out.mean(dim=0)
        #distal_out, _ = torch.max(distal_out, dim=0)
        #distal_out = self.distal_fc(distal_out)
        
        distal_out = distal_out.permute(1, 2, 0)
        #print('distal_out.shape4:', distal_out.shape)
        distal_out = distal_out.reshape(distal_out.shape[0], -1)
        distal_out = self.distal_fc2(distal_out)
        ##############
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        #distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00001*local_out.shape[0] and self.training == False:
            print('local_out1:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            #print('local_out2:', torch.min(local_out[:,2]).item(), torch.max(local_out[:,2]).item(), torch.var(local_out[:,2]).item(), torch.var(F.softmax(local_out, dim=1)[:,2]).item())
            #print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())

        out = torch.cat((local_out, distal_out), dim=1)
        out = self.out_fc(out)
        
        #out = torch.log((F.softmax(local_out, dim=1) + F.softmax(local_out, dim=1))/2) 
        
        #return distal_out
        return out
     
    def batch_predict(self, dataloader, criterion, device):
        """Do prediction using batches in DataLoader to save memory"""
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

class Network6(nn.Module):
    """Combined model with FeedForward and ResNet componets"""
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, distal_radius, distal_order, distal_fc_dropout, n_class, emb_padding_idx=None):
        """  
        Args:
            emb_dims: embedding dimensions
            no_of_cont: number of continuous features
            lin_layer_sizes: sizes of linear layers
            emb_dropout: dropout for the embedding layer
            lin_layer_dropouts: dropouts for linear layers            
            in_channels: number of input channels
            out_channels: number of output channels after first covolution layer
            kernel_size: kernel size of first covolution layer
            distal_radius: distal radius of a focal site to be considered
            distal_order: sequece order for distal sequences
            distal_fc_dropout: dropout for distal fc layer
            n_class: number of classes (labels)
            emb_padding_idx: number to be used for padding in embeddings
        """
        
        super(Network6, self).__init__()
        
        self.n_class = n_class
        
        # FeedForward layers for local input
        # Embedding layers
        print('emb_dims: ', emb_dims)
        print('emb_padding_idx: ', emb_padding_idx)
        
        self.emb_layers = nn.ModuleList([nn.Embedding(emb_padding_idx+1, y, padding_idx = emb_padding_idx) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])
        
        self.kernel_size = kernel_size
        self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        # CNN layers for distal input
        maxpool_kernel_size = 10
        maxpool_stride = 10

        rb1_kernel_size = 3
        rb2_kernel_size = 5
        
        # 1st conv layer
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, kernel_size*5, stride=5), # in_channels, out_channels, kernel_size
            #nn.ReLU(),
        )
        
        # 1st set of residual blocks
        self.RBs1 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb1_kernel_size, stride=1, padding=(rb1_kernel_size-1)//2, dilation=1) for x in range(4)])
            

        self.maxpool1 = nn.MaxPool1d(maxpool_kernel_size, maxpool_stride) # kernel_size, stride  
        self.conv2 = nn.Sequential(    
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            #nn.ReLU(),
        )
        
        # 2nd set of residual blocks
        self.RBs2 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb2_kernel_size, stride=1, padding=(rb2_kernel_size-1)//2, dilation=1) for x in range(4)])

        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.ReLU(),
        )
        
        cnn_fc_in_size = out_channels

        # calculate the current sequence len
        last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
        last_seq_len = (last_seq_len - (kernel_size-1) )//2 # For the 2nd conv1d

        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(cnn_fc_in_size),
            nn.Dropout(distal_fc_dropout), 
            nn.Linear(cnn_fc_in_size, n_class), 
            #nn.ReLU(),
            
        )
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], n_class), 
        )       
    
    def forward(self, local_input, distal_input):
        """
        Forward pass
        
        Args:
            local_input: local input
            distal_input: distal input
        """
        
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
        jump_input = distal_out = self.conv1(distal_input) #output shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)   
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
        distal_out, _ = torch.max(distal_out, dim=2)    
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00001*local_out.shape[0] and self.training == False:
            print('local_out1:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            #print('local_out2:', torch.min(local_out[:,2]).item(), torch.max(local_out[:,2]).item(), torch.var(local_out[:,2]).item(), torch.var(F.softmax(local_out, dim=1)[:,2]).item())
            #print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())

        
        out = torch.log((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2) 
        
        return out
     
    def batch_predict(self, dataloader, criterion, device):
        """Do prediction using batches in DataLoader to save memory"""
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

class Network7(nn.Module):
    """Combined model with FeedForward and ResNet componets"""
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, distal_radius, distal_order, distal_fc_dropout, n_class, emb_padding_idx=None):
        """  
        Args:
            emb_dims: embedding dimensions
            no_of_cont: number of continuous features
            lin_layer_sizes: sizes of linear layers
            emb_dropout: dropout for the embedding layer
            lin_layer_dropouts: dropouts for linear layers            
            in_channels: number of input channels
            out_channels: number of output channels after first covolution layer
            kernel_size: kernel size of first covolution layer
            distal_radius: distal radius of a focal site to be considered
            distal_order: sequece order for distal sequences
            distal_fc_dropout: dropout for distal fc layer
            n_class: number of classes (labels)
            emb_padding_idx: number to be used for padding in embeddings
        """
        
        super(Network7, self).__init__()
        
        self.n_class = n_class
        
        # FeedForward layers for local input
        # Embedding layers
        print('emb_dims: ', emb_dims)
        print('emb_padding_idx: ', emb_padding_idx)
        
        self.emb_layers = nn.ModuleList([nn.Embedding(emb_padding_idx+1, y, padding_idx = emb_padding_idx) for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])
        
        self.kernel_size = kernel_size
        self.seq_len = distal_radius*2+1 - (distal_order-1)
        
        # CNN layers for distal input
        maxpool_kernel_size = 10
        maxpool_stride = 10

        rb1_kernel_size = 3
        rb2_kernel_size = 5
        
        # 1st conv layer
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channels), # This is important!
            nn.Conv1d(in_channels, out_channels, 25, stride=10), # in_channels, out_channels, kernel_size
            #nn.ReLU(),
        )
        
        # 1st set of residual blocks
        self.RBs1 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb1_kernel_size, stride=1, padding=(rb1_kernel_size-1)//2, dilation=1) for x in range(4)])
            

        self.maxpool1 = nn.MaxPool1d(maxpool_kernel_size, maxpool_stride) # kernel_size, stride  
        self.conv2 = nn.Sequential(    
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            #nn.ReLU(),
        )
        
        # 2nd set of residual blocks
        self.RBs2 = nn.Sequential(*[ResBlock(out_channels, kernel_size=rb2_kernel_size, stride=1, padding=(rb2_kernel_size-1)//2, dilation=1) for x in range(4)])

        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.ReLU(),
        )
        
        cnn_fc_in_size = out_channels

        # calculate the current sequence len
        last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
        last_seq_len = (last_seq_len - (kernel_size-1) )//2 # For the 2nd conv1d

        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(cnn_fc_in_size),
            nn.Dropout(distal_fc_dropout), 
            nn.Linear(cnn_fc_in_size, n_class), 
            #nn.ReLU(),
            
        )
        
        # Local FC layers
        self.local_fc = nn.Sequential(
            #nn.BatchNorm1d(lin_layer_sizes[-1]),
            #nn.Dropout(0.15),
            nn.Linear(lin_layer_sizes[-1], n_class), 
        )       
    
    def forward(self, local_input, distal_input):
        """
        Forward pass
        
        Args:
            local_input: local input
            distal_input: distal input
        """
        
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
        jump_input = distal_out = self.conv1(distal_input) #output shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)   
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
        distal_out, _ = torch.max(distal_out, dim=2)    
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00001*local_out.shape[0] and self.training == False:
            print('local_out1:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            #print('local_out2:', torch.min(local_out[:,2]).item(), torch.max(local_out[:,2]).item(), torch.var(local_out[:,2]).item(), torch.var(F.softmax(local_out, dim=1)[:,2]).item())
            #print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())

        
        out = torch.log((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2) 
        
        return out
     
    def batch_predict(self, dataloader, criterion, device):
        """Do prediction using batches in DataLoader to save memory"""
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