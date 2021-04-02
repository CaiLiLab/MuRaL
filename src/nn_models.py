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

from nn_utils import *
from evaluation import *

#from torchsummary import summary


class FeedForwardNN(nn.Module):

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, n_class, emb_padding_idx=None):
        
        super(FeedForwardNN, self).__init__()
        
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

####
class Network0(nn.Module):
    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, n_class, emb_padding_idx=None):
        
        super(Network0, self).__init__()
        self.model = FeedForwardNN(emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, n_class, emb_padding_idx)
    
    def forward(self, local_input, distal_input=None):
        cont_data, cat_data = local_input
        
        return self.model.forward(cont_data, cat_data)
        
#####        
class Network0r(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, last_lin_size, distal_radius, distal_order, n_class, emb_padding_idx=None):
        
        super(Network0r, self).__init__()
        
        self.n_class = n_class     
        
        self.kernel_size = kernel_size
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
        
    
        #fc_in_size = out_channels + lin_layer_sizes[-1]
        #cnn_fc_in_size = out_channels*2
        cnn_fc_in_size = out_channels

        # Use the flattened output of CNN instead of torch.max
        last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
        last_seq_len = (last_seq_len - (second_kernel_size-1) )//2 # For the 2nd conv1d
            
        
        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(cnn_fc_in_size),
            nn.Dropout(0.25), #control overfitting
            nn.Linear(cnn_fc_in_size, n_class), 
            #nn.ReLU(),
      
        )     
        

        # Learn the weight parameter 
        #self.w_ld = torch.nn.Parameter(torch.Tensor([0]))
        
    
    def forward(self, local_input, distal_input):
        
        # FeedForward layers for local input
        cont_data, cat_data = local_input
            
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
            
            # Use flattened layer instead of torchmax
            #distal_out = distal_out.view(distal_out.shape[0], -1)
 
        distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00005*distal_out.shape[0] and self.training == False:
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())
        
        
        return distal_out

# Hybrid network with feedforward and ResNet layers; the FC layers of local and distal data are separated.
class Network3m(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, last_lin_size, distal_radius, distal_order, n_class, emb_padding_idx=None):
        
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
        
        cnn_fc_in_size = out_channels

        # Use the flattened output of CNN instead of torch.max
        last_seq_len = (distal_radius*2+1 - (distal_order-1) - (kernel_size-1) - (maxpool_kernel_size-maxpool_stride))//maxpool_stride
        last_seq_len = (last_seq_len - (second_kernel_size-1) )//2 # For the 2nd conv1d

        #cnn_fc_in_size = out_channels*last_seq_len
        
        # Separate FC layers for distal and local data
        self.distal_fc = nn.Sequential(
            nn.BatchNorm1d(cnn_fc_in_size),
            nn.Dropout(0.25), #control overfitting
            nn.Linear(cnn_fc_in_size, n_class), 
            #nn.ReLU(),
            
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

        distal_out, _ = torch.max(distal_out, dim=2)    
        
        # Separate FC layers 
        local_out = self.local_fc(local_out)
        distal_out = self.distal_fc(distal_out)
        
        if np.random.uniform(0,1) < 0.00005*local_out.shape[0] and self.training == False:
            print('local_out1:', torch.min(local_out[:,1]).item(), torch.max(local_out[:,1]).item(), torch.var(local_out[:,1]).item(), torch.var(F.softmax(local_out, dim=1)[:,1]).item())
            print('distal_out1:', torch.min(distal_out[:,1]).item(), torch.max(distal_out[:,1]).item(),torch.var(distal_out[:,1]).item(), torch.var(F.softmax(distal_out, dim=1)[:,1]).item())
            print('local_out2:', torch.min(local_out[:,2]).item(), torch.max(local_out[:,2]).item(), torch.var(local_out[:,2]).item(), torch.var(F.softmax(local_out, dim=1)[:,2]).item())
            print('distal_out2:', torch.min(distal_out[:,2]).item(), torch.max(distal_out[:,2]).item(),torch.var(distal_out[:,2]).item(), torch.var(F.softmax(distal_out, dim=1)[:,2]).item())
            #print('local_out3:', torch.min(local_out[:,3]).item(), torch.max(local_out[:,3]).item(), torch.var(local_out[:,3]).item(), torch.var(F.softmax(local_out, dim=1)[:,3]).item())
            #print('distal_out3:', torch.min(distal_out[:,3]).item(), torch.max(distal_out[:,3]).item(),torch.var(distal_out[:,3]).item(), torch.var(F.softmax(distal_out, dim=1)[:,3]).item())
        
        
        out = torch.log((F.softmax(local_out, dim=1) + F.softmax(distal_out, dim=1))/2) 
        
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
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

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
        print('Before temperature - NLL: %.5f, ECE: %.5f, CwECE: %.5f, AdaECE: %.5f,' % (before_temperature_nll, before_temperature_ece, before_temperature_c_ece, before_temperature_ada_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=10000)
        #print('temp optimizer:', optimizer)

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
        print('After temperature - NLL: %.5f, ECE: %.5f, CwECE: %.5f, AdaECE: %.5f,' % (after_temperature_nll, after_temperature_ece, after_temperature_c_ece, after_temperature_ada_ece))

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
