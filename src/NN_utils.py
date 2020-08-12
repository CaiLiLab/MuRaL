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

from evaluation import f3mer_comp, f5mer_comp, f7mer_comp

from torchsummary import summary

def to_np(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

def seq2ohe(sequence,motlen):
    rows = len(sequence)+2*motlen-2
    S = np.empty([rows,4])
    base = 'ACGT'
    for i in range(rows):
        for j in range(4):
            if i-motlen+1<len(sequence) and sequence[i-motlen+1].upper() =='N' or i<motlen-1 or i>len(sequence)+motlen-2:
                S[i,j]=np.float32(0.25)
            elif sequence[i-motlen+1].upper() == base[j]:
                S[i,j]=np.float32(1)
            else:
                S[i,j]=np.float32(0)
    return np.transpose(S)

def seqs2ohe(sequences,motiflen=24):

    dataset=[]
    for row in sequences:             
        dataset.append(seq2ohe(row,motiflen))
        
  
    return dataset

class seqDataset(Dataset):
    """ Diabetes dataset."""

    def __init__(self,xy=None):
        #self.x_data = np.asarray([el for el in xy[0]],dtype=np.float32)
        self.x_data = np.asarray(xy[0], dtype=np.float32)
        #self.y_data = np.asarray([el for el in xy[1]],dtype=np.float32)
        self.y_data = np.asarray(xy[1], dtype=np.float32)
        
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        
        self.len=len(self.x_data)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class CombinedDataset(Dataset):
    """ Combined dataset."""

    def __init__(self,local_dataset, distal_dataset):
        
        self.y = local_dataset.y
        self.local_cont_X = local_dataset.cont_X
        self.local_cat_X = local_dataset.cat_X
        self.distal_X = distal_dataset.x_data
        self.len=len(self.y)

    def __getitem__(self, index):
        return self.y[index], self.local_cont_X[index], self.local_cat_X[index], self.distal_X[index]

    def __len__(self):
        return self.len
    
def gen_ohe_dataset(data):
    seq_data = data['seq']
    y_data = data['mut_type'].astype(np.float32).values.reshape(-1, 1)

    seqs_ohe = seqs2ohe(seq_data, motiflen=6)

    dataset = seqDataset([seqs_ohe, y_data])
    #print(dataset[0:2][0][0][0:4,4:10])
    
    return dataset
    
def separate_local_distal(data, radius = 5): 
    seq_len = len(data['seq'][0])
    mid_pos = int((seq_len+1)/2)

    adj_seq = pd.DataFrame([list(el[mid_pos-(radius+1):mid_pos+radius]) for el in data['seq']])
    adj_seq.columns = ['us'+str(radius - i)for i in range(radius)] + ['mid'] + ['ds'+str(i+1)for i in range(radius)]

    #local sequences and functional genomic data
    data_local = pd.concat([adj_seq, data.drop(['pos','seq'], axis=1)], axis=1)

    #consider more distal sequences
    data_distal = data[['seq', 'mut_type']]
    
    categorical_features = list(adj_seq.columns)
    #categorical_features = ["us5", "us4", "us3", "us2", "us1", "ds1", "ds2", "ds3", "ds4", "ds5"]
    
    return data_local, data_distal, categorical_features

    
class TabularDataset(Dataset):
    def __init__(self, data, cat_cols, output_col):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------

        data: pandas data frame
            The data frame object for the input data. It must
            contain all the continuous, categorical and the
            output columns to be used.

        cat_cols: List of strings
            The names of the categorical columns in the data.
            These columns will be passed through the embedding
            layers in the model. These columns must be
            label encoded beforehand. 

        output_col: string
            The name of the output variable column in the data
            provided.
        """
        #first, change labels to digits
        label_encoders = {}
        for cat_col in cat_cols:
            label_encoders[cat_col] = LabelEncoder()
            data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])
        
        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y =    np.zeros((self.n, 1))

        self.cat_cols = cat_cols
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if len(self.cat_cols) >0:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X =    np.zeros((self.n, 1))

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


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

    def batch_predict(self, dataloader, device):
        
        self.eval()
        
        pred_y = torch.empty(0, 1).to(device)

        with torch.no_grad():
            for y, cont_x, cat_x, _ in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
        
                preds = self.forward(cont_x, cat_x)
                pred_y = torch.cat((pred_y, preds), dim=0)

        return pred_y

#
def load_data(data_file):
    
    data = pd.read_csv(data_file, sep='\t').dropna()
    seq_data = data['sequence']
    y_data = data['label'].astype(np.float32).values.reshape(-1, 1)

    seqs_ohe = seqs2ohe(seq_data, 6)

    dataset = seqDataset([seqs_ohe, y_data])
    #print(dataset[0:2][0][0][0:4,4:10])
    
    return dataset

class Network(nn.Module):
    def __init__(self,  emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, last_lin_size):
        
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
        
        # CNN layers for distal input
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool1d(4, 2), # kernel_size, stride
            nn.Conv1d(out_channels, out_channels*2, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(4, 2)
        )
        
        # RNN layers
        if self.RNN_hidden_size > 0 and self.RNN_layers > 0:
            self.rnn = nn.LSTM(out_channels*2, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
            fc_in_size = RNN_hidden_size*2 + lin_layer_sizes[-1]
        else:
            fc_in_size = out_channels*2 + lin_layer_sizes[-1]
        
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
        #print("RNN out.shape")
        #print(distal_out.shape)        

        out = torch.cat([local_out, distal_out], dim=1)
        
        out = self.fc(out)
        
        return torch.sigmoid(out)
    
    # do prediction using batches in DataLoader to save memory 
    def batch_predict(self, dataloader, device):
 
        self.eval()
        pred_y = torch.empty(0, 1).to(device)

        with torch.no_grad():
            for y, cont_x, cat_x, distal_x in dataloader:
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                distal_x = distal_x.to(device)
                #y  = y.to(device)
        
                preds = self.forward((cont_x, cat_x), distal_x)
                pred_y = torch.cat((pred_y, preds), dim=0)

        return pred_y
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        #nn.init.normal_(m.weight, 0.0, 1e-06)
        nn.init.xavier_uniform_(m.weight)
        nn.init.normal_(m.bias)

        
        print(m.weight.shape)
    elif classname.find('Linear') != -1:
        #nn.init.normal_(m.weight, 0, 0.004)
        nn.init.xavier_uniform_(m.weight)

        nn.init.normal_(m.bias)
        print(m.weight.shape)
    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(m.__getattr__(p))


