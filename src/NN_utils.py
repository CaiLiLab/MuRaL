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

def load_data(data_file):
    data = pd.read_csv(data_file, sep='\t').dropna()
    seq_data = data['sequence']
    y_data = data['label'].astype(np.float32).values.reshape(-1, 1)

    seqs_ohe = seqs2ohe(seq_data, 6)

    dataset = seqDataset([seqs_ohe, y_data])
    #print(dataset[0:2][0][0][0:4,4:10])
    
    return dataset

class Network(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, RNN_hidden_size, RNN_layers, lin_layer_size):
        super(Network, self).__init__()
        #self.cnn = nn.Conv1d(in_channels, out_channels, kernel_size)
        #self.maxpool =  nn.MaxPool1d(kernel_size, stride)
        
        self.kernel_size = kernel_size
        self.RNN_hidden_size = RNN_hidden_size
        self.RNN_layers = RNN_layers
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool1d(3, 1), # kernel_size, stride
            nn.Conv1d(out_channels, out_channels*2, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(3, 1)
        )
        
        self.rnn = nn.LSTM(out_channels*2, RNN_hidden_size, num_layers=RNN_layers, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(RNN_hidden_size*2),
            nn.Linear(RNN_hidden_size*2, lin_layer_size), 
            nn.ReLU(),
            nn.Dropout(0.1), #dropout prob
            #nn.Linear(out_channels*2, 1),
            nn.Linear(lin_layer_size, 1),
            #nn.ReLU(), #NOTE: addting this makes two early convergence
            nn.Dropout(0.1)
        )
        
        #nn.init.kaiming_normal_(self.fc.weight.data) #
    
    def forward(self, input_data):
        #input data shape: batch_size, in_channels, L_in (lenth of sequence)
        out = self.conv(input_data) #out_shape: batch_size, L_out; L_out = floor((L_in+2*padding-kernel_size)/stride + 1)
        #out, _ = torch.max(out, dim=2)
        #print("out.shape")
        #print(out.shape)
        #print(out[0:5])        
        
        #RNN
        out = out.permute(2,0,1)
        out, _ = self.rnn(out) #output of shape (seq_len, batch, num_directions * hidden_size)
        Fwd_RNN=out[-1, :, :self.RNN_hidden_size] #output of last position
        Rev_RNN=out[0, :, self.RNN_hidden_size:]
        out = torch.cat((Fwd_RNN, Rev_RNN), 1)
        print("RNN out.shape")
        print(out.shape)        

        
        out = self.fc(out)
        
        return torch.sigmoid(out)

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
                    torch.nn.init.xavier_uniform(m.__getattr__(p))

def train_network(model, no_of_epochs, dataloader):
    
    #weight = torch.tensor([1,10])
    criterion = nn.BCELoss()
    #criterion = nn.NLLLoss()
    #criterion = nn.BCEWithLogitsLoss()

    #set Optimizer
    #print("model.parameters:")
    #print(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9859, nesterov=True,
    #                           weight_decay=3e-06)

    for epoch in range(no_of_epochs):
        for x_data, y in dataloader:
            x_data = x_data.to(device)
            y  = y.to(device)
        
            # Forward Pass
            #preds = model(cont_x, cat_x) #original
            preds = model(x_data)
            print("preds:")
            #print(preds[1:20])
            #print(y)
            
            loss = criterion(preds, y)
            #loss = F.binary_cross_entropy(preds,y)
            print("Loss is %.3f" %(loss))
        
            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model
    #pred_y = model.forward(test_cont_x, test_cat_x)

def eval_model(model, test_x, test_y, device):
    model.eval()
    
    model=model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    pred_y = model.forward(test_x)
    y_prob = pd.Series(data=to_np(pred_y).T[0], name="prob")
    
    print("data_and_prob:")
    #print(to_np(test_x))
    #data_and_prob = pd.concat([pd.DataFrame(to_np(test_x)), y_prob], axis=1)
    print("print test_y, pred_y:")
    print("sum(test_y) is %0.3f; sum(pred_y) is %0.3f" % (torch.sum(test_y), torch.sum(pred_y)) )
    print(to_np(test_y))
    print(to_np(pred_y))

    auc_score = metrics.roc_auc_score(to_np(test_y), to_np(pred_y)) #TO CHECK
    brier_score = metrics.brier_score_loss(to_np(test_y), to_np(pred_y))
    print ("AUC score: ", auc_score)
    print ("Brier score: ", brier_score)

def main():
    
    print("CUDA: ", torch.cuda.is_available())
    
    global device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #set train file
    train_file = sys.argv[1]
    #train_file = "/public/home/licai/DNMML/analysis/test/merge.95win.A.pos.c1000.train.NN.300k.gz"
    
    #set test file
    test_file = sys.argv[2]
    #test_file = "/public/home/licai/DNMML/analysis/test/merge.95win.A.pos.c1000.test.NN.300k.gz"
    
    dataset = load_data(train_file)
    train_x, train_y = dataset.x_data, dataset.y_data
    dataset_test = load_data(test_file)
    test_x, test_y = dataset_test.x_data, dataset_test.y_data,
    #test_x, test_y = torch.tensor(dataset_test[0].values), torch.tensor(dataset_test[1].values)

    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    
    model = Network(4, 60, 10, 30, 1, 60).to(device)
    #summary(model)
    
    model.apply(weights_init)
    print("model:")
    print(model)
    
    no_of_epochs = 5
    
    model.train()
    model = train_network(model, no_of_epochs, dataloader)

    #######
    model.to('cpu')
    device = 'cpu'
    print("for the tain data:")
    eval_model(model, train_x, train_y, device)
    
    print("for the test data:")
    eval_model(model, test_x, test_y, device)
    #print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
    #print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
    #print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
        
if __name__ == "__main__":
    main()



