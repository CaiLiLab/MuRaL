import pandas as pd
import numpy as np
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.stats.stats import pearsonr

# Import warnings filter
from warnings import simplefilter
# Ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import jax
jax.config.update('jax_platform_name', 'cpu')

from dirichletcal.calib.vectorscaling import VectorScaling
from dirichletcal.calib.tempscaling import TemperatureScaling
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from sklearn.multiclass import OneVsRestClassifier

def count_parameters(model):
    """Count parameters in a network model"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    
    print(table)
    print(f"Total Trainable Params: {total_params}")
    
    return total_params
    
def f3mer_comp(data_and_prob):
    """Compare the observed and predicted frequencies of mutations in 3-mers"""
    obs_pred_freq = data_and_prob[['us1','ds1','mut_type','prob']].groupby(['us1','ds1']).mean()
    
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])

def freq_kmer_comp_multi(data_and_prob, k, n_class):
    """Compare the observed and predicted frequencies of mutations in 3-mers"""
    
    # Generate the column names
    d = k//2
    mer_list = ['us'+str(i) for i in list(range(1, d+1))[::-1]] + ['ds'+str(i) for i in list(range(1, d+1))]
    
    prob_list = ['prob'+str(i) for i in range(n_class)]
    
    corr_list = []
    for i in range(0, n_class):
        obs_pred_freq = pd.concat([data_and_prob[ mer_list + [prob_list[i]]], data_and_prob['mut_type']==i ], axis=1)
        
        # Get average rates for each k-mer
        obs_pred_freq = obs_pred_freq.groupby(mer_list).mean()
        
        # Calcuate correlations 
        corr_list.append(obs_pred_freq['mut_type'].astype(float).corr(obs_pred_freq[prob_list[i]].astype(float)))
        
    return corr_list

def f3mer_comp_rand(df, n_rows):
    """Compare the frequencies of mutations in 3-mers in two randomly generated datasets"""
    mean_corr = 0
    
    # Sampling 10 times
    sampling_times = 10 
    
    for i in range(sampling_times):
        freq1 = df[['us1','ds1','mut_type']].sample(n = n_rows).groupby(['us1','ds1']).mean()
        freq2 = df[['us1','ds1','mut_type']].sample(n = n_rows).groupby(['us1','ds1']).mean()
        #print(freq1, freq2)
        
        corr = freq1['mut_type'].astype(float).corr(freq2['mut_type'].astype(float))
        print('corr of 3mer freq1 and freq2:', corr)
        
        mean_corr += corr
    
    print('mean corr:', mean_corr/sampling_times)

def f5mer_comp_rand(df, n_rows):
    """Compare the frequencies of mutations in 5-mers in two randomly generated datasets"""
    mean_corr = 0
    
    # Sampling 10 times
    sampling_times = 10 
   
    for i in range(sampling_times):
        freq1 = df[['us2','us1','ds1','ds2','mut_type']].sample(n = n_rows).groupby(['us2','us1','ds1','ds2']).mean()
        freq2 = df[['us2','us1','ds1','ds2','mut_type']].sample(n = n_rows).groupby(['us2','us1','ds1','ds2']).mean()
        
        corr = freq1['mut_type'].astype(float).corr(freq2['mut_type'].astype(float))
        print('corr of 5mer freq1 and freq2:', corr)
        
        mean_corr += corr
    
    print('mean corr:', mean_corr/sampling_times)

def f7mer_comp_rand(df, n_rows):
    """Compare the frequencies of mutations in 7mers in two randomly generated datasets"""
    mean_corr = 0
    
    # Sampling 10 times
    sampling_times = 10 
   
    for i in range(sampling_times):
        freq1 = df[['us3','us2','us1','ds1','ds2','ds3', 'mut_type']].sample(n = n_rows).groupby(['us3','us2','us1','ds1','ds2','ds3']).mean()
        freq2 = df[['us3','us2','us1','ds1','ds2','ds3','mut_type']].sample(n = n_rows).groupby(['us3','us2','us1','ds1','ds2','ds3']).mean()
        
        corr = freq1['mut_type'].astype(float).corr(freq2['mut_type'].astype(float))
        print('corr of 7mer freq1 and freq2:', corr)
        
        mean_corr += corr
    
    print('mean corr:', mean_corr/sampling_times)

def corr_calc_sub(data, window, prob_names):
    """Calculate regional correlations"""
    n_class = len(prob_names)
    obs = [0]*n_class
    pred = [0]*n_class
    
    count = 0
    n_sites = len(data) 
    
    avg_names = []
    for i in range(n_class):
        avg_names = avg_names +['avg_obs'+str(i), 'avg_pred'+str(i)]
    
    last_chrom = data.loc[0, 'chrom']
    last_start = data.loc[0, 'start']//window * window # Find the window start
    
    result = pd.DataFrame(columns=avg_names)
    for i in range(n_sites):
        # First, find the corresponding window
        start = data.loc[i, 'start']//window * window
        chrom = data.loc[i, 'chrom']
        
        if chrom != last_chrom or start != last_start:
            # Calculate avg of the last region  
            avg_list = []
            for j in range(n_class):
                avg_list += [obs[j]/count, pred[j]/count]

            result = result.append(pd.DataFrame([avg_list], columns=avg_names))

            obs = [0]*n_class
            pred = [0]*n_class
            count = 0
            last_chrom = chrom
            last_start = start
            
        # Count for observed type +1
        obs[int(data.loc[i, 'mut_type'])] += 1              
        
        # Add to the cumulative mutation probs
        for j in range(n_class):
            pred[j] += data.loc[i, prob_names[j]]

        count = count + 1
 
    # Add the data of last window
    avg_list = []
    for j in range(n_class):
        avg_list += [obs[j]/count, pred[j]/count]
    result = result.append(pd.DataFrame([avg_list], columns=avg_names))
    
    # Calculate correlation for each mutation subtype
    corr_list = []
    for i in range(n_class):  
        if sum(list(result['avg_obs'+str(i)] == 0) | (result['avg_obs'+str(i)] == 1))/result.shape[0] > 0.5:
            print('Warning: too many zeros/ones (>50%) in the obs windows of size', window, 'subtype', i)
        CV_obs = result['avg_obs'+str(i)].std()/result['avg_obs'+str(i)].mean()
        CV_pred = result['avg_pred'+str(i)].std()/result['avg_pred'+str(i)].mean()
        print('CV for ', str(window)+'bp:', CV_obs, CV_pred)
        
        #corr = result['avg_obs'+str(i)].astype(float).corr(result['avg_pred'+str(i)]).astype(float)
        #from scipy.stats.stats import pearsonr 
        if result.shape[0] >= 3:
            corr = pearsonr(result['avg_obs'+str(i)], result['avg_pred'+str(i)])[0]
        else:
            corr = 0
            print('Warning: too few windows for calculating correlation', window, 'subtype', i)
        corr_list.append(corr)
    
    return corr_list

def calc_avg_prob(df, n_class):
    
    avg_list = []
    for i in range(n_class):
        avg_list.append(sum(list(df['mut_type'] == i)) / df.shape[0])

    for i in range(n_class):
        avg_list.append(df['prob'+str(i)].mean())
        
    return avg_list
        
                        
class ECELoss(nn.Module):
    """
    Compute ECE (Expected Calibration Error)
    
    Use code from https://github.com/torrvision/focal_calibration
    """
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
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

class ClasswiseECELoss(nn.Module):
    """
    Compute Classwise ECE
    
    Use code from https://github.com/torrvision/focal_calibration
    """
    def __init__(self, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            
            # one-hot vector of all positions where the label belongs to the class i
            labels_in_class = labels.eq(i) 

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce

class BrierScore(nn.Module):
    """implementation of Brier score using nn.Module (not used)"""
    def __init__(self):
        super(BrierScore, self).__init__()

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(input.shape).to(target.device)
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = F.softmax(input, dim=1)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff) / float(input.shape[0])
        return loss
    
def calibrate_prob(y_prob, y, device, calibr_name='FullDiri'):
    """
    Fit the calibrator
    
    Use calibrators in dirichletcal package
    """
    
    if calibr_name == 'VectS':
        calibr = VectorScaling(logit_constant=0.0)
    elif calibr_name == 'TempS':
        calibr = TemperatureScaling(logit_constant=0.0)
    elif calibr_name == 'FullDiri':
        #calibr = FullDirichletCalibrator(optimizer='fmin_l_bfgs_b')
        calibr = FullDirichletCalibrator()
    elif calibr_name == 'FullDiriODIR':
        l2_odir = 1e-2
        calibr = FullDirichletCalibrator(reg_lambda=l2_odir, reg_mu=l2_odir)
    elif calibr_name == 'FullDiri1':
        calibr = FullDirichletCalibrator(reg_norm=True)
    elif calibr_name == 'FullDiri2':
        calibr = FullDirichletCalibrator(ref_row=False)
    
    # Fit the calibrator
    calibr.fit(y_prob, y)
    prob_cal = calibr.predict_proba(y_prob)
    print('y_prob.head():', y_prob[0:6,])
    print('y:', y[0:6])
    print('prob_cal:', prob_cal[0:6, ])
    print('calibr.coef_: ', calibr.coef_)
    print('calibr.weights_:', calibr.weights_)
    print("prob_cal.min:", prob_cal.min(axis=0))
    print("prob_cal.max:", prob_cal.max(axis=0))
    print("CV:", y_prob.std(axis=0)/y_prob.mean(axis=0))
    print("CV (after calibration):", prob_cal.std(axis=0)/prob_cal.mean(axis=0))
    
    #######
    #OvR = OneVsRestClassifier(FullDirichletCalibrator()).fit(y_prob, y)
    #prob_cal1 = OvR.predict_proba(y_prob)
    #print('prob_cal1:', prob_cal1[0:6, ])
    #######
    
    
    nll_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    ece_criterion = ECELoss(n_bins=50).to(device)
    c_ece_criterion = ClasswiseECELoss(n_bins=50).to(device)
    brier_criterion = BrierScore().to(device)

    # Generate pseudo-logits
    logits0 = torch.log(torch.from_numpy(y_prob)).to(device)
    logits = torch.log(torch.from_numpy(np.copy(prob_cal))).to(device)
    
    labels = torch.from_numpy(y).long().to(device)
    
    # Calculate metrics before and after calibration
    nll0 = nll_criterion(logits0, labels).item()
    nll = nll_criterion(logits, labels).item()
    ece0 = ece_criterion(logits0, labels).item()
    ece = ece_criterion(logits, labels).item()
    c_ece0 = c_ece_criterion(logits0, labels).item()
    c_ece = c_ece_criterion(logits, labels).item()
    
    brier0 = brier_criterion(logits0, labels).item()
    brier = brier_criterion(logits, labels).item()
    
    print('Before ' + calibr_name + ' scaling - NLL: %.8f, ECE: %.8f, CwECE: %.8f, Brier: %.8f' % (nll0, ece0, c_ece0, brier0))
    print('After ' + calibr_name +  ' scaling - NLL: %.8f, ECE: %.8f, CwECE: %.8f, Brier: %.8f' % (nll, ece, c_ece, brier))

    
    return calibr, nll

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

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

class CBLoss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, loss_type="sigmoid", beta=0.9999, gamma=1):
        super(CBLoss, self).__init__() 
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
    def forward(self, logits, labels):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.to(logits.device) #### device
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss       


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss
