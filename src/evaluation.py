import pandas as pd
import numpy as np
from prettytable import PrettyTable


def count_parameters(model):
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
    
    
# Compare the observed and predicted frequencies of mutations in 3mers
def f3mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us1','ds1','mut_type','prob']].groupby(['us1','ds1']).mean()
    #print (obs_pred_freq[['mut_type', 'prob']])
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])

def freq_kmer_comp_multi(data_and_prob, k, n_class):
    
    d = k//2
    mer_list = ['us'+str(i) for i in list(range(1, d+1))[::-1]] + ['ds'+str(i) for i in list(range(1, d+1))]
    
    prob_list = ['prob'+str(i) for i in range(n_class)]
    
    corr_list = []
    for i in range(0, n_class):
        obs_pred_freq = pd.concat([data_and_prob[ mer_list + [prob_list[i]]], data_and_prob['mut_type']==i ], axis=1)
        
        #print('obs_pred_freq:', i, obs_pred_freq[1:5], np.sum(obs_pred_freq['mut_type']))
        obs_pred_freq = obs_pred_freq.groupby(mer_list).mean()
        #if k == 3:
        #    print('obs_pred_freq 3mer:\n', obs_pred_freq)
        
        #obs_pred_freq.columns = 
        corr_list.append(obs_pred_freq['mut_type'].corr(obs_pred_freq[prob_list[i]]))
        
    return corr_list

# Compare the observed and predicted frequencies of mutations in 5mers
def f5mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us2','us1','ds1','ds2','mut_type','prob']].groupby(['us2','us1','ds1','ds2']).mean()
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])

# Compare the observed and predicted frequencies of mutations in 7mers
def f7mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us3','us2','us1','ds1','ds2','ds3','mut_type','prob']].groupby(['us3','us2','us1','ds1','ds2', 'ds3']).mean()
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])

# Compare the frequencies of mutations in 3mers in two randomly generated datasets
def f3mer_comp_rand(df, n_rows):
    
    mean_corr = 0
    
    # Sampling 10 times
    sampling_times = 10 
    
    for i in range(sampling_times):
        freq1 = df[['us1','ds1','mut_type']].sample(n = n_rows).groupby(['us1','ds1']).mean()
        freq2 = df[['us1','ds1','mut_type']].sample(n = n_rows).groupby(['us1','ds1']).mean()
        #print(freq1, freq2)
        
        corr = freq1['mut_type'].corr(freq2['mut_type'])
        print('corr of 3mer freq1 and freq2:', corr)
        
        mean_corr += corr
    
    print('mean corr:', mean_corr/sampling_times)
    #return freq1['mut_type'].corr(freq2['mut_type'])

# Compare the frequencies of mutations in 5mers in two randomly generated datasets
def f5mer_comp_rand(df, n_rows):
    
    mean_corr = 0
    
    # Sampling 10 times
    sampling_times = 10 
   
    for i in range(sampling_times):
        freq1 = df[['us2','us1','ds1','ds2','mut_type']].sample(n = n_rows).groupby(['us2','us1','ds1','ds2']).mean()
        freq2 = df[['us2','us1','ds1','ds2','mut_type']].sample(n = n_rows).groupby(['us2','us1','ds1','ds2']).mean()
        #print(freq1, freq2)
        
        corr = freq1['mut_type'].corr(freq2['mut_type'])
        print('corr of 5mer freq1 and freq2:', corr)
        
        mean_corr += corr
    
    print('mean corr:', mean_corr/sampling_times)

# Compare the frequencies of mutations in 7mers in two randomly generated datasets

def f7mer_comp_rand(df, n_rows):
    
    mean_corr = 0
    
    # Sampling 10 times
    sampling_times = 10 
   
    for i in range(sampling_times):
        freq1 = df[['us3','us2','us1','ds1','ds2','ds3', 'mut_type']].sample(n = n_rows).groupby(['us3','us2','us1','ds1','ds2','ds3']).mean()
        freq2 = df[['us3','us2','us1','ds1','ds2','ds3','mut_type']].sample(n = n_rows).groupby(['us3','us2','us1','ds1','ds2','ds3']).mean()
        #print(freq1, freq2)
        
        corr = freq1['mut_type'].corr(freq2['mut_type'])
        print('corr of 7mer freq1 and freq2:', corr)
        
        mean_corr += corr
    
    print('mean corr:', mean_corr/sampling_times)

def corr_calc(data, window, model):
    
    obs = pred = avg_obs = avg_pred = 0
    
    count = 0
    
    site_length = len(data) ### confirm cycle time
    
    start = 0
    
    last_chrom = data.loc[0, 'chrom']
    last_start = data.loc[0, 'start']//window * window ### confirm start region
    result = pd.DataFrame(columns=('avg_obs', 'avg_pred'))
    for i in range(site_length):
        start = data.loc[i, 'start']//window * window
        chrom = data.loc[i, 'chrom']
        if chrom != last_chrom or start != last_start:
            ### calculate avg of the last region
            if obs >0 and pred>0 and obs < count:
                avg_obs = obs/count
                avg_pred = pred/count
                result = result.append(pd.DataFrame({'avg_obs':[avg_obs], 'avg_pred':[avg_pred]}))
                #if i <100:
                #    print(chrom, start, count, avg_obs, avg_pred, sep='\t')
            obs = pred = 0
            count = 0
            last_chrom = chrom
            last_start = start
            obs += data.loc[i, 'mut_type']
            pred += data.loc[i, model]
            count = count + 1
        else:
            obs += data.loc[i, 'mut_type']
            pred += data.loc[i, model]
            count = count + 1

    avg_obs = obs/count
    avg_pred = pred/count
    result = result.append(pd.DataFrame({'avg_obs':[avg_obs], 'avg_pred':[avg_pred]}))
    #print(chrom, start, count, avg_obs, avg_pred, sep='\t')
    #print('no of sites:', site_length)
    #print('result.head', result.head(5))
    #print('result.tail', result.tail(5))
    #print('result.shape', result.shape)
    if sum(list(result['avg_obs'] == 0) | (result['avg_obs'] == 1))/result.shape[0] > 0.5:
        print('Warning: too many zeros/ones in the obs windows of size', window)
    
    #print('obs==0 rows:', result.loc[result['avg_obs']==0].shape)

    corr = result['avg_obs'].corr(result['avg_pred'])
    return corr

def corr_calc_sub(data, window, prob_names):
    
    #print('in the corr_calc_sub, data.head()', data.head())
    n_class = len(prob_names)
    obs = [0]*n_class
    pred = [0]*n_class
    
    count = 0
    site_length = len(data) ### confirm cycle time
    
    start = 0  
    avg_names = []
    for i in range(n_class):
        avg_names = avg_names +['avg_obs'+str(i), 'avg_pred'+str(i)]
    
    #print('avg_names.shape:', avg_names.shape)
    last_chrom = data.loc[0, 'chrom']
    last_start = data.loc[0, 'start']//window * window ### confirm start region
    result = pd.DataFrame(columns=avg_names)
    for i in range(site_length):
        start = data.loc[i, 'start']//window * window
        chrom = data.loc[i, 'chrom']
        if chrom != last_chrom or start != last_start:
            ### calculate avg of the last region
            
            avg_list = []
            for j in range(n_class):
                avg_list += [obs[j]/count, pred[j]/count]

            result = result.append(pd.DataFrame([avg_list], columns=avg_names))
                #if i <100:
                #    print(chrom, start, count, avg_obs, avg_pred, sep='\t')
            obs = [0]*n_class
            pred = [0]*n_class
            count = 0
            last_chrom = chrom
            last_start = start
            
        #count for observed type +1
        obs[int(data.loc[i, 'mut_type'])] += 1              
        for j in range(n_class):
            pred[j] += data.loc[i, prob_names[j]]

        count = count + 1
 
    
    avg_list = []
    for j in range(n_class):
        avg_list += [obs[j]/count, pred[j]/count]

    result = result.append(pd.DataFrame([avg_list], columns=avg_names))
    
    corr_list = []
    for i in range(n_class):
        
        if sum(list(result['avg_obs'+str(i)] == 0) | (result['avg_obs'+str(i)] == 1))/result.shape[0] > 0.5:
            print('Warning: too many zeros/ones in the obs windows of size', window, 'subtype', i)
    
        #print('obs==0 rows:', result.loc[result['avg_obs']==0].shape)
        corr = result['avg_obs'+str(i)].corr(result['avg_pred'+str(i)])
        #print('corr:', corr)
        corr_list.append(corr)
    return corr_list