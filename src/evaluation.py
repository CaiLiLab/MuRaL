import pandas as pd

# Compare the observed and predicted frequencies of mutations in 3mers
def f3mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us1','ds1','mut_type','prob']].groupby(['us1','ds1']).mean()
    #print (obs_pred_freq[['mut_type', 'prob']])
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])


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
    start = obs = pred = avg_obs = avg_pred = 0
    
    count = 0
    
    site_length = len(data) ### confirm cycle time
    
    last_start = data.loc[0, 'start']//window*window ### confirm start region
    result = pd.DataFrame(columns=('avg_obs', 'avg_pred'))
    for i in range(site_length):
        start = data.loc[i, 'start']//window*window
        chromosome = data.loc[i, 'chr']
        if start != last_start:
            ### calculate avg of the last region
            avg_obs = obs/count
            avg_pred = pred/count
            result = result.append(pd.DataFrame({'avg_obs':[avg_obs], 'avg_pred':[avg_pred]}))
            print('chr20', start, count, avg_obs, avg_pred, sep='\t')
            obs = pred = 0
            count = 0
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
    print('chr20', start, count, avg_obs, avg_pred, sep='\t')
    ### calculate correlation

    corr = result['avg_obs'].corr(result['avg_pred'])
    return corr