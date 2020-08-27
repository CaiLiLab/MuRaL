import pandas as pd

# compare the observed and predicted frequencies of mutations in 3mers
def f3mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us1','ds1','mut_type','prob']].groupby(['us1','ds1']).mean()
    #print (obs_pred_freq[['mut_type', 'prob']])
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])


# compare the observed and predicted frequencies of mutations in 5mers
def f5mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us2','us1','ds1','ds2','mut_type','prob']].groupby(['us2','us1','ds1','ds2']).mean()
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])

# compare the observed and predicted frequencies of mutations in 7mers
def f7mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us3','us2','us1','ds1','ds2','ds3','mut_type','prob']].groupby(['us3','us2','us1','ds1','ds2', 'ds3']).mean()
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])

# compare the frequencies of mutations in 3mers in two randomly generated datasets
def f3mer_comp_rand(df, n_rows):
    
    mean_corr = 0
    sampling_times = 10 #sampling 10 times
    
    for i in range(sampling_times):
        freq1 = df[['us1','ds1','mut_type']].sample(n = n_rows).groupby(['us1','ds1']).mean()
        freq2 = df[['us1','ds1','mut_type']].sample(n = n_rows).groupby(['us1','ds1']).mean()
        #print(freq1, freq2)
        
        corr = freq1['mut_type'].corr(freq2['mut_type'])
        print('corr of 3mer freq1 and freq2:', corr)
        
        mean_corr += corr
    
    print('mean corr:', mean_corr/sampling_times)
    #return freq1['mut_type'].corr(freq2['mut_type'])

# compare the frequencies of mutations in 5mers in two randomly generated datasets
def f5mer_comp_rand(df, n_rows):
    
    mean_corr = 0
    sampling_times = 10 #sampling 10 times
   
    for i in range(sampling_times):
        freq1 = df[['us2','us1','ds1','ds2','mut_type']].sample(n = n_rows).groupby(['us2','us1','ds1','ds2']).mean()
        freq2 = df[['us2','us1','ds1','ds2','mut_type']].sample(n = n_rows).groupby(['us2','us1','ds1','ds2']).mean()
        #print(freq1, freq2)
        
        corr = freq1['mut_type'].corr(freq2['mut_type'])
        print('corr of 5mer freq1 and freq2:', corr)
        
        mean_corr += corr
    
    print('mean corr:', mean_corr/sampling_times)

# compare the frequencies of mutations in 7mers in two randomly generated datasets

def f7mer_comp_rand(df, n_rows):
    
    mean_corr = 0
    sampling_times = 10 #sampling 10 times
   
    for i in range(sampling_times):
        freq1 = df[['us3','us2','us1','ds1','ds2','ds3', 'mut_type']].sample(n = n_rows).groupby(['us3','us2','us1','ds1','ds2','ds3']).mean()
        freq2 = df[['us3','us2','us1','ds1','ds2','ds3','mut_type']].sample(n = n_rows).groupby(['us3','us2','us1','ds1','ds2','ds3']).mean()
        #print(freq1, freq2)
        
        corr = freq1['mut_type'].corr(freq2['mut_type'])
        print('corr of 7mer freq1 and freq2:', corr)
        
        mean_corr += corr
    
    print('mean corr:', mean_corr/sampling_times)
