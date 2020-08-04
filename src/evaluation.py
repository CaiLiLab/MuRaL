import pandas as pd

def f3mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us1','ds1','mut_type','prob']].groupby(['us1','ds1']).mean()
    #print (obs_pred_freq[['mut_type', 'prob']])
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])

def f5mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us2','us1','ds1','ds2','mut_type','prob']].groupby(['us2','us1','ds1','ds2']).mean()
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])

def f7mer_comp(data_and_prob):
    obs_pred_freq = data_and_prob[['us3','us2','us1','ds1','ds2','ds3','mut_type','prob']].groupby(['us3','us2','us1','ds1','ds2', 'ds3']).mean()
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])
