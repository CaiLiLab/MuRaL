import pandas as pd
import numpy as np
import torch
import argparse
import sys
from pynvml import *


from evaluation import calibrate_prob

def parse_arguments(parser):
    """
    Parse parameters from the command line
    """

    parser.add_argument('--input_file', type=str, default='',
                        help='path for input data')
    
    parser.add_argument('--calibrator', type=str, default='FullDiri', help='calibrator')
    
    parser.add_argument('--cpu_only', default=False, action='store_true',  help='only use CPU computing')
    args = parser.parse_args()

    return args

def main():
    
    #parse the command line
    parser = argparse.ArgumentParser(description='Mutation rate modeling using machine learning')
    args = parse_arguments(parser)
    
    print(' '.join(sys.argv)) # print the command line
    input_file = args.input_file
    calibrator = args.calibrator
    
    df = pd.read_csv(input_file, header=None, sep='\t')
    
    #device = torch.device('cuda') 
    if args.cpu_only:
        device = torch.device('cpu')
    else:
        # Find a GPU with enough memory
        nvmlInit()
        cuda_id = '0'
        for i in range(nvmlDeviceGetCount()):
            h = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(h)
            if info.free > 1.5*(2**30): # Reserve 1.5GB
                cuda_id = str(i)
                break

        print('CUDA: ', torch.cuda.is_available())
        print('using'  , 'cuda:'+cuda_id)
        device = torch.device('cuda:'+cuda_id if torch.cuda.is_available() else 'cpu')
    
    fdiri_cal, fdiri_nll = calibrate_prob(df.iloc[:, 1:5].to_numpy(), df[0].to_numpy(), device, calibr_name=calibrator)
    
    #fdiri_cal, fdiri_nll = calibrate_prob(df.iloc[:, 1:5].to_numpy(), df[0].to_numpy(), device, calibr_name='FullDiri1')
    
    #fdiri_cal, fdiri_nll = calibrate_prob(df.iloc[:, 1:5].to_numpy(), df[0].to_numpy(), device, calibr_name='FullDiri2')
    
    #fdiri_cal, fdiri_nll = calibrate_prob(df.iloc[:, 1:5].to_numpy(), df[0].to_numpy(), device, calibr_name='FullDiriODIR')
    
    #fvect_cal, fvect_nll = calibrate_prob(df.iloc[:, 1:5].to_numpy(), df[0].to_numpy(), device, calibr_name='VectS')
    
    #ftemp_cal, ftemp_nll = calibrate_prob(df.iloc[:, 1:5].to_numpy(), df[0].to_numpy(), device, calibr_name='TempS')


if __name__ == '__main__':
    main()
