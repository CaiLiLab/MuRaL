import pandas as pd
import numpy as np
import torch
import argparse
import sys


from evaluation import calibrate_prob

def parse_arguments(parser):
    """
    Parse parameters from the command line
    """

    parser.add_argument('--input_file', type=str, default='',
                        help='path for input data')

    args = parser.parse_args()

    return args

def main():
    
    #parse the command line
    parser = argparse.ArgumentParser(description='Mutation rate modeling using machine learning')
    args = parse_arguments(parser)
    
    print(' '.join(sys.argv)) # print the command line
    input_file = args.input_file
    
    df = pd.read_csv(input_file, header=None, sep='\t')
    
    device = torch.device('cuda') 
    
    fdiri_cal, fdiri_nll = calibrate_prob(df.iloc[:, 1:5].to_numpy(), df[0].to_numpy(), device, calibr_name='FullDiri')
    
    #valid_y = valid_data_and_prob['mut_type'].to_numpy().squeeze()
            
            # Train the calibrator using the validataion data
    #fdiri_cal, fdiri_nll = calibrate_prob(valid_y_prob.to_numpy(), valid_y, device, calibr_name='FullDiri')

if __name__ == '__main__':
    main()