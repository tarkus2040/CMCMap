import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.generated.multivar import MyMultivarData

from utils.causal_simplex import DCMC_simplex 

import argparse
parser = argparse.ArgumentParser("single experiment of DCMC on synthetic data")
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='3V_direct', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=1000, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level')

parser.add_argument('--tau', type=int, default=1, help="Cross mapping tau-lag")
parser.add_argument('--emd', type=int, default=3, help="Cross mapping embedding dimension")
parser.add_argument('--knn', type=int, default=4, help="Number of nearest neighbors for DCMC")

parser.add_argument('--dcmc_thres', type=float, default=0.5, help="Threshold for direct causality score")

# name of cause and effect (each is single variable, the rest are all treated as conditions)
parser.add_argument('--cause', type=str, default='X')
parser.add_argument('--effect', type=str, default='Z')

args=parser.parse_args()

# set seeds
seed=args.seed
random.seed(seed)
np.random.seed(seed)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'DCMC_test', args.causality_type, 'seed'+str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# file name/names for each type of causality
if args.noiseType!=None and args.noiseType.lower()!='none': # with noise
    if args.causality_type == '3V_direct' or args.causality_type=='4V_both_noCycle' or args.causality_type=='4V_both_Cycle':
        prefix = args.causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
        file_names = [prefix+'_1', prefix+'_2', prefix+'_3']
    elif args.causality_type == '3V_indirect' or args.causality_type=='3V_both_noCycle' or args.causality_type=='3V_both_Cycle':
        file_names = [args.causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}']
    elif args.causality_type == '4V_direct' or args.causality_type=='4V_indirect':
        prefix = args.causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
        file_names = [prefix+'_1', prefix+'_2']
    else: # beyond 4V, only one case each
        file_names = [args.causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}']
else: # no noise
    if args.causality_type == '3V_direct' or args.causality_type=='4V_both_noCycle' or args.causality_type=='4V_both_Cycle':
        prefix=args.causality_type+'_noNoise'
        file_names = [prefix+'_1', prefix+'_2', prefix+'_3']
    elif args.causality_type == '3V_indirect' or args.causality_type=='3V_both_noCycle' or args.causality_type=='3V_both_Cycle':
        file_names = [args.causality_type+'_noNoise']
    elif args.causality_type == '4V_direct' or args.causality_type=='4V_indirect':
        prefix=args.causality_type+'_noNoise'
        file_names = [prefix+'_1', prefix+'_2']
    else: # beyond 4V, only one case each
        file_names = [args.causality_type+'_noNoise']


for file_name in file_names:
    # load data
    dataset=MyMultivarData(os.path.join(root,args.data_dir,args.causality_type,file_name+'.csv'))
    df=dataset.df

    # get the name lists of cause, effect and condition
    var_list = df.columns
    list_cause=[args.cause]
    list_effect=[args.effect]
    list_conditions=[var for var in var_list if var!=args.cause and var!=args.effect]

    # Initialize DCMC
    dcmc=DCMC_simplex(df=df, causes=list_cause, effects=list_effect, cond=list_conditions, tau=args.tau, emd=args.emd, L=args.L, knn=args.knn)
    
    # DCMC causality outputs 4 values: dir_score_c2e, dir_score_e2c, cmc_score_c2e, cmc_score_e2c
    output=dcmc.causality() 

    del dcmc

    # save all outputs first as text
    file_save_name=file_name+f'_L{args.L}__tau{args.tau}_emd{args.emd}_knn{args.knn}_dcmcThres{args.dcmc_thres}'
    with open(os.path.join(save_dir, file_save_name+'_output.txt'), 'w') as f:
        f.write('dir_score_c2e, dir_score_e2c, cmc_score_c2e, cmc_score_e2c\n')
        f.write(','.join([str(x) for x in output])+'\n\n')
    np.save(os.path.join(save_dir, file_save_name+'_output.npy'), output)


    # then depend on the threshold, determine the direct causality
    dir_score_c2e = output[0]
    dir_score_e2c = output[1]

    if dir_score_c2e >= args.dcmc_thres and dir_score_e2c < args.dcmc_thres:
        result_idx = 1
        msg = 'Direct causality detected: Cause -> Effect.\n'
    elif dir_score_e2c >= args.dcmc_thres and dir_score_c2e < args.dcmc_thres:
        result_idx = 2
        msg = 'Direct causality detected: Effect -> Cause.\n'
    elif dir_score_c2e >= args.dcmc_thres and dir_score_e2c >= args.dcmc_thres:
        result_idx = 3
        msg = 'Bidirectional direct causality detected.\n'
    else:
        result_idx = 0
        msg = 'No direct causality detected.\n'

    # print the result statement to the text file
    with open(os.path.join(save_dir, file_save_name+'_conclus.txt'), 'w') as f:
        f.write(msg)

    # save the index of the result
    np.save(os.path.join(save_dir, file_save_name+'_result_idx.npy'), result_idx)