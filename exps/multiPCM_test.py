import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.generated.multivar import MyMultivarData
from utils.causal_simplex import PCM_simplex

import argparse
parser = argparse.ArgumentParser("single experiment of PCM on synthetic data")
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='3V_direct_CMC', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=1000, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level')

parser.add_argument('--tau', type=int, default=1, help="Cross mapping tau-lag")
parser.add_argument('--emd', type=int, default=16, help="Cross mapping embedding dimension")
parser.add_argument('--knn', type=int, default=10, help="Number of nearest neighbors for PCM")

parser.add_argument('--score_type', type=str, default='corr', help="Options are err or corr or r2")
parser.add_argument('--pcm_thres', type=float, default=0.45, help="Threshold for PCM")

# name of cause and effect (each is single variable, the rest are all treated as conditions)
parser.add_argument('--cause', type=str, default='Y')
parser.add_argument('--effect', type=str, default='Z')

args=parser.parse_args()

def determine_condition(output, score_type, threshold):
    ratio_index = {'err': 2, 'corr': 5, 'r2': 8}[score_type]
    ratio = output[ratio_index]
    if 0.95 < ratio < 1.05:
        return 0
    if score_type == 'err':
        return int(ratio >= threshold)
    return int(ratio <= threshold)

# set seeds
seed=args.seed
random.seed(seed)
np.random.seed(seed)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'multiPCM_test', args.causality_type, 'seed'+str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# file name/names for each type of causality
# 3V_direct has 3 structures, denoted as _1, _2, _3
# 3V_indirect, 3V_both_noCycle and 3V_both_Cycle have only 1 structure, no specification in file names
# 4V_direct, 4V_indirect has 2 structures, denoted as _1, _2
# 4V_both_noCycle and 4V_both_Cycle have 3 structures, denoted as _1, _2, _3 (pas encore fait - note Oct.9)

# get file name list
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
    list_conditions=[var for var in var_list if var!=args.cause and var!=args.effect]

    outputs = []
    for cause, effect in [(args.cause, args.effect), (args.effect, args.cause)]:
        pcm=PCM_simplex(
            df=df,
            causes=[cause],
            effects=[effect],
            cond=list_conditions,
            tau=args.tau,
            emd=args.emd,
            L=args.L,
            knn=args.knn,
        )
        outputs.append(pcm.causality())
        del pcm

    # (threshold independent) save all outputs first as text
    file_save_name=file_name+f'_L{args.L}__tau{args.tau}_emd{args.emd}_knn{args.knn}_pcmThres{args.pcm_thres}'
    metric_index = {'err': 0, 'corr': 3, 'r2': 6}[args.score_type]
    metric_name = {'err': 'Error', 'corr': 'Correlation', 'r2': 'R2'}[args.score_type]
    result_indices = [determine_condition(output, args.score_type, args.pcm_thres) for output in outputs]
    conclusions = [
        'The other variables are condition.' if result_idx else 'The other variables are not condition.'
        for result_idx in result_indices
    ]

    with open(os.path.join(save_dir, file_save_name+'_output.txt'), 'w') as f:
        f.write(f'Dataset: {file_name}\n')
        f.write(f'Variables: {", ".join(var_list)}\n')
        f.write(f'Condition variables: {", ".join(list_conditions)}\n')
        f.write(f'Score type: {args.score_type}  Threshold: {args.pcm_thres}\n\n')
        f.write(f'{"Direction":<20} {"Direct":>14} {"Conditioned":>14} {"Ratio":>14}  Conclusion\n')
        for (cause, effect), output, conclusion in zip(
            [(args.cause, args.effect), (args.effect, args.cause)], outputs, conclusions
        ):
            f.write(
                f'{cause + " -> " + effect:<20} {output[metric_index]:>14.6g} '
                f'{output[metric_index + 1]:>14.6g} {output[metric_index + 2]:>14.6g}  '
                f'{conclusion}\n'
            )
    np.save(os.path.join(save_dir, file_save_name+'_output.npy'), outputs[0])
    np.save(os.path.join(save_dir, file_save_name+'_reverse_output.npy'), outputs[1])

    with open(os.path.join(save_dir, file_save_name+'_conclus.txt'), 'w') as f:
        for (cause, effect), conclusion in zip(
            [(args.cause, args.effect), (args.effect, args.cause)], conclusions
        ):
            f.write(f'{cause} -> {effect}: {conclusion}\n')

    np.save(os.path.join(save_dir, file_save_name+'_result_idx.npy'), result_indices[0])
    np.save(os.path.join(save_dir, file_save_name+'_reverse_result_idx.npy'), result_indices[1])

    print(f'\nデータ: {file_name}')
    print(f'指標: {metric_name}  しきい値: {args.pcm_thres}')
    print(f'{"方向":<12} {"直接":>14} {"条件付き":>14} {"比率":>14}')
    for (cause, effect), output, result_idx in zip(
        [(args.cause, args.effect), (args.effect, args.cause)], outputs, result_indices
    ):
        print(
            f'{cause} -> {effect:<7} {output[metric_index]:>14.6g} '
            f'{output[metric_index + 1]:>14.6g} {output[metric_index + 2]:>14.6g}'
        )
        print(f'  判定: {"他の変数は条件" if result_idx else "他の変数は条件ではない"}')
    print(f'保存先: {os.path.join(save_dir, file_save_name)}')
