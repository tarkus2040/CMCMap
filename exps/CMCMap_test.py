import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import time
import numpy as np

from data_loaders.generated.multivar import MyMultivarData
from utils.CMCMap import CMCMap

import argparse
parser = argparse.ArgumentParser("single experiment of CMCMap on simulated data")
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='6V_both_Cycle', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=197, help='random seed, for sampling a random start point for input time series')
parser.add_argument('--L', type=int, default=4000, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level')

parser.add_argument('--tau', type=int, default=2, help="Cross mapping tau-lag")
parser.add_argument('--emd', type=int, default=6, help="Cross mapping embedding dimension")
parser.add_argument('--knn', type=int, default=10, help="Number of nearest neighbors")
parser.add_argument('--kNN_model', type=str, default='vanilla', choices=['PCA', 'vanilla'])
parser.add_argument('--pca_dim', type=int, default=3, help="Number of PCA components")

parser.add_argument('--bivCMC_thres', type=float, default=0.5, help="Minimum CMC score for creating a candidate edge")
parser.add_argument('--dcmc_thres', type=float, default=0.5, help="Minimum DCMC score for retaining a direct edge")

args = parser.parse_args()

# set seeds
random.seed(args.seed)
np.random.seed(args.seed)

# folder to store outputs
save_dir = os.path.join(
    root,
    'outputs',
    'CMCMap_test',
    args.causality_type,
    'seed' + str(args.seed),
    f'tau_{args.tau}_emd_{args.emd}',
)
if args.kNN_model == 'PCA':
    save_dir = os.path.join(save_dir, f'pca_dim_{args.pca_dim}')
os.makedirs(save_dir, exist_ok=True)

# determine file names for the selected causality structure and noise setting
if args.noiseType is not None and args.noiseType.lower() != 'none':
    noise_suffix = f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
else:
    noise_suffix = '_noNoise'

if args.causality_type in ('3V_direct', '4V_both_noCycle', '4V_both_Cycle'):
    file_names = [f'{args.causality_type}{noise_suffix}_{i}' for i in range(1, 4)]
elif args.causality_type in ('4V_direct', '4V_indirect'):
    file_names = [f'{args.causality_type}{noise_suffix}_{i}' for i in range(1, 3)]
else:
    file_names = [f'{args.causality_type}{noise_suffix}']

for file_name in file_names:
    file_dir = os.path.join(save_dir, file_name)
    print(f'\nStarting dataset: {file_name}', flush=True)
    dataset = MyMultivarData(
        os.path.join(root, args.data_dir, args.causality_type, file_name + '.csv')
    )
    df = dataset.df

    model_kwargs = {
        'knn': args.knn,
        'L': args.L,
        'method': args.kNN_model,
    }
    if args.kNN_model == 'PCA':
        model_kwargs['pca_dim'] = args.pca_dim

    model = CMCMap(
        df,
        tau=args.tau,
        emd=args.emd,
        bivCMC_thres=args.bivCMC_thres,
        dcmc_thres=args.dcmc_thres,
        **model_kwargs,
    )

    start_time = time.time()
    ch = model.fit()
    time_spent = time.time() - start_time

    model.draw_graph(file_dir)

    print('Time spent:', time_spent)
    with open(file_dir + '_time.txt', 'w') as f:
        f.write(str(time_spent))

    print('ch:', ch)
    with open(file_dir + '_ch.txt', 'w') as f:
        f.write(str(ch))

    print('Phase 1 stats (CMC):')
    print(model.phase1_stats)
    with open(file_dir + '_phase1_stats.txt', 'w') as f:
        f.write(str(model.phase1_stats))

    print('Phase 2 stats (multiDCMC):')
    print(model.phase2_stats)
    with open(file_dir + '_phase2_stats.txt', 'w') as f:
        f.write(str(model.phase2_stats))
