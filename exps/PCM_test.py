import os
import sys
from itertools import combinations

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np

from data_loaders.generated.multivar import MyMultivarData
from utils.causal_simplex import PCM_simplex

import argparse


parser = argparse.ArgumentParser(
    "Run phase 2 (multiPCM) for all variable and conditioning combinations"
)
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument(
    '--causality_type',
    type=str,
    default='3V_both_Cycle_CMC',
    help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, '
         '3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, '
         '4V_both_noCycle',
)
parser.add_argument('--seed', type=int, default=197)
parser.add_argument('--L', type=int, default=1000)
parser.add_argument('--noiseType', type=str, default='None')
parser.add_argument('--noiseInjectType', type=str, default='add')
parser.add_argument('--noiseLevel', type=float, default=1e-2)
parser.add_argument('--tau', type=int, default=2)
parser.add_argument('--emd', type=int, default=6)
parser.add_argument('--knn', type=int, default=10)
parser.add_argument('--kNN_model', type=str, default='vanilla')
parser.add_argument('--pca_dim', type=int, default=3)
parser.add_argument('--score_type', type=str, default='corr',
                    choices=('err', 'corr', 'r2'))
parser.add_argument('--pcm_thres', type=float, default=0.45)
parser.add_argument(
    '--output_dir',
    type=str,
    default=os.path.join('outputs', 'PCM_test'),
    help='Output directory relative to the repository root unless absolute',
)
args = parser.parse_args()


def get_file_names(causality_type, noise_type, noise_inject_type, noise_level):
    if noise_type is not None and noise_type.lower() != 'none':
        if causality_type in ('3V_direct', '4V_both_noCycle', '4V_both_Cycle'):
            prefix = f'{causality_type}_{noise_type}_{noise_inject_type}_{noise_level}'
            return [f'{prefix}_{index}' for index in (1, 2, 3)]
        if causality_type in ('3V_indirect', '3V_both_noCycle', '3V_both_Cycle'):
            return [f'{causality_type}_{noise_type}_{noise_inject_type}_{noise_level}']
        if causality_type in ('4V_direct', '4V_indirect'):
            prefix = f'{causality_type}_{noise_type}_{noise_inject_type}_{noise_level}'
            return [f'{prefix}_{index}' for index in (1, 2)]
        return [f'{causality_type}_{noise_type}_{noise_inject_type}_{noise_level}']

    if causality_type in ('3V_direct', '4V_both_noCycle', '4V_both_Cycle'):
        prefix = f'{causality_type}_noNoise'
        return [f'{prefix}_{index}' for index in (1, 2, 3)]
    if causality_type in ('3V_indirect', '3V_both_noCycle', '3V_both_Cycle'):
        return [f'{causality_type}_noNoise']
    if causality_type in ('4V_direct', '4V_indirect'):
        prefix = f'{causality_type}_noNoise'
        return [f'{prefix}_{index}' for index in (1, 2)]
    return [f'{causality_type}_noNoise']


def get_conditioned_score(output, score_type):
    if score_type == 'err':
        return output[1]
    if score_type == 'r2':
        return output[7]
    return output[4]


seed = args.seed
random.seed(seed)
np.random.seed(seed)

output_root = args.output_dir
if not os.path.isabs(output_root):
    output_root = os.path.join(root, output_root)
save_dir = os.path.join(
    output_root,
    args.causality_type,
    f'seed{seed}',
    f'tau_{args.tau}_emd_{args.emd}',
)
os.makedirs(save_dir, exist_ok=True)
if args.kNN_model == 'PCA':
    save_dir = os.path.join(save_dir, f'pca_dim_{args.pca_dim}')
    os.makedirs(save_dir, exist_ok=True)

file_names = get_file_names(
    args.causality_type,
    args.noiseType,
    args.noiseInjectType,
    args.noiseLevel,
)

model_kwargs = {
    'knn': args.knn,
    'L': args.L,
    'method': args.kNN_model,
}
if args.kNN_model == 'PCA':
    model_kwargs['pca_dim'] = args.pca_dim

for file_name in file_names:
    file_dir = os.path.join(save_dir, file_name)
    print(f'\nStarting dataset: {file_name}', flush=True)

    csv_path = os.path.join(
        root,
        args.data_dir,
        args.causality_type,
        file_name + '.csv',
    )
    dataset = MyMultivarData(csv_path)
    df = dataset.df
    var_list = list(df.columns)

    if len(var_list) < 2:
        raise ValueError(
            f'{csv_path} must contain at least two variables for PCM'
        )

    phase2_results = {}

    import time
    start_time = time.time()

    for cause_ind in range(len(var_list)):
        for effect_ind in range(len(var_list)):
            if cause_ind == effect_ind:
                continue

            remaining_inds = [
                index for index in range(len(var_list))
                if index not in (cause_ind, effect_ind)
            ]
            for condition_count in range(1, len(remaining_inds) + 1):
                for condition_inds in combinations(
                    remaining_inds,
                    condition_count,
                ):
                    condition_names = [
                        var_list[index] for index in condition_inds
                    ]
                    key = (
                        f'{var_list[cause_ind]} -> {var_list[effect_ind]}'
                        f' | {", ".join(condition_names)}'
                    )

                    pcm = PCM_simplex(
                        df=df,
                        causes=[var_list[cause_ind]],
                        effects=[var_list[effect_ind]],
                        cond=[var_list[index] for index in condition_inds],
                        tau=args.tau,
                        emd=args.emd,
                        **model_kwargs,
                    )
                    output = pcm.causality()
                    del pcm

                    phase2_results[key] = {
                        'conditioned_score': float(
                            get_conditioned_score(output, args.score_type)
                        ),
                    }

    time_spent = time.time() - start_time

    print('Time spent:', time_spent)
    with open(file_dir + '_time.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(str(time_spent))

    score_name = {
        'err': 'conditioned_error',
        'corr': 'conditioned_corr',
        'r2': 'conditioned_r2',
    }[args.score_type]
    print(f'Phase 2 conditioned scores (all combinations, {score_name}):')
    for key, result in phase2_results.items():
        print(
            f'{key}: {score_name}={result["conditioned_score"]:.6g}'
        )

    with open(
        file_dir + '_phase2_results.txt',
        'w',
        encoding='utf-8',
    ) as output_file:
        for key, result in phase2_results.items():
            output_file.write(
                f'{key}: {score_name}='
                f'{result["conditioned_score"]:.6g}\n'
            )
