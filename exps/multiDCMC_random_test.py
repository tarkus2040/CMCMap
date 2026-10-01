import argparse
import csv
import os
import random
import sys
from itertools import combinations

import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

from data_loaders.generated.multivar import MyMultivarData
from utils.causal_simplex import DCMC_simplex


parser = argparse.ArgumentParser(
    "Run DCMC for all variable pairs over random contiguous time-series windows"
)
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='4V_direct_CMC')
parser.add_argument('--seed', type=int, default=97, help='seed used to choose window start indices')
parser.add_argument('--L', type=int, default=1000, help='length of each input time-series window')
parser.add_argument('--repeats', type=int, default=5, help='number of random windows per dataset')
parser.add_argument('--noiseType', type=str, default='None')
parser.add_argument('--noiseInjectType', type=str, default='add')
parser.add_argument('--noiseLevel', type=float, default=1e-2)
parser.add_argument('--tau', type=int, default=1)
parser.add_argument('--emd', type=int, default=3)
parser.add_argument('--knn', type=int, default=4)
parser.add_argument('--dcmc_thres', type=float, default=0.5)
parser.add_argument(
    '--output_dir',
    type=str,
    default=os.path.join('outputs', 'multiDCMC_random_test'),
    help='output directory relative to the repository root unless absolute',
)
args = parser.parse_args()

if args.L <= 0:
    parser.error('--L must be greater than 0')
if args.repeats <= 0:
    parser.error('--repeats must be greater than 0')
if args.tau <= 0 or args.emd <= 0 or args.knn <= 0:
    parser.error('--tau, --emd, and --knn must be greater than 0')

random.seed(args.seed)
rng = np.random.default_rng(args.seed)

if args.noiseType is not None and args.noiseType.lower() != 'none':
    if args.causality_type in ('3V_direct', '4V_both_noCycle', '4V_both_Cycle'):
        prefix = f'{args.causality_type}_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
        file_names = [f'{prefix}_{index}' for index in (1, 2, 3)]
    elif args.causality_type in ('3V_indirect', '3V_both_noCycle', '3V_both_Cycle'):
        file_names = [f'{args.causality_type}_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}']
    elif args.causality_type in ('4V_direct', '4V_indirect'):
        prefix = f'{args.causality_type}_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
        file_names = [f'{prefix}_{index}' for index in (1, 2)]
    else:
        file_names = [f'{args.causality_type}_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}']
else:
    if args.causality_type in ('3V_direct', '4V_both_noCycle', '4V_both_Cycle'):
        prefix = f'{args.causality_type}_noNoise'
        file_names = [f'{prefix}_{index}' for index in (1, 2, 3)]
    elif args.causality_type in ('3V_indirect', '3V_both_noCycle', '3V_both_Cycle'):
        file_names = [f'{args.causality_type}_noNoise']
    elif args.causality_type in ('4V_direct', '4V_indirect'):
        prefix = f'{args.causality_type}_noNoise'
        file_names = [f'{prefix}_{index}' for index in (1, 2)]
    else:
        file_names = [f'{args.causality_type}_noNoise']

output_root = args.output_dir
if not os.path.isabs(output_root):
    output_root = os.path.join(root, output_root)
save_dir = os.path.join(output_root, args.causality_type, f'seed{args.seed}')
os.makedirs(save_dir, exist_ok=True)

summary_rows = []

for file_name in file_names:
    csv_path = os.path.join(root, args.data_dir, args.causality_type, f'{file_name}.csv')
    dataset = MyMultivarData(csv_path)
    df = dataset.df
    variables = list(df.columns)

    if len(variables) < 2:
        raise ValueError(f'{csv_path} must contain at least two variables')
    if len(df) < args.L:
        raise ValueError(
            f'{csv_path} has {len(df)} rows, fewer than the requested window length L={args.L}'
        )

    available_starts = len(df) - args.L + 1
    # Use distinct starts whenever there are enough possible windows; otherwise allow repeats.
    start_indices = rng.choice(
        available_starts,
        size=args.repeats,
        replace=args.repeats > available_starts,
    )
    pairs = list(combinations(variables, 2))

    for cause, effect in pairs:
        conditions = [name for name in variables if name not in (cause, effect)]
        run_outputs = []

        for start in start_indices:
            window = df.iloc[start:start + args.L].reset_index(drop=True)
            dcmc = DCMC_simplex(
                df=window,
                causes=[cause],
                effects=[effect],
                cond=conditions,
                tau=args.tau,
                emd=args.emd,
                L=args.L,
                knn=args.knn,
            )
            run_outputs.append(dcmc.causality())
            del dcmc

        run_outputs = np.asarray(run_outputs, dtype=float)
        mean_output = np.mean(run_outputs, axis=0)
        dir_score_c2e, dir_score_e2c = mean_output[:2]

        if dir_score_c2e >= args.dcmc_thres and dir_score_e2c < args.dcmc_thres:
            result_idx = 1
            conclusion = f'直接因果を検出: {cause} -> {effect}'
        elif dir_score_e2c >= args.dcmc_thres and dir_score_c2e < args.dcmc_thres:
            result_idx = 2
            conclusion = f'直接因果を検出: {effect} -> {cause}'
        elif dir_score_c2e >= args.dcmc_thres and dir_score_e2c >= args.dcmc_thres:
            result_idx = 3
            conclusion = f'双方向の直接因果を検出: {cause} <-> {effect}'
        else:
            result_idx = 0
            conclusion = '直接因果は検出されませんでした。'

        pair_label = f'{cause}_to_{effect}'
        file_save_name = (
            f'{file_name}_{pair_label}_L{args.L}_repeats{args.repeats}'
            f'_tau{args.tau}_emd{args.emd}_knn{args.knn}_dcmcThres{args.dcmc_thres}'
        )
        output_path = os.path.join(save_dir, file_save_name + '_output.txt')

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(f'Dataset: {file_name}\n')
            output_file.write(f'Cause: {cause}  Effect: {effect}\n')
            output_file.write(f'Conditions: {", ".join(conditions) if conditions else "なし"}\n')
            output_file.write(f'Window length: {args.L}  Repeats: {args.repeats}\n')
            output_file.write(f'Start indices: {", ".join(map(str, start_indices))}\n')
            output_file.write(f'Direct causality threshold: {args.dcmc_thres}\n\n')
            output_file.write(f'{"Score":<28} {f"{cause} -> {effect}":>16} {f"{effect} -> {cause}":>16}\n')
            output_file.write(f'{"Direct causality (DCMC)":<28} {mean_output[0]:>16.6g} {mean_output[1]:>16.6g}\n')
            output_file.write(f'{"Cross mapping (CMC)":<28} {mean_output[2]:>16.6g} {mean_output[3]:>16.6g}\n')
            output_file.write(f'\n判定: {conclusion}\n')

        np.save(os.path.join(save_dir, file_save_name + '_output.npy'), mean_output)
        np.save(os.path.join(save_dir, file_save_name + '_runs.npy'), run_outputs)
        np.save(os.path.join(save_dir, file_save_name + '_result_idx.npy'), result_idx)

        summary_rows.append({
            'dataset': file_name,
            'cause': cause,
            'effect': effect,
            'conditions': ','.join(conditions),
            'window_length': args.L,
            'repeats': args.repeats,
            'start_indices': ','.join(map(str, start_indices)),
            'dcmc_cause_to_effect': mean_output[0],
            'dcmc_effect_to_cause': mean_output[1],
            'cmc_cause_to_effect': mean_output[2],
            'cmc_effect_to_cause': mean_output[3],
            'result': conclusion,
        })

        print(f'\nデータ: {file_name}  変数ペア: {cause} / {effect}')
        print(f'使用区間長: {args.L}  繰り返し: {args.repeats}')
        print(f'始点: {", ".join(map(str, start_indices))}')
        print(f'{"スコア":<24} {f"{cause} -> {effect}":>16} {f"{effect} -> {cause}":>16}')
        print(f'{"直接因果 (DCMC) 平均":<24} {mean_output[0]:>16.6g} {mean_output[1]:>16.6g}')
        print(f'{"クロスマッピング (CMC) 平均":<24} {mean_output[2]:>16.6g} {mean_output[3]:>16.6g}')
        print(f'判定: {conclusion}')
        print(f'保存先: {output_path}')

summary_path = os.path.join(
    save_dir,
    f'all_pairs_{args.causality_type}_seed{args.seed}_L{args.L}_repeats{args.repeats}'
    f'_tau{args.tau}_emd{args.emd}_knn{args.knn}_dcmcThres{args.dcmc_thres}.csv',
)
with open(summary_path, 'w', newline='', encoding='utf-8-sig') as summary_file:
    writer = csv.DictWriter(summary_file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)
print(f'\n全ペアの集約結果: {summary_path}')
