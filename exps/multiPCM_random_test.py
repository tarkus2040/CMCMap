import argparse
import csv
import os
import sys
from itertools import combinations

import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

from data_loaders.generated.multivar import MyMultivarData
from utils.causal_simplex import PCM_simplex


def determine_condition(output, score_type, threshold):
    ratio_index = {'err': 2, 'corr': 5, 'r2': 8}[score_type]
    ratio = output[ratio_index]
    if 0.95 < ratio < 1.05:
        return 0
    if score_type == 'err':
        return int(ratio >= threshold)
    return int(ratio <= threshold)


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


parser = argparse.ArgumentParser(
    "Run PCM for all variable pairs over repeated random contiguous time-series windows"
)
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='3V_direct_CMC')
parser.add_argument('--seed', type=int, default=97, help='seed used to choose window start indices')
parser.add_argument('--L', type=int, default=1000, help='length of each input time-series window')
parser.add_argument('--repeats', type=int, default=5, help='number of random windows per dataset')
parser.add_argument('--noiseType', type=str, default='None')
parser.add_argument('--noiseInjectType', type=str, default='add')
parser.add_argument('--noiseLevel', type=float, default=1e-2)
parser.add_argument('--tau', type=int, default=1)
parser.add_argument('--emd', type=int, default=16)
parser.add_argument('--knn', type=int, default=10)
parser.add_argument('--score_type', choices=('err', 'corr', 'r2'), default='corr')
parser.add_argument('--pcm_thres', type=float, default=0.45)
parser.add_argument(
    '--output_dir',
    type=str,
    default=os.path.join('outputs', 'multiPCM_random_test'),
    help='output directory relative to the repository root unless absolute',
)
args = parser.parse_args()

if args.L <= 0:
    parser.error('--L must be greater than 0')
if args.repeats <= 0:
    parser.error('--repeats must be greater than 0')
if args.tau <= 0 or args.emd <= 0 or args.knn <= 0:
    parser.error('--tau, --emd, and --knn must be greater than 0')

rng = np.random.default_rng(args.seed)
file_names = get_file_names(
    args.causality_type,
    args.noiseType,
    args.noiseInjectType,
    args.noiseLevel,
)
output_root = args.output_dir
if not os.path.isabs(output_root):
    output_root = os.path.join(root, output_root)
save_dir = os.path.join(output_root, args.causality_type, f'seed{args.seed}')
os.makedirs(save_dir, exist_ok=True)

metric_index = {'err': 0, 'corr': 3, 'r2': 6}[args.score_type]
metric_name = {'err': 'Error', 'corr': 'Correlation', 'r2': 'R2'}[args.score_type]
summary_rows = []

for file_name in file_names:
    csv_path = os.path.join(root, args.data_dir, args.causality_type, f'{file_name}.csv')
    dataset = MyMultivarData(csv_path)
    df = dataset.df
    variables = list(df.columns)

    if len(variables) < 3:
        raise ValueError(
            f'{csv_path} must contain at least three variables for PCM conditioning'
        )
    if len(df) < args.L:
        raise ValueError(
            f'{csv_path} has {len(df)} rows, fewer than the requested window length L={args.L}'
        )

    available_starts = len(df) - args.L + 1
    start_indices = rng.choice(
        available_starts,
        size=args.repeats,
        replace=args.repeats > available_starts,
    )
    pairs = list(combinations(variables, 2))

    for first_variable, second_variable in pairs:
        conditions = [
            name for name in variables if name not in (first_variable, second_variable)
        ]
        directions = [
            (first_variable, second_variable),
            (second_variable, first_variable),
        ]
        direction_outputs = []
        direction_run_outputs = []

        for cause, effect in directions:
            run_outputs = []
            for start in start_indices:
                window = df.iloc[start:start + args.L].reset_index(drop=True)
                pcm = PCM_simplex(
                    df=window,
                    causes=[cause],
                    effects=[effect],
                    cond=conditions,
                    tau=args.tau,
                    emd=args.emd,
                    L=args.L,
                    knn=args.knn,
                )
                run_outputs.append(pcm.causality())
                del pcm

            run_outputs = np.asarray(run_outputs, dtype=float)
            direction_run_outputs.append(run_outputs)
            direction_outputs.append(np.mean(run_outputs, axis=0))

        result_indices = [
            determine_condition(output, args.score_type, args.pcm_thres)
            for output in direction_outputs
        ]
        conclusions = [
            '他の変数は条件' if result_idx else '他の変数は条件ではない'
            for result_idx in result_indices
        ]

        file_save_name = (
            f'{file_name}_{first_variable}_to_{second_variable}'
            f'_L{args.L}_repeats{args.repeats}_tau{args.tau}_emd{args.emd}'
            f'_knn{args.knn}_pcmThres{args.pcm_thres}_{args.score_type}'
        )
        output_path = os.path.join(save_dir, file_save_name + '_output.txt')

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(f'Dataset: {file_name}\n')
            output_file.write(f'Variables: {", ".join(variables)}\n')
            output_file.write(
                f'Condition variables: {", ".join(conditions) if conditions else "なし"}\n'
            )
            output_file.write(f'Score type: {args.score_type}  Threshold: {args.pcm_thres}\n')
            output_file.write(f'Window length: {args.L}  Repeats: {args.repeats}\n')
            output_file.write(f'Start indices: {", ".join(map(str, start_indices))}\n\n')
            output_file.write(
                f'{"Direction":<20} {"Direct":>14} {"Conditioned":>14} {"Ratio":>14}  Conclusion\n'
            )
            for (cause, effect), output, conclusion in zip(
                directions, direction_outputs, conclusions
            ):
                output_file.write(
                    f'{cause + " -> " + effect:<20} {output[metric_index]:>14.6g} '
                    f'{output[metric_index + 1]:>14.6g} '
                    f'{output[metric_index + 2]:>14.6g}  {conclusion}\n'
                )

        np.save(os.path.join(save_dir, file_save_name + '_output.npy'), direction_outputs[0])
        np.save(
            os.path.join(save_dir, file_save_name + '_reverse_output.npy'),
            direction_outputs[1],
        )
        np.save(
            os.path.join(save_dir, file_save_name + '_runs.npy'),
            np.asarray(direction_run_outputs),
        )
        np.save(os.path.join(save_dir, file_save_name + '_result_idx.npy'), result_indices[0])
        np.save(
            os.path.join(save_dir, file_save_name + '_reverse_result_idx.npy'),
            result_indices[1],
        )
        with open(os.path.join(save_dir, file_save_name + '_conclus.txt'), 'w', encoding='utf-8') as conclusion_file:
            for (cause, effect), conclusion in zip(directions, conclusions):
                conclusion_file.write(f'{cause} -> {effect}: {conclusion}\n')

        for (cause, effect), output, result_idx, conclusion in zip(
            directions, direction_outputs, result_indices, conclusions
        ):
            summary_rows.append({
                'dataset': file_name,
                'cause': cause,
                'effect': effect,
                'conditions': ','.join(conditions),
                'score_type': args.score_type,
                'window_length': args.L,
                'repeats': args.repeats,
                'start_indices': ','.join(map(str, start_indices)),
                'direct_score_mean': output[metric_index],
                'conditioned_score_mean': output[metric_index + 1],
                'ratio_mean': output[metric_index + 2],
                'result': conclusion,
            })
            print(
                f'{file_name}: {cause} -> {effect} | {metric_name} '
                f'{output[metric_index]:.6g}, {output[metric_index + 1]:.6g}, '
                f'ratio={output[metric_index + 2]:.6g} | {conclusion}'
            )
        print(f'  始点: {", ".join(map(str, start_indices))}')
        print(f'  保存先: {output_path}')

summary_path = os.path.join(
    save_dir,
    f'all_pairs_{args.causality_type}_seed{args.seed}_L{args.L}_repeats{args.repeats}'
    f'_tau{args.tau}_emd{args.emd}_knn{args.knn}_pcmThres{args.pcm_thres}'
    f'_{args.score_type}.csv',
)
with open(summary_path, 'w', newline='', encoding='utf-8-sig') as summary_file:
    writer = csv.DictWriter(summary_file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)
print(f'全ペアの集約結果: {summary_path}')
