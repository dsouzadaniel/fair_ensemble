import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from beautifultable import BeautifulTable

parser = argparse.ArgumentParser(description='Generate Plots and Metrics')
parser.add_argument('--input_folder', type=str, help='location of experiment artifacts', required=True)
parser.add_argument('--output_folder', type=str, help='location to write plots to', default='Output')

parser.add_argument('--model_desc', type=str, help='describe the model', default='SOME_MODEL')
parser.add_argument('--data_desc', type=str, help='describe the dataset', default='SOME_DATASET')

parser.add_argument('--K', type=int, help='How many top/bottom classes to consider for metrics/plots(default: 10)',
                    default=10)
args = parser.parse_args()
folder = Path(os.path.join('./', args.output_folder)).absolute()
folder.mkdir(parents=True, exist_ok=True)

_ablation_mapper = {"DA": "Change_ModelInit_BatchOrder",
                    "DA_MS": "Change_BatchOrder",
                    "DA_BS": "Change_ModelInit",
                    "DA_MS_BS": "Fixed_ModelInit_BatchOrder",
                    "MS_BS": "Change_DA",
                    "RANDOM": "All_Sources_Noise",
                    "SUBSET": "Subset_TrainData"}

_ablation_mapper_cosmetic = {"DA": "Change ModelInit & BatchOrder",
                    "DA_MS": "Change BatchOrder",
                    "DA_BS": "Change ModelInit",
                    "DA_MS_BS": "Fixed ModelInit & BatchOrder",
                    "MS_BS": "Change DA",
                    "RANDOM": "All Sources Noise (Baseline)",
                    "SUBSET": "Subset TrainData"}

_subset = 'eval'
_baseline = 'RANDOM'

avg_accs = []
_ablation_folder = f'{args.input_folder}/{_baseline}'

for _, variants, _ in os.walk(_ablation_folder):
    for vx, variant in enumerate(variants):
        if variant.startswith('ix'):
            npy_folder = os.path.join(_ablation_folder, variant)
            accs = np.load(os.path.join(npy_folder, f'{_subset}_per_class_acc.npy'))
            # Collect Final Accuracy from each run
            avg_accs.append(accs[-1, :])

# Average out the accuracies
avg_accs = np.mean(np.stack(avg_accs), axis=0)

# Collect the Kth Worst and Best performing classes
K_WORST = np.argsort(avg_accs)[:args.K]
K_BEST = np.argsort(avg_accs)[-args.K:]

print(f"Worst Performing Classes : {K_WORST}")
print(f"Best Performing Classes : {K_BEST}")

# CLASS_INFO = [(K_WORST, f'Bottom-{args.K}', '#d62728'), (K_BEST, f'Top-{args.K}', '#2ca02c')]  # 0 indexed
CLASS_INFO = [(K_WORST, f'Bottom-{args.K}', '#4b6ee8'), (K_BEST, f'Top-{args.K}', '#2ca02c')]  # 0 indexed


mosaic = [["DA", "DA_MS_BS"], ["DA_MS", "DA_BS"], ["MS_BS", "RANDOM"]]

fig = plt.figure(constrained_layout=True, figsize=(16, 8))
fig.suptitle(f'{args.model_desc} on {args.data_desc} [Avg {_subset.title()} Accuracy]', fontsize=25)
ax_dict = fig.subplot_mosaic(mosaic)

legend_elements = []
for kth_class_ixs, kth_class_str, kth_class_color in CLASS_INFO:
    legend_elements.append(Line2D([0], [0], color=kth_class_color, lw=4, label=kth_class_str))

collect_for_metrics = {}
for _ablation in [i for r in mosaic for i in r]:

    for _, kth_class_str, _ in CLASS_INFO:
        collect_for_metrics[f"{_ablation}__{kth_class_str}"] = []

        # Fig Settings
    ax_dict[_ablation].set_title(f'{_ablation_mapper[_ablation]}', fontdict={'fontsize': 20})
    ax_dict[_ablation].grid(True, linewidth=0.5, color='gray', linestyle='-', alpha=0.5)
    ax_dict[_ablation].set_xlabel('Epochs', fontdict={'fontsize': 15})
    ax_dict[_ablation].set_ylabel('Avg Accuracy', fontdict={'fontsize': 15})

    ax_dict[_ablation].legend(handles=legend_elements, loc='lower right')

    mean_top1 = []
    _ablation_folder = f'{args.input_folder}/{_ablation}'
    for _, variants, _ in os.walk(_ablation_folder):
        for vx, variant in enumerate(variants):
            if variant.startswith('ix'):
                npy_folder = os.path.join(_ablation_folder, variant)
                accs = np.load(os.path.join(npy_folder, f'{_subset}_per_class_acc.npy'))
                mean_top1.append(np.mean(accs, axis=-1))
                for kth_class_ixs, kth_class_str, kth_class_color in CLASS_INFO:
                    mean_accs = np.mean(accs[:, kth_class_ixs], axis=-1)
                    collect_for_metrics[f"{_ablation}__{kth_class_str}"].append(mean_accs)

                    ax_dict[_ablation].plot(mean_accs, label=kth_class_str, color=kth_class_color, linewidth=1)
                    ax_dict[_ablation].title.set_text(f'{_ablation_mapper[_ablation]}')

# Write Plot
fig.savefig(os.path.join(args.output_folder, f'{args.data_desc}_{args.model_desc}_{_subset}.png'), facecolor='w',
            transparent=True)
fig.savefig(os.path.join(args.output_folder, f'{args.data_desc}_{args.model_desc}_{_subset}.pdf'), bbox_inches='tight')
plt.close()

collect_for_metrics = {}
for _ablation in [i for r in mosaic for i in r]:
    for _, kth_class_str, _ in CLASS_INFO:
        collect_for_metrics[f"{_ablation}__{kth_class_str}"] = []

    fig = plt.figure(figsize=(8, 6))

    # Fig Settings
    # plt.title(f'{_ablation_mapper_cosmetic[_ablation]}', fontdict={'fontsize': 20})
    plt.grid(True, linewidth=0.5, color='gray', linestyle='-', alpha=0.5)
    # plt.xlabel('Epochs', fontdict={'fontsize': 14})
    # plt.ylabel('Avg Accuracy', fontdict={'fontsize': 14})
    

    # plt.xlim((0, 1000))
    
    mean_top1 = []
    _ablation_folder = f'{args.input_folder}/{_ablation}'
    for _, variants, _ in os.walk(_ablation_folder):
        for vx, variant in enumerate(variants):
            if variant.startswith('ix'):
                npy_folder = os.path.join(_ablation_folder, variant)
                accs = np.load(os.path.join(npy_folder, f'{_subset}_per_class_acc.npy'))
                mean_top1.append(np.mean(accs, axis=-1))
                plt.plot(np.mean(accs, axis=-1), label='Overall' if vx == 0 else '', color='black', linewidth=0.5, alpha=0.7)
                for kth_class_ixs, kth_class_str, kth_class_color in CLASS_INFO:
                    mean_accs = np.mean(accs[:, kth_class_ixs], axis=-1)
                    collect_for_metrics[f"{_ablation}__{kth_class_str}"].append(mean_accs)
                    plt.plot(mean_accs, label=kth_class_str if vx == 0 else '', color=kth_class_color, linewidth=0.5, alpha=0.7)
    
    # print(mean_accs.shape[0])
    plt.xlim([0,mean_accs.shape[0]-1])
    lgnd = plt.legend(loc='lower right', prop={'size': 15})
    for h in lgnd.legendHandles:
        h.set_linewidth(5)

    handles, labels = plt.gca().get_legend_handles_labels()

    order = [2,0,1]
    plt.legend([lgnd.legendHandles[idx] for idx in order],[labels[idx] for idx in order],loc='lower right', prop={'size': 15}) 
    plt.tick_params(axis='both', labelsize=14)
    # plt.xticks(list(np.arange(0,mean_accs.shape[0], 5)))
    plt.yticks(list(np.arange(0,101,20)))

    # Write Plot
    plt.savefig(os.path.join(args.output_folder, f'{args.data_desc}_{args.model_desc}_{_ablation_mapper[_ablation]}.png'), facecolor='w',
                transparent=True, bbox_inches='tight')
    plt.savefig(os.path.join(args.output_folder, f'{args.data_desc}_{args.model_desc}_{_ablation_mapper[_ablation]}.pdf'), bbox_inches='tight')


###########################################################
####################### Metrics ###########################
###########################################################

def average_l2_distance_1D(points):
    # Take averaged absolute differences of all combinations of points
    distances = []
    for ix in range(len(points)):
        for jx in range(ix + 1, len(points)):
            distances.append(abs(points[ix] - points[jx]))
    distances = np.array(distances)
    return distances.mean()


def variance_1D(points):
    # Take variance
    points = np.array(points)
    return points.var()


print("\n\n")

print(f"K-Best ( K = {args.K} )")
table = BeautifulTable()
collect_for_df = []
table.columns.header = ["Ablation", "Pairwise Distance", "Variance"]
for k, v in collect_for_metrics.items():
    if 'Top' in k and len(v) > 0:
        pair_dist_vals = np.apply_along_axis(average_l2_distance_1D, axis=0, arr=np.stack(v))
        var_vals = np.apply_along_axis(variance_1D, axis=0, arr=np.stack(v))

        ablation_str = '_'.join(_ablation_mapper[k.split('__')[0]].split('_'))
        metric1_str = f"{round(np.mean(pair_dist_vals), 2)} +/-{round(np.std(pair_dist_vals), 2)}"
        metric2_str = f"{round(np.mean(var_vals), 2)} +/-{round(np.std(var_vals), 2)}"
        table.rows.append([ablation_str, metric1_str, metric2_str])
        collect_for_df.append((ablation_str, metric1_str, metric2_str))
df = pd.DataFrame(collect_for_df, columns=table.columns.header)
df.to_csv(os.path.join(args.output_folder, f"{args.data_desc}_{args.model_desc}__{args.K}_Best.csv"), index=False)
print(table)
print("\n\n")
print(f"K-Worst ( K = {args.K} )")
table = BeautifulTable()
collect_for_df = []
table.columns.header = ["Ablation", "Pairwise Distance", "Variance"]
for k, v in collect_for_metrics.items():
    if 'Bottom' in k and len(v) > 0:
        pair_dist_vals = np.apply_along_axis(average_l2_distance_1D, axis=0, arr=np.stack(v))
        var_vals = np.apply_along_axis(variance_1D, axis=0, arr=np.stack(v))

        ablation_str = '_'.join(_ablation_mapper[k.split('__')[0]].split('_'))
        metric1_str = f"{round(np.mean(pair_dist_vals), 2)} +/-{round(np.std(pair_dist_vals), 2)}"
        metric2_str = f"{round(np.mean(var_vals), 2)} +/-{round(np.std(var_vals), 2)}"
        table.rows.append([ablation_str, metric1_str, metric2_str])
        collect_for_df.append((ablation_str, metric1_str, metric2_str))
df = pd.DataFrame(collect_for_df, columns=table.columns.header)
df.to_csv(os.path.join(args.output_folder, f"{args.data_desc}_{args.model_desc}__{args.K}_Worst.csv"), index=False)
print(table)
