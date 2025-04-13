import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
import os

from store import Configs

configs = Configs()

def plot_distribution(data, column, title, save_path):
    plt.figure(figsize=(10, 6))
    dist = data[column].value_counts(normalize=True).sort_index()
    sns.barplot(x=dist.index, y=dist.values)
    plt.title(title)
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return dist.to_dict()

def plot_label_distribution(data, title, save_path):
    plt.figure(figsize=(14, 8))
    label_counts = data[configs.LABELS].sum().sort_values(ascending=False)
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title(title)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return (label_counts / len(data)).to_dict()

def plot_demographic_label_dist(data, demo_col, title, save_path):
    plt.figure(figsize=(16, 8))
    demo_label_props = {}
    for group in data[demo_col].unique():
        group_data = data[data[demo_col] == group]
        demo_label_props[group] = group_data[configs.LABELS].mean().values

    demo_label_df = pd.DataFrame(demo_label_props, index=configs.LABELS).T

    sns.heatmap(demo_label_df, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_demographics(df, output_dir):
    demographics = {}
    demo_dir = os.path.join(output_dir, "demographics")
    os.makedirs(demo_dir, exist_ok=True)
    if 'gender' in df.columns:
        gender_dist = plot_distribution(
            df, 'gender', 'Gender Distribution',
            os.path.join(demo_dir, 'gender_distribution.png')
        )
        demographics['gender'] = gender_dist
    if 'anchor_age' in df.columns:
        age_dist = plot_distribution(
            df, 'anchor_age', 'Age Group Distribution',
            os.path.join(demo_dir, 'age_distribution.png')
        )
        demographics['anchor_age'] = age_dist
    if 'race' in df.columns:
        race_dist = plot_distribution(
            df, 'race', 'Race Distribution',
            os.path.join(demo_dir, 'race_distribution.png')
        )
        demographics['race'] = race_dist
    label_dist = plot_label_distribution(
        df, 'Label Distribution',
        os.path.join(demo_dir, 'label_distribution.png')
    )
    demographics['labels'] = label_dist
    if 'gender' in df.columns:
        plot_demographic_label_dist(
            df, 'gender', 'Label Distribution by Gender',
            os.path.join(demo_dir, 'label_by_gender.png')
        )

    if 'anchor_age' in df.columns:
        plot_demographic_label_dist(
            df, 'anchor_age', 'Label Distribution by Age Group',
            os.path.join(demo_dir, 'label_by_age.png')
        )
    if 'race' in df.columns:
        plot_demographic_label_dist(
            df, 'race', 'Label Distribution by Race',
            os.path.join(demo_dir, 'label_by_race.png')
        )
    with open(os.path.join(demo_dir, 'demographics.json'), 'w') as f:
        json.dump(demographics, f, indent=4)
    print("Demographic analysis saved to demographics directory")
    return demographics

def plot_label_auc_scores(aucs, epoch, output_dir):
    plt.figure(figsize=(12, 6))
    sorted_items = sorted(zip(configs.LABELS, aucs), key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)
    labels, values = zip(*sorted_items)

    bars = plt.bar(range(len(labels)), values)

    plt.xlabel('Labels')
    plt.ylabel('AUC Score')
    plt.title(f'AUC Scores by Label (Epoch {epoch + 1})')
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylim(0, 1.0)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'auc_plots'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'auc_plots', f'label_auc_epoch_{epoch + 1}.png'))
    plt.close()


def compare_model_performance(original_results, debiased_results):
    comparison_dir = os.path.join(configs.OUTPUT_DIR, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    original_aucs = {label: data['auc'] for label, data in original_results.items()
                     if not np.isnan(data['auc'])}
    debiased_aucs = {label: data['auc'] for label, data in debiased_results.items()
                     if not np.isnan(data['auc'])}
    common_labels = set(original_aucs.keys()).intersection(set(debiased_aucs.keys()))

    labels = list(common_labels)
    original_values = [original_aucs[label] for label in labels]
    debiased_values = [debiased_aucs[label] for label in labels]

    plt.figure(figsize=(14, 8))
    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, original_values, width, label='Original Model')
    plt.bar(x + width / 2, debiased_values, width, label='Debiased Model')

    plt.xlabel('Labels')
    plt.ylabel('AUC')
    plt.title('AUC Comparison: Original vs. Debiased Model')
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'auc_comparison.png'))
    plt.close()
    original_fprs = {label: data['false_positive_rate'] for label, data in original_results.items()
                     if not np.isnan(data['false_positive_rate'])}
    debiased_fprs = {label: data['false_positive_rate'] for label, data in debiased_results.items()
                     if not np.isnan(data['false_positive_rate'])}

    common_labels = set(original_fprs.keys()).intersection(set(debiased_fprs.keys()))

    labels = list(common_labels)
    original_values = [original_fprs[label] for label in labels]
    debiased_values = [debiased_fprs[label] for label in labels]

    plt.figure(figsize=(14, 8))
    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, original_values, width, label='Original Model')
    plt.bar(x + width / 2, debiased_values, width, label='Debiased Model')

    plt.xlabel('Labels')
    plt.ylabel('False Positive Rate')
    plt.title('FPR Comparison: Original vs. Debiased Model')
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'fpr_comparison.png'))
    plt.close()

    original_fnrs = {label: data['false_negative_rate'] for label, data in original_results.items()
                     if not np.isnan(data['false_negative_rate'])}
    debiased_fnrs = {label: data['false_negative_rate'] for label, data in debiased_results.items()
                     if not np.isnan(data['false_negative_rate'])}

    common_labels = set(original_fnrs.keys()).intersection(set(debiased_fnrs.keys()))

    labels = list(common_labels)
    original_values = [original_fnrs[label] for label in labels]
    debiased_values = [debiased_fnrs[label] for label in labels]
    plt.figure(figsize=(14, 8))
    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, original_values, width, label='Original Model')
    plt.bar(x + width / 2, debiased_values, width, label='Debiased Model')

    plt.xlabel('Labels')
    plt.ylabel('False Negative Rate')
    plt.title('FNR Comparison: Original vs. Debiased Model')
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'fnr_comparison.png'))
    plt.close()
    with open(os.path.join(comparison_dir, 'performance_comparison.json'), 'w') as f:
        comparison = {
            'average_auc': {
                'original': np.mean(list(original_aucs.values())),
                'debiased': np.mean(list(debiased_aucs.values())),
                'difference': np.mean(list(debiased_aucs.values())) - np.mean(list(original_aucs.values()))
            },
            'average_fpr': {
                'original': np.mean(list(original_fprs.values())),
                'debiased': np.mean(list(debiased_fprs.values())),
                'difference': np.mean(list(debiased_fprs.values())) - np.mean(list(original_fprs.values()))
            },
            'average_fnr': {
                'original': np.mean(list(original_fnrs.values())),
                'debiased': np.mean(list(debiased_fnrs.values())),
                'difference': np.mean(list(debiased_fnrs.values())) - np.mean(list(original_fnrs.values()))
            },
            'per_label': {
                label: {
                    'auc': {
                        'original': original_aucs.get(label, float('nan')),
                        'debiased': debiased_aucs.get(label, float('nan')),
                        'difference': debiased_aucs.get(label, float('nan')) - original_aucs.get(label, float('nan'))
                    },
                    'fpr': {
                        'original': original_fprs.get(label, float('nan')),
                        'debiased': debiased_fprs.get(label, float('nan')),
                        'difference': debiased_fprs.get(label, float('nan')) - original_fprs.get(label, float('nan'))
                    },
                    'fnr': {
                        'original': original_fnrs.get(label, float('nan')),
                        'debiased': debiased_fnrs.get(label, float('nan')),
                        'difference': debiased_fnrs.get(label, float('nan')) - original_fnrs.get(label, float('nan'))
                    }
                }
                for label in set(original_aucs.keys()).union(set(debiased_aucs.keys()))
            }
        }
        json.dump(comparison, f, indent=4)

    print("Model comparison analysis completed and saved")