import concurrent
import concurrent.futures
import pandas as pd
import numpy as np
import torch

from store import Configs

configs = Configs()

def preprocess_dataframe(df):
    columns_to_drop = ['subject_id', 'study_id', 'dicom_id', 'split']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    if 'age' in df.columns:
        age_bins = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 150], labels=configs.AGE_BIN_LABELS, right=False)
        df['anchor_age'] = age_bins

    return df

def preprocess_labels(df, label_columns):
    print(f"Preprocessing {len(label_columns)} label columns...")

    missing_cols = [col for col in label_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing label columns: {missing_cols}")

    for col in label_columns:
        if col in df.columns:
            orig_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')

            na_count = df[col].isna().sum()
            if na_count > 0:
                print(f"  Column '{col}': Filling {na_count} NaN values ({na_count / len(df) * 100:.2f}%)")
                df[col] = df[col].fillna(0.0)
            df[col] = df[col].astype(np.float32)

            if df[col].dtype != orig_dtype:
                print(f"  Column '{col}': Converted from {orig_dtype} to {df[col].dtype}")

    return df


def safe_convert_labels(row, label_columns):
    try:
        values = np.array([row[col] for col in label_columns], dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Error in safe_convert_labels: {e}")
        values = []
        for col in label_columns:
            try:
                if col in row and not pd.isna(row[col]):
                    values.append(float(row[col]))
                else:
                    values.append(0.0)
            except:
                values.append(0.0)
        return torch.tensor(values, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

def create_debiased_dataframe(df, protected_attrs=['gender', 'race', 'anchor_age'], n_jobs=9):
    print("Creating debiased dataset for multilabel classification...")
    potential_labels = [col for col in df.columns if col not in protected_attrs
                        and df[col].nunique() <= 2 and pd.api.types.is_numeric_dtype(df[col])]
    LABELS = potential_labels
    print(f"Detected label columns: {LABELS}")
    attr_counts = {}
    for attr in protected_attrs:
        if attr in df.columns:
            attr_counts[attr] = df[attr].value_counts()
            print(f"{attr} distribution: {attr_counts[attr].to_dict()}")
    if len(protected_attrs) > 1:
        df['protected_combination'] = df[protected_attrs].astype(str).agg('_'.join, axis=1)
        attr_counts['protected_combination'] = df['protected_combination'].value_counts()
        median_count = attr_counts['protected_combination'].median()
        target_counts = {group: int(median_count) for group in attr_counts['protected_combination'].index}
        print("Analyzing label distribution across protected groups...")
        label_bias_metrics = {}
        for label_col in LABELS:
            group_metrics = {}
            overall_positive_rate = df[label_col].mean()

            for group in df['protected_combination'].unique():
                group_df = df[df['protected_combination'] == group]
                group_positive_rate = group_df[label_col].mean()
                bias_metric = abs(group_positive_rate - overall_positive_rate)
                group_metrics[group] = {
                    'positive_rate': group_positive_rate,
                    'bias_metric': bias_metric,
                    'size': len(group_df)
                }

            label_bias_metrics[label_col] = group_metrics

            print(f"Label '{label_col}' - Overall positive rate: {overall_positive_rate:.4f}")
            for group, metrics in group_metrics.items():
                print(
                    f"  Group '{group}': Positive rate = {metrics['positive_rate']:.4f}, Bias = {metrics['bias_metric']:.4f}")

        aggregate_bias = {}
        for group in df['protected_combination'].unique():
            total_bias = sum(label_bias_metrics[label][group]['bias_metric'] for label in LABELS)
            aggregate_bias[group] = total_bias / len(LABELS)

        print("Aggregate bias across all labels:")
        for group, bias in aggregate_bias.items():
            print(f"  Group '{group}': {bias:.4f}")
        df_debiased = df.copy()

        def process_group_oversampling(group_data):
            group, target_count = group_data
            group_df = df[df['protected_combination'] == group]
            current_count = len(group_df)
            additional_needed = target_count - current_count

            if additional_needed <= 0:
                return pd.DataFrame()
            print(f"  Group '{group}': Adding {additional_needed} samples")

            group_indices = group_df.index.tolist()
            group_indices = np.array(group_indices, dtype=np.int64)

            if group in aggregate_bias and aggregate_bias[group] > 0.1:
                bias_scores = []
                for idx in group_indices:
                    sample = df.loc[idx]
                    score = 0
                    for label in LABELS:
                        overall_rate = df[label].mean()
                        score += abs(sample[label] - overall_rate)
                    bias_scores.append(score)

                index_scores = list(zip(group_indices, bias_scores))
                index_scores.sort(key=lambda x: x[1], reverse=True)
                priority_indices = np.array([idx for idx, _ in index_scores[:int(0.7 * len(index_scores))]],
                                            dtype=np.int64)
                sampling_indices = []

                while len(sampling_indices) < additional_needed:
                    remaining = additional_needed - len(sampling_indices)
                    samples_to_add = min(remaining, len(priority_indices))
                    sampling_indices.extend(priority_indices[:samples_to_add])

                    if len(sampling_indices) < additional_needed:
                        remaining = additional_needed - len(sampling_indices)
                        remaining_int = int(remaining)
                        sampling_indices.extend(np.random.choice(group_indices,
                                                                 size=remaining_int,
                                                                 replace=True))
            else:
                additional_needed_int = int(additional_needed)
                sampling_indices = np.random.choice(group_indices,
                                                    size=additional_needed_int,
                                                    replace=True)
            new_samples = df.loc[sampling_indices].copy()
            return new_samples

        groups_to_oversample = {group: target for group, target in target_counts.items()
                                if attr_counts['protected_combination'][group] < median_count}

        if groups_to_oversample:
            print("Applying oversampling to underrepresented groups using multi-threading...")
            oversample_tasks = list(groups_to_oversample.items())

            additional_samples = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, len(oversample_tasks))) as executor:
                for result in executor.map(process_group_oversampling, oversample_tasks):
                    if not result.empty:
                        additional_samples.append(result)
            if additional_samples:
                df_debiased = pd.concat([df_debiased] + additional_samples, ignore_index=True)
                print(f"  Added {sum(len(samples) for samples in additional_samples)} oversampled records")

        def process_group_undersampling(group_data):
            group, target_count = group_data
            group_df = df_debiased[df_debiased['protected_combination'] == group]
            current_count = len(group_df)

            if current_count <= target_count:
                return group_df.index.tolist()
            print(f"  Group '{group}': Reducing from {current_count} to {target_count} samples")

            group_indices = group_df.index.tolist()
            if group in aggregate_bias and aggregate_bias[group] > 0.1:
                bias_scores = []
                for idx in group_indices:
                    sample = df_debiased.loc[idx]
                    # Calculate how much this sample matches the overall distributions
                    # (lower score = more representative, higher score = more biased)
                    score = 0
                    for label in LABELS:
                        overall_rate = df[label].mean()
                        score += abs(sample[label] - overall_rate)
                    bias_scores.append(score)
                index_scores = list(zip(group_indices, bias_scores))
                index_scores.sort(key=lambda x: x[1])

                # Keep more representative samples (lowest bias)
                samples_to_keep = index_scores[:target_count]
                keep_indices = [idx for idx, _ in samples_to_keep]
                return keep_indices
            else:
                target_count_int = int(target_count)
                group_indices_np = np.array(group_indices, dtype=np.int64)
                keep_indices = np.random.choice(group_indices_np,
                                                size=target_count_int,
                                                replace=False)
                return keep_indices.tolist()
        groups_to_undersample = {group: target for group, target in target_counts.items()
                                 if attr_counts['protected_combination'][group] > median_count}

        if groups_to_undersample:
            print("Applying undersampling to overrepresented groups using multi-threading...")
            indices_to_keep = []
            for group in df_debiased['protected_combination'].unique():
                if group not in groups_to_undersample:
                    group_indices = df_debiased[df_debiased['protected_combination'] == group].index.tolist()
                    indices_to_keep.extend(group_indices)

            undersample_tasks = list(groups_to_undersample.items())

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, len(undersample_tasks))) as executor:
                for keep_indices in executor.map(process_group_undersampling, undersample_tasks):
                    indices_to_keep.extend(keep_indices)

            df_debiased = df_debiased.loc[indices_to_keep].reset_index(drop=True)
            print(f"  Reduced dataset size through undersampling to {len(df_debiased)} samples")

        if 'protected_combination' in df_debiased.columns:
            df_debiased = df_debiased.drop(columns=['protected_combination'])

        print(f"Original dataset size: {len(df)}, Debiased dataset size: {len(df_debiased)}")
        return df_debiased

    return df