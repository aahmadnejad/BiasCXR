import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve

import matplotlib.pyplot as plt

import json
import argparse

from model import MultiLabelClassifier, train_model, evaluate_model
from store import Configs

from processor import *
from cxr import *
from plotting import *

configs = Configs()

torch.manual_seed(42)
np.random.seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.makedirs(configs.OUTPUT_DIR, exist_ok=True)


def main(csv_path):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(configs.OUTPUT_DIR, 'tensorboard_logs'))

    # Load and preprocess dataset
    print("Loading and preprocessing dataset...")
    df = pd.read_csv(csv_path)
    df = preprocess_dataframe(df)

    # Analyze original demographics
    print("Analyzing demographic distributions...")
    analyze_demographics(df, configs.OUTPUT_DIR)

    # Split dataset before debiasing to ensure consistent test set
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


def main(csv_path):
    writer = SummaryWriter(log_dir=os.path.join(configs.OUTPUT_DIR, 'tensorboard_logs'))

    print("Loading and preprocessing dataset...")
    df = pd.read_csv(csv_path)
    df = preprocess_dataframe(df)
    df = preprocess_labels(df, configs.LABELS)

    print("Analyzing demographic distributions...")
    analyze_demographics(df, configs.OUTPUT_DIR)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    if os.path.exists(os.path.join(configs.OUTPUT_DIR, "debiased_output.csv")):
        debiased_train_df = pd.read_csv(os.path.join(configs.OUTPUT_DIR, "debiased_output.csv"))
    else:
        debiased_train_df = create_debiased_dataframe(train_df, protected_attrs=['gender', 'race', 'anchor_age'])
        debiased_train_df.to_csv(os.path.join(configs.OUTPUT_DIR, "debiased_output.csv"))
    debiased_dir = os.path.join(configs.OUTPUT_DIR, "debiased")
    os.makedirs(debiased_dir, exist_ok=True)
    analyze_demographics(debiased_train_df, debiased_dir)
    print("Creating datasets and dataloaders for original data...")
    train_val_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_dataset = CXRDataset(train_val_df)
    val_dataset = CXRDataset(val_df)
    test_dataset = CXRDataset(test_df)

    sample_features, _ = train_dataset[0]
    input_dim = sample_features.shape[0]

    print(f"Input feature dimension: {input_dim}")
    print(f"Output dimension: {len(configs.LABELS)}")

    train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

    print("Creating model, loss function, and optimizer...")
    model = MultiLabelClassifier(
        input_dim=input_dim,
        hidden_dim1=configs.HIDDEN_DIM_1,
        hidden_dim2=configs.HIDDEN_DIM_2,
        output_dim=len(configs.LABELS)
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)

    print("\n=== Training Original Model ===")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=configs.NUM_EPOCHS,
        writer=writer,
        model_save_path=configs.MODEL_SAVE_PATH
    )

    print("\n=== Evaluating Original Model ===")
    best_model = MultiLabelClassifier(
        input_dim=input_dim,
        hidden_dim1=configs.HIDDEN_DIM_1,
        hidden_dim2=configs.HIDDEN_DIM_2,
        output_dim=len(configs.LABELS)
    )
    best_model.load_state_dict(torch.load(configs.MODEL_SAVE_PATH))

    original_results = evaluate_model(best_model, test_loader, "original")

    print("\n=== Creating datasets and dataloaders for debiased data ===")
    debiased_train_val_df, debiased_val_df = train_test_split(debiased_train_df, test_size=0.2, random_state=42)

    debiased_train_dataset = CXRDataset(debiased_train_val_df)
    debiased_val_dataset = CXRDataset(debiased_val_df)

    debiased_train_loader = DataLoader(debiased_train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True )
    debiased_val_loader = DataLoader(debiased_val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

    debiased_model = MultiLabelClassifier(
        input_dim=input_dim,
        hidden_dim1=configs.HIDDEN_DIM_1,
        hidden_dim2=configs.HIDDEN_DIM_2,
        output_dim=len(configs.LABELS)
    )

    debiased_optimizer = optim.Adam(debiased_model.parameters(), lr=configs.LEARNING_RATE)

    print("\n=== Training Debiased Model ===")
    debiased_trained_model = train_model(
        model=debiased_model,
        train_loader=debiased_train_loader,
        val_loader=debiased_val_loader,
        criterion=criterion,
        optimizer=debiased_optimizer,
        num_epochs=configs.NUM_EPOCHS,
        writer=writer,
        model_save_path=configs.DEBIASED_MODEL_SAVE_PATH
    )

    print("\n=== Evaluating Debiased Model ===")
    best_debiased_model = MultiLabelClassifier(
        input_dim=input_dim,
        hidden_dim1=configs.HIDDEN_DIM_1,
        hidden_dim2=configs.HIDDEN_DIM_2,
        output_dim=len(configs.LABELS)
    )
    best_debiased_model.load_state_dict(torch.load(configs.DEBIASED_MODEL_SAVE_PATH))

    debiased_results = evaluate_model(best_debiased_model, test_loader, "debiased")

    print("\n=== Comparing Model Performances ===")
    compare_model_performance(original_results, debiased_results)

    print("\n=== Analyzing Fairness Across Protected Attributes ===")
    analyze_model_fairness(best_model, best_debiased_model, test_df, test_dataset)
    writer.close()

    print("All processes completed successfully!")


def analyze_model_fairness(original_model, debiased_model, test_df, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model = original_model.to(device)
    debiased_model = debiased_model.to(device)

    original_model.eval()
    debiased_model.eval()

    # Create directories for fairness analysis
    fairness_dir = os.path.join(configs.OUTPUT_DIR, "fairness")
    os.makedirs(fairness_dir, exist_ok=True)

    all_features = []
    all_labels = []
    original_preds = []
    debiased_preds = []

    for idx in range(len(test_dataset)):
        features, labels = test_dataset[idx]
        features = features.unsqueeze(0).to(device)

        with torch.no_grad():
            original_output = original_model(features).cpu().numpy().squeeze()
            debiased_output = debiased_model(features).cpu().numpy().squeeze()

        all_features.append(features.cpu().numpy().squeeze())
        all_labels.append(labels.cpu().numpy())
        original_preds.append(original_output)
        debiased_preds.append(debiased_output)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    original_preds = np.array(original_preds)
    debiased_preds = np.array(debiased_preds)
    original_binary_preds = (original_preds >= 0.5).astype(int)
    debiased_binary_preds = (debiased_preds >= 0.5).astype(int)

    protected_attributes = []
    if 'gender' in test_df.columns:
        protected_attributes.append('gender')
    if 'race' in test_df.columns:
        protected_attributes.append('race')
    if 'anchor_age' in test_df.columns:
        protected_attributes.append('anchor_age')
    fairness_metrics = {}

    for attr in protected_attributes:
        fairness_metrics[attr] = {}
        attr_values = test_df[attr].unique()
        for label_idx, label in enumerate(configs.LABELS):
            fairness_metrics[attr][label] = {
                "original": {},
                "debiased": {}
            }
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))

            orig_fprs = []
            orig_fnrs = []
            debiased_fprs = []
            debiased_fnrs = []
            attr_labels = []

            for val in attr_values:
                val_indices = test_df[test_df[attr] == val].index.values
                val_indices = [i for i in range(len(test_dataset)) if test_df.iloc[i][attr] == val]

                if len(val_indices) == 0:
                    continue
                val_labels = all_labels[val_indices, label_idx]
                val_orig_preds = original_binary_preds[val_indices, label_idx]
                val_debiased_preds = debiased_binary_preds[val_indices, label_idx]

                if len(np.unique(val_labels)) > 1:
                    tn, fp, fn, tp = confusion_matrix(val_labels, val_orig_preds, labels=[0, 1]).ravel()
                    orig_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    orig_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                    tn, fp, fn, tp = confusion_matrix(val_labels, val_debiased_preds, labels=[0, 1]).ravel()
                    debiased_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    debiased_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                    fairness_metrics[attr][label]["original"][val] = {
                        "fpr": orig_fpr,
                        "fnr": orig_fnr
                    }
                    fairness_metrics[attr][label]["debiased"][val] = {
                        "fpr": debiased_fpr,
                        "fnr": debiased_fnr
                    }

                    orig_fprs.append(orig_fpr)
                    orig_fnrs.append(orig_fnr)
                    debiased_fprs.append(debiased_fpr)
                    debiased_fnrs.append(debiased_fnr)
                    attr_labels.append(val)

            if attr_labels:
                x = np.arange(len(attr_labels))
                width = 0.35

                # FPR plot
                ax[0].bar(x - width / 2, orig_fprs, width, label='Original Model')
                ax[0].bar(x + width / 2, debiased_fprs, width, label='Debiased Model')
                ax[0].set_ylabel('False Positive Rate')
                ax[0].set_xlabel(attr)
                ax[0].set_title(f'FPR for {label} by {attr}')
                ax[0].set_xticks(x)
                ax[0].set_xticklabels(attr_labels, rotation=45)
                ax[0].legend()

                # FNR plot
                ax[1].bar(x - width / 2, orig_fnrs, width, label='Original Model')
                ax[1].bar(x + width / 2, debiased_fnrs, width, label='Debiased Model')
                ax[1].set_ylabel('False Negative Rate')
                ax[1].set_xlabel(attr)
                ax[1].set_title(f'FNR for {label} by {attr}')
                ax[1].set_xticks(x)
                ax[1].set_xticklabels(attr_labels, rotation=45)
                ax[1].legend()

                plt.tight_layout()
                plt.savefig(os.path.join(fairness_dir, f'fairness_{attr}_{label}.png'))
                plt.close()

    # Calculate and save fairness disparity metrics
    fairness_disparity = {}

    for attr in protected_attributes:
        fairness_disparity[attr] = {}

        for label in configs.LABELS:
            if label not in fairness_metrics[attr]:
                continue

            # Calculate max disparity in FPR and FNR for both models
            orig_fprs = [metrics.get('fpr', 0) for val, metrics in fairness_metrics[attr][label]['original'].items()]
            orig_fnrs = [metrics.get('fnr', 0) for val, metrics in fairness_metrics[attr][label]['original'].items()]

            debiased_fprs = [metrics.get('fpr', 0) for val, metrics in
                             fairness_metrics[attr][label]['debiased'].items()]
            debiased_fnrs = [metrics.get('fnr', 0) for val, metrics in
                             fairness_metrics[attr][label]['debiased'].items()]

            if orig_fprs and orig_fnrs and debiased_fprs and debiased_fnrs:
                fairness_disparity[attr][label] = {
                    'original': {
                        'fpr_disparity': max(orig_fprs) - min(orig_fprs),
                        'fnr_disparity': max(orig_fnrs) - min(orig_fnrs)
                    },
                    'debiased': {
                        'fpr_disparity': max(debiased_fprs) - min(debiased_fprs),
                        'fnr_disparity': max(debiased_fnrs) - min(debiased_fnrs)
                    },
                    'improvement': {
                        'fpr_disparity': (max(orig_fprs) - min(orig_fprs)) - (max(debiased_fprs) - min(debiased_fprs)),
                        'fnr_disparity': (max(orig_fnrs) - min(orig_fnrs)) - (max(debiased_fnrs) - min(debiased_fnrs))
                    }
                }

    with open(os.path.join(fairness_dir, 'fairness_metrics.json'), 'w') as f:
        json.dump(fairness_metrics, f, indent=4)

    with open(os.path.join(fairness_dir, 'fairness_disparity.json'), 'w') as f:
        json.dump(fairness_disparity, f, indent=4)

    overall_fpr_improvement = []
    overall_fnr_improvement = []
    labels_for_plot = []

    for attr in protected_attributes:
        for label in configs.LABELS:
            if label in fairness_disparity[attr]:
                overall_fpr_improvement.append(fairness_disparity[attr][label]['improvement']['fpr_disparity'])
                overall_fnr_improvement.append(fairness_disparity[attr][label]['improvement']['fnr_disparity'])
                labels_for_plot.append(f"{attr}_{label}")

    if labels_for_plot:
        plt.figure(figsize=(14, 8))
        plt.bar(np.arange(len(labels_for_plot)), overall_fpr_improvement)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Attribute-Label Combinations')
        plt.ylabel('FPR Disparity Reduction')
        plt.title('Fairness Improvement in FPR Disparity (Positive values = Better)')
        plt.xticks(np.arange(len(labels_for_plot)), labels_for_plot, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(fairness_dir, 'overall_fpr_improvement.png'))
        plt.close()

        plt.figure(figsize=(14, 8))
        plt.bar(np.arange(len(labels_for_plot)), overall_fnr_improvement)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Attribute-Label Combinations')
        plt.ylabel('FNR Disparity Reduction')
        plt.title('Fairness Improvement in FNR Disparity (Positive values = Better)')
        plt.xticks(np.arange(len(labels_for_plot)), labels_for_plot, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(fairness_dir, 'overall_fnr_improvement.png'))
        plt.close()

    print("Fairness analysis completed and saved")

def force_cuda_initialize():
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system!")

    print("\n==== GPU VERIFICATION ====")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    test_tensor = torch.ones(10, 10)

    try:
        test_tensor = test_tensor.cuda()
        print(f"Test tensor device: {test_tensor.device}")
        result = test_tensor + test_tensor
        print(f"Test operation successful: {result.shape} on {result.device}")

        del test_tensor
        del result
        torch.cuda.empty_cache()
        print(f"Current GPU memory: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print("✅ CUDA initialized successfully")
        print("==========================\n")

    except Exception as e:
        print(f"❌ CUDA initialization failed: {e}")
        print("==========================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CXR Classification')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file')
    args = parser.parse_args()
    force_cuda_initialize()
    main(args.csv_path)