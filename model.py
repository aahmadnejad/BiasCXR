import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
import numpy as np
from sklearn.metrics import auc as AUC_S
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import os
import json

from store import Configs
from plotting import plot_label_auc_scores

configs = Configs()


class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, writer, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

    torch.backends.cudnn.benchmark = True
    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            features, labels = features.to(device), labels.to(device)
            print(f"GPU Memory usage: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            batch_count += 1

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)

                val_preds.append(outputs.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)

        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)

        aucs = []
        for i in range(len(configs.LABELS)):
            if len(np.unique(val_labels[:, i])) > 1:
                auc = roc_auc_score(val_labels[:, i], val_preds[:, i])
                aucs.append(auc)
            else:
                aucs.append(float('nan'))

        mean_auc = np.nanmean(aucs)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('AUC/mean', mean_auc, epoch)

        for i, label in enumerate(configs.LABELS):
            writer.add_scalar(f'AUC/{label}', aucs[i], epoch)

        plot_label_auc_scores(aucs, epoch, configs.OUTPUT_DIR)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Mean AUC: {mean_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU Memory after epoch: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

    return model


def evaluate_model(model, test_loader, result_prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)

            test_preds.append(outputs.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_preds = np.vstack(test_preds)
    test_labels = np.vstack(test_labels)

    results = {}

    for i, label in enumerate(configs.LABELS):
        # ROC AUC
        if len(np.unique(test_labels[:, i])) > 1:
            auc = roc_auc_score(test_labels[:, i], test_preds[:, i])

            preds_binary = (test_preds[:, i] >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(test_labels[:, i], preds_binary, labels=[0, 1]).ravel()

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            precision, recall, _ = precision_recall_curve(test_labels[:, i], test_preds[:, i])
            pr_auc = AUC_S(recall, precision)

            fpr_curve, tpr_curve, _ = roc_curve(test_labels[:, i], test_preds[:, i])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_curve, tpr_curve, label=f'AUC = {auc:.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {label}')
            plt.legend()
            plt.savefig(os.path.join(configs.OUTPUT_DIR, f'{result_prefix}_roc_{label}.png'))
            plt.close()

            results[label] = {
                'auc': auc,
                'pr_auc': pr_auc,
                'false_positive_rate': fpr,
                'false_negative_rate': fnr,
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        else:
            results[label] = {
                'auc': float('nan'),
                'pr_auc': float('nan'),
                'false_positive_rate': float('nan'),
                'false_negative_rate': float('nan')
            }

    with open(os.path.join(configs.OUTPUT_DIR, f'{result_prefix}_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    aucs = [v['auc'] for v in results.values() if not np.isnan(v['auc'])]
    fprs = [v['false_positive_rate'] for v in results.values() if not np.isnan(v['false_positive_rate'])]
    fnrs = [v['false_negative_rate'] for v in results.values() if not np.isnan(v['false_negative_rate'])]

    print(f"[{result_prefix}] Average AUC: {np.mean(aucs):.4f}")
    print(f"[{result_prefix}] Average False Positive Rate: {np.mean(fprs):.4f}")
    print(f"[{result_prefix}] Average False Negative Rate: {np.mean(fnrs):.4f}")

    return results


