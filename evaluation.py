# utils/evaluation.py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, \
    recall_score


def calculate_metrics(y_true, y_pred, y_score):
    """Calculate evaluation metrics"""
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {
        'auroc': auroc,
        'aupr': aupr,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def evaluate_model(model, data_loader, device):
    """Evaluate model on data loader"""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in data_loader:
            circ_idx = batch['circ_idx'].to(device)
            disease_idx = batch['disease_idx'].to(device)
            labels = batch['label'].to(device)

            scores = model(circ_idx, disease_idx)
            preds = (scores > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    metrics = calculate_metrics(all_labels, all_preds, all_scores)
    return metrics, all_preds, all_scores