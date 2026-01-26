# main.py
import torch
import numpy as np
import pandas as pd
import argparse
import os
import time
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

from config import Config
from data_loader import HeteroNetworkData, CircDiseaseDataset
from metapath_utils import convert_metapaths_to_torch
from evaluation import evaluate_model
from visualization import (
    plot_learning_curves, plot_roc_curve, plot_pr_curve,
    plot_confusion_matrix, plot_metapath_importance
)
from adamh_rl_dcl import AdaMHRLDCL


def parse_args():
    parser = argparse.ArgumentParser(description='AdaMH-RL-DCL for circRNA-disease association prediction')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--k_fold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_bce_loss = 0
    total_dcl_loss = 0

    for batch in train_loader:
        circ_idx = batch['circ_idx'].to(device)
        disease_idx = batch['disease_idx'].to(device)
        labels = batch['label'].float().to(device)

        optimizer.zero_grad()

        # Calculate loss
        loss, bce_loss, dcl_loss = model.calculate_loss(circ_idx, disease_idx, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_dcl_loss += dcl_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_bce_loss = total_bce_loss / len(train_loader)
    avg_dcl_loss = total_dcl_loss / len(train_loader)

    return {
        'loss': avg_loss,
        'bce_loss': avg_bce_loss,
        'dcl_loss': avg_dcl_loss
    }

def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            circ_idx = batch['circ_idx'].to(device)
            disease_idx = batch['disease_idx'].to(device)
            labels = batch['label'].float().to(device)

            # Forward pass
            pred = model(circ_idx, disease_idx)
            loss = torch.nn.functional.binary_cross_entropy(pred, labels)

            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    pred_labels = (all_preds > 0.5).astype(int)

    auroc = roc_auc_score(all_labels, all_preds)
    aupr = average_precision_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)
    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    mcc = matthews_corrcoef(all_labels, pred_labels)

    avg_loss = total_loss / len(val_loader)

    metrics = {
        'loss': avg_loss,
        'auroc': auroc,
        'aupr': aupr,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc
    }

    # Add reward to RL agent
    reward = model.add_rl_reward(metrics)
    metrics['rl_reward'] = reward

    return metrics, all_preds, all_labels


def train_model(model, train_loader, val_loader, optimizer, device, config):
    """Train model for multiple epochs"""
    best_val_auroc = 0
    best_model_state = None
    train_metrics_history = {
        'loss': [], 'auroc': [], 'aupr': [], 'f1': [], 'accuracy': [],
        'precision': [], 'recall': [], 'mcc': []
    }
    val_metrics_history = {
        'loss': [], 'auroc': [], 'aupr': [], 'f1': [], 'accuracy': [],
        'precision': [], 'recall': [], 'mcc': []
    }

    for epoch in range(1, config.epochs + 1):
        print(f"\n==== Epoch {epoch}/{config.epochs} ====")

        # Train for one epoch
        train_info = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate on validation set
        val_metrics, val_preds, val_labels = validate(model, val_loader, device)

        # Add reward to RL agent based on validation metrics
        reward = model.add_rl_reward(val_metrics)

        # Now update the RL policy after adding the reward
        rl_update_info = model.rl_selector.update()

        # Evaluate on training set
        train_metrics, _, _ = validate(model, train_loader, device)

        # Update metrics history
        for key in train_metrics_history:
            if key in train_metrics:
                train_metrics_history[key].append(train_metrics[key])

        for key in val_metrics_history:
            if key in val_metrics:
                val_metrics_history[key].append(val_metrics[key])

        # Print progress
        print(
            f"  Train Loss: {train_info['loss']:.4f}, BCE: {train_info['bce_loss']:.4f}, DCL: {train_info['dcl_loss']:.4f}")
        print(
            f"  Train AUROC: {train_metrics['auroc']:.4f}, AUPR: {train_metrics['aupr']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(
            f"  Val Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  RL Reward: {reward:.4f}")

        # Print metapath importance
        importance = model.rl_selector.metapath_importance
        metapath_importance_str = ", ".join([f"{i:.3f}" for i in importance])
        print(f"  Metapath Importance: [{metapath_importance_str}]")

        # Save best model
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_model_state = model.state_dict().copy()
            print(f"  New best model with AUROC: {best_val_auroc:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_metrics_history, val_metrics_history


def main():
    # Parse arguments
    args = parse_args()

    # Update config with args
    config = Config()
    config.epochs = args.epochs
    config.lr = args.lr
    config.batch_size = args.batch_size
    config.embed_dim = args.embed_dim
    config.hidden_dim = args.hidden_dim
    config.k_fold = args.k_fold
    config.seed = args.seed

    # Set random seed
    set_seed(config.seed)

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = HeteroNetworkData(config)
    data.check_data_dimensions()

    # Generate metapaths
    print("Generating metapaths...")
    metapaths = data.generate_metapaths()
    torch_metapaths = convert_metapaths_to_torch(metapaths, device)

    # Prepare for k-fold cross-validation
    kf = KFold(n_splits=config.k_fold, shuffle=True, random_state=config.seed)

    # Get all circRNA-disease pairs and labels
    pos_pairs, neg_pairs = data._get_pos_neg_pairs()
    all_pairs = np.vstack([pos_pairs, neg_pairs])
    all_labels = np.hstack([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])

    # Initialize metrics for all folds
    all_fold_metrics = []
    fold_results = []

    # K-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_pairs)):
        print(f"\n==== Fold {fold + 1}/{config.k_fold} ====")

        # Split train/val
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=config.val_ratio,
            stratify=all_labels[train_idx],
            random_state=config.seed
        )

        # Create datasets
        train_dataset = CircDiseaseDataset(all_pairs[train_idx], all_labels[train_idx])
        val_dataset = CircDiseaseDataset(all_pairs[val_idx], all_labels[val_idx])
        test_dataset = CircDiseaseDataset(all_pairs[test_idx], all_labels[test_idx])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

        # Initialize model
        model = AdaMHRLDCL(
            num_circrnas=data.num_circrnas,
            num_diseases=data.num_diseases,
            num_mirnas=data.num_mirnas,
            metapaths=torch_metapaths,
            config=config,
            device=device
        ).to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Train model
        print("Training model...")
        model, train_metrics_history, val_metrics_history = train_model(
            model, train_loader, val_loader, optimizer, device, config
        )

        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics, test_preds, test_labels = validate(model, test_loader, device)
        print(f"Test Results:")
        print(f"  AUROC: {test_metrics['auroc']:.4f}")
        print(f"  AUPR: {test_metrics['aupr']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  MCC: {test_metrics['mcc']:.4f}")

        # Save metrics for this fold
        all_fold_metrics.append(test_metrics)
        fold_results.append({
            'fold': fold + 1,
            'auroc': test_metrics['auroc'],
            'aupr': test_metrics['aupr'],
            'accuracy': test_metrics['accuracy'],
            'f1': test_metrics['f1'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'mcc': test_metrics['mcc']
        })

        # Plot learning curves
        plot_learning_curves(
            train_metrics_history, val_metrics_history,
            save_path=os.path.join(args.save_dir, f'learning_curves_fold{fold + 1}.png')
        )

        # Plot ROC and PR curves
        plot_roc_curve(
            test_labels, test_preds,
            save_path=os.path.join(args.save_dir, f'roc_curve_fold{fold + 1}.png')
        )

        plot_pr_curve(
            test_labels, test_preds,
            save_path=os.path.join(args.save_dir, f'pr_curve_fold{fold + 1}.png')
        )

        # Plot confusion matrix
        plot_confusion_matrix(
            test_labels, (test_preds > 0.5).astype(int),
            save_path=os.path.join(args.save_dir, f'confusion_matrix_fold{fold + 1}.png')
        )

        # Plot metapath importance
        metapath_names = list(metapaths.keys())
        metapath_importance = model.rl_selector.metapath_importance
        plot_metapath_importance(
            metapath_names, metapath_importance,
            save_path=os.path.join(args.save_dir, f'metapath_importance_fold{fold + 1}.png')
        )

        # Save model
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_fold{fold + 1}.pt'))

    # Calculate average metrics across all folds
    avg_metrics = {key: np.mean([fold[key] for fold in all_fold_metrics]) for key in all_fold_metrics[0]}
    std_metrics = {key: np.std([fold[key] for fold in all_fold_metrics]) for key in all_fold_metrics[0]}

    print("\n==== Average Results Across All Folds ====")
    for key in ['auroc', 'aupr', 'accuracy', 'f1', 'precision', 'recall', 'mcc']:
        print(f"{key.upper()}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")

    # Save detailed fold results to CSV
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(os.path.join(args.save_dir, 'fold_results.csv'), index=False)

    # Save average results to CSV
    results_df = pd.DataFrame({
        'Metric': list(avg_metrics.keys()),
        'Mean': list(avg_metrics.values()),
        'Std': list(std_metrics.values())
    })
    results_df.to_csv(os.path.join(args.save_dir, 'average_results.csv'), index=False)

    # Create a summary table for the paper
    summary_metrics = ['auroc', 'aupr', 'accuracy', 'f1', 'precision', 'recall', 'mcc']
    summary_data = {
        'Metric': [m.upper() for m in summary_metrics],
        'Mean': [f"{avg_metrics[m]:.4f}" for m in summary_metrics],
        'Std': [f"{std_metrics[m]:.4f}" for m in summary_metrics],
        'Fold 1': [f"{fold_results[0][m]:.4f}" for m in summary_metrics],
        'Fold 2': [f"{fold_results[1][m]:.4f}" for m in summary_metrics],
        'Fold 3': [f"{fold_results[2][m]:.4f}" for m in summary_metrics],
        'Fold 4': [f"{fold_results[3][m]:.4f}" for m in summary_metrics],
        'Fold 5': [f"{fold_results[4][m]:.4f}" for m in summary_metrics],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(args.save_dir, 'summary_results.csv'), index=False)

    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()