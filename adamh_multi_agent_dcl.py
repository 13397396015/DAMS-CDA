# model/adamh_multi_agent_dcl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multi_agent_rl_module import MultiAgentMetapathSelector
from dcl_module import DiffusionContrastiveLearning
from layers import MetapathAggregation


class AdaMHMultiAgentDCL(nn.Module):
    def __init__(self, num_circrnas, num_diseases, num_mirnas, metapaths, config, device):
        super(AdaMHMultiAgentDCL, self).__init__()
        self.num_circrnas = num_circrnas
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.metapaths = metapaths
        self.config = config
        self.device = device

        # Get metapath names list
        self.metapath_names = list(metapaths.keys())

        # Node embeddings
        self.circ_embedding = nn.Embedding(num_circrnas, config.embed_dim)
        self.disease_embedding = nn.Embedding(num_diseases, config.embed_dim)
        self.mirna_embedding = nn.Embedding(num_mirnas, config.embed_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.circ_embedding.weight)
        nn.init.xavier_uniform_(self.disease_embedding.weight)
        nn.init.xavier_uniform_(self.mirna_embedding.weight)

        # Group metapaths into short and long categories
        # Short: CMC, DMD, CMD, DMC (2-3 steps)
        # Long: CDMC, DCMD (4+ steps)
        metapath_groups = {
            'short': ['CMC', 'DMD', 'CMD', 'DMC'],
            'long': ['CDMC', 'DCMD']
        }

        # Verify all metapath names exist in the provided metapaths
        for group, paths in metapath_groups.items():
            valid_paths = []
            for path in paths:
                if path in self.metapath_names:
                    valid_paths.append(path)
                else:
                    print(f"Warning: Metapath {path} not found in provided metapaths. Removed from {group} group.")
            metapath_groups[group] = valid_paths

        # Initialize Multi-Agent RL Metapath Selector
        self.multi_agent_selector = MultiAgentMetapathSelector(
            config=config,
            metapath_groups=metapath_groups,
            device=device
        )

        # Diffusion Contrastive Learning
        self.dcl = DiffusionContrastiveLearning(
            embed_dim=config.embed_dim,
            diffusion_steps=config.diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            temperature=config.temperature,
            device=device
        )

        # Metapath Aggregation
        self.metapath_aggregation = MetapathAggregation(
            embed_dim=config.embed_dim,
            num_metapaths=len(metapaths),
            num_heads=config.num_heads
        )

        # MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def get_node_embeddings(self):
        """Get all node embeddings"""
        circ_emb = self.circ_embedding.weight
        disease_emb = self.disease_embedding.weight
        mirna_emb = self.mirna_embedding.weight

        return circ_emb, disease_emb, mirna_emb

    def get_metapath_embeddings(self, circ_idx=None, disease_idx=None):
        """
        Get embeddings from different metapaths
        Args:
            circ_idx: circRNA indices [batch_size]
            disease_idx: disease indices [batch_size]
        Returns:
            List of metapath embeddings for circRNAs and diseases
        """
        circ_emb, disease_emb, mirna_emb = self.get_node_embeddings()

        # If specific indices are provided, extract those embeddings
        if circ_idx is not None:
            circ_emb_batch = circ_emb[circ_idx]
        else:
            circ_emb_batch = circ_emb

        if disease_idx is not None:
            disease_emb_batch = disease_emb[disease_idx]
        else:
            disease_emb_batch = disease_emb

        # Get metapath embeddings
        circ_metapath_embs = []
        disease_metapath_embs = []

        metapath_metrics = {}

        for i, (name, adj) in enumerate(self.metapaths.items()):
            if name == 'CMC':
                # CircRNA-miRNA-CircRNA
                circ_emb_mp = torch.matmul(adj, circ_emb)
                disease_emb_mp = disease_emb_batch  # No change for disease

            elif name == 'DMD':
                # Disease-miRNA-Disease
                circ_emb_mp = circ_emb_batch  # No change for circRNA
                disease_emb_mp = torch.matmul(adj, disease_emb)

            elif name == 'CMD':
                # CircRNA-miRNA-Disease
                circ_emb_mp = torch.matmul(adj, disease_emb)
                disease_emb_mp = torch.matmul(adj.t(), circ_emb)

            elif name == 'DMC':
                # Disease-miRNA-CircRNA
                circ_emb_mp = torch.matmul(adj.t(), disease_emb)
                disease_emb_mp = torch.matmul(adj, circ_emb)

            elif name == 'CDMC':
                # CircRNA-Disease-miRNA-CircRNA path
                # Use the provided adjacency matrix for CircRNA-Disease-miRNA-CircRNA
                circ_emb_mp = torch.matmul(adj, circ_emb)
                disease_emb_mp = disease_emb_batch  # No change for disease

            elif name == 'DCMD':
                # Disease-CircRNA-miRNA-Disease path
                # Use the provided adjacency matrix for Disease-CircRNA-miRNA-Disease
                circ_emb_mp = circ_emb_batch  # No change for circRNA
                disease_emb_mp = torch.matmul(adj, disease_emb)

            # Extract batch embeddings if needed
            if circ_idx is not None and circ_emb_mp.shape[0] != len(circ_idx):
                circ_emb_mp = circ_emb_mp[circ_idx]

            if disease_idx is not None and disease_emb_mp.shape[0] != len(disease_idx):
                disease_emb_mp = disease_emb_mp[disease_idx]

            # Store metapath embeddings
            circ_metapath_embs.append(circ_emb_mp)
            disease_metapath_embs.append(disease_emb_mp)

            # Calculate metapath metrics for RL
            performance = torch.norm(circ_emb_mp, dim=1).mean().item() + torch.norm(disease_emb_mp, dim=1).mean().item()
            sparsity = torch.count_nonzero(adj).item() / (adj.shape[0] * adj.shape[1])

            metapath_metrics[i] = {
                'performance': performance,
                'sparsity': sparsity
            }

        return circ_metapath_embs, disease_metapath_embs, metapath_metrics

    def select_metapaths(self, metapath_metrics):
        """
        Select metapaths using the multi-agent RL approach

        Args:
            metapath_metrics: Metrics for each metapath
        Returns:
            Importance scores for each metapath
        """
        # Use multi-agent selector to choose the best metapath
        _, action_probs = self.multi_agent_selector.select_metapath(metapath_metrics)
        importance_scores = self.multi_agent_selector.update_importance(action_probs)

        return importance_scores

    def enhance_embeddings(self, embeddings, adjacency=None):
        """
        Enhance embeddings using DCL

        Args:
            embeddings: Node embeddings to enhance
            adjacency: Optional adjacency matrix for defining positive pairs
        Returns:
            Enhanced embeddings and DCL loss
        """
        enhanced_embeddings, dcl_loss = self.dcl(embeddings, adjacency)
        return enhanced_embeddings, dcl_loss

    def forward(self, circ_idx, disease_idx):
        """
        Forward pass
        Args:
            circ_idx: circRNA indices [batch_size]
            disease_idx: disease indices [batch_size]
        Returns:
            Association probability
        """
        # Get metapath embeddings
        circ_metapath_embs, disease_metapath_embs, metapath_metrics = self.get_metapath_embeddings(circ_idx,
                                                                                                   disease_idx)

        # Select metapaths using multi-agent RL
        importance_scores = self.select_metapaths(metapath_metrics)

        # Apply importance scores
        weighted_circ_embs = []
        weighted_disease_embs = []

        for i in range(len(circ_metapath_embs)):
            weighted_circ_embs.append(circ_metapath_embs[i] * importance_scores[i])
            weighted_disease_embs.append(disease_metapath_embs[i] * importance_scores[i])

        # Aggregate metapath embeddings
        circ_embs = torch.stack(weighted_circ_embs, dim=1)  # [batch_size, num_metapaths, embed_dim]
        disease_embs = torch.stack(weighted_disease_embs, dim=1)  # [batch_size, num_metapaths, embed_dim]

        # Add sequence length dimension for attention mechanism
        circ_embs = circ_embs.unsqueeze(2)  # [batch_size, num_metapaths, 1, embed_dim]
        disease_embs = disease_embs.unsqueeze(2)  # [batch_size, num_metapaths, 1, embed_dim]

        # Debug prints
        print(f"circ_embs shape before aggregation: {circ_embs.shape}")

        # Apply multi-head attention
        circ_embs, _ = self.metapath_aggregation([circ_embs])
        disease_embs, _ = self.metapath_aggregation([disease_embs])

        # Take first token as the representation
        circ_embs = circ_embs[:, 0, :]
        disease_embs = disease_embs[:, 0, :]

        # Enhance embeddings using DCL
        circ_embs, circ_dcl_loss = self.enhance_embeddings(circ_embs)
        disease_embs, disease_dcl_loss = self.enhance_embeddings(disease_embs)

        # Concatenate embeddings
        combined_embs = torch.cat([circ_embs, disease_embs], dim=1)

        # Predict association
        logits = self.mlp(combined_embs)
        prob = torch.sigmoid(logits)

        return prob.squeeze()

    def calculate_loss(self, circ_idx, disease_idx, labels):
        """
        Calculate total loss
        Args:
            circ_idx: circRNA indices [batch_size]
            disease_idx: disease indices [batch_size]
            labels: Ground truth labels [batch_size]
        Returns:
            Total loss
        """
        # Get predictions
        pred = self.forward(circ_idx, disease_idx)

        # BCE loss
        bce_loss = F.binary_cross_entropy(pred, labels)

        # Get metapath embeddings for DCL
        circ_metapath_embs, disease_metapath_embs, _ = self.get_metapath_embeddings(circ_idx, disease_idx)

        # Combine all metapath embeddings
        circ_embs = torch.stack(circ_metapath_embs, dim=1).mean(dim=1)
        disease_embs = torch.stack(disease_metapath_embs, dim=1).mean(dim=1)

        # Apply DCL
        _, circ_dcl_loss = self.enhance_embeddings(circ_embs)
        _, disease_dcl_loss = self.enhance_embeddings(disease_embs)
        dcl_loss = circ_dcl_loss + disease_dcl_loss

        # Total loss
        total_loss = bce_loss + self.config.lambda_dcl * dcl_loss

        return total_loss, bce_loss, dcl_loss

    def add_rl_reward(self, val_metrics, sparsity_factor=0.1):
        """
        Add reward to RL agents based on validation performance
        Args:
            val_metrics: Validation metrics
            sparsity_factor: Weight for sparsity reward
        Returns:
            Total reward value
        """
        # Performance reward based on validation metrics
        performance_reward = val_metrics['auroc'] + val_metrics['aupr']

        # Get metapath importance scores
        metapath_weights = self.multi_agent_selector.metapath_importance

        # Sparsity reward based on metapath weights
        sparsity_reward = -sparsity_factor * np.sum(metapath_weights * np.log(metapath_weights + 1e-10))

        # Total reward
        total_reward = performance_reward + sparsity_reward

        # Add reward to multi-agent RL system
        reward_added = self.multi_agent_selector.add_reward(total_reward)

        if reward_added:
            print(
                f"Added RL reward: {total_reward:.4f} (AUROC: {val_metrics['auroc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Sparsity: {sparsity_reward:.4f})")

            # Get agent selection statistics for logging
            agent_stats = self.multi_agent_selector.get_agent_selection_statistics()
            print(f"Current agent: {agent_stats['current_agent']}, Strategy: {agent_stats['strategy']}")

            # Print importance by group
            for group_name in agent_stats['metapath_importance']:
                print(f"  {group_name}: {agent_stats['metapath_importance'][group_name]:.4f}")
        else:
            print(f"Failed to add RL reward - no actions in buffer")

        return total_reward

    def update_rl_agents(self):
        """
        Update the RL agents

        Returns:
            Dictionary of update information
        """
        update_info = self.multi_agent_selector.update()
        return update_info