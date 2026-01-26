# utils/metapath_utils.py
import torch


def convert_metapaths_to_torch(metapaths, device):
    """Convert metapath adjacency matrices to torch tensors"""
    torch_metapaths = {}
    for name, adj in metapaths.items():
        torch_metapaths[name] = torch.FloatTensor(adj.toarray()).to(device)
    return torch_metapaths


def get_metapath_neighbors(metapath_adj, nodes, k=10):
    """Get top-k neighbors for each node based on metapath adjacency"""
    # Get neighbor scores
    neighbor_scores = metapath_adj[nodes]

    # Get top-k neighbors
    if k < neighbor_scores.shape[1]:
        _, top_k_indices = torch.topk(neighbor_scores, k, dim=1)
    else:
        # If k is larger than the number of neighbors, return all neighbors
        top_k_indices = torch.arange(neighbor_scores.shape[1]).repeat(len(nodes), 1)

    return top_k_indices


def sample_metapath_walk(metapath_adj, start_node, walk_length=5):
    """Sample a random walk based on metapath adjacency"""
    walk = [start_node]
    current_node = start_node

    for _ in range(walk_length - 1):
        neighbors = torch.nonzero(metapath_adj[current_node]).squeeze()
        if neighbors.dim() == 0:  # Only one neighbor
            if len(neighbors.shape) == 0:  # Handle scalar case
                neighbors = neighbors.unsqueeze(0)
            else:
                break  # No neighbors, end walk

        # Sample next node based on transition probabilities
        probs = metapath_adj[current_node, neighbors]
        probs = probs / probs.sum()
        next_idx = torch.multinomial(probs, 1).item()
        next_node = neighbors[next_idx].item()

        walk.append(next_node)
        current_node = next_node

    return walk