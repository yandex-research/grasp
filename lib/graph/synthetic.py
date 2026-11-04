import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleConv(nn.Module):
    def __init__(self, d_inp: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(2 * d_inp, d_out)

    def forward(
        self,
        graph: dgl.DGLGraph,
        x: Tensor,
    ) -> Tensor:
        message = dgl.ops.copy_u_mean(graph, x)  # type: ignore
        x = torch.cat([x, message], dim=-1)
        x = self.linear(x)
        return x


def _normalize(x: Tensor) -> Tensor:
    """Normalize tensor to zero mean and unit variance."""
    return (x - x.mean()) / x.std()


@torch.no_grad()
def _apply_random_mlp(
    x: Tensor,
    output_dim: int,
    hidden_dim: int,
    n_layers: int,
) -> Tensor:
    """Apply a randomly initialized MLP to input features."""
    input_dim = x.shape[-1]

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))

    mlp = nn.Sequential(*layers).to(x.device)
    return mlp(x).squeeze(-1).detach()


@torch.no_grad()
def _compute_structure_target(
    graph: dgl.DGLGraph,
    n_samples: int,
    hidden_dim: int,
    batch_size: int = 1_000,
) -> Tensor:
    """Compute structure-dependent target via random GNN.

    Applies the same random 2-layer GNN to multiple random inputs and averages.
    """
    assert n_samples % batch_size == 0

    device = graph.device
    n_nodes = graph.num_nodes()

    conv1 = SimpleConv(1, hidden_dim).to(device)
    conv2 = SimpleConv(hidden_dim, 1).to(device)

    targets = []

    for _ in range(n_samples // batch_size):
        x = torch.randn((n_nodes, batch_size, 1), device=device)
        x = F.relu(conv1(graph, x))
        x = conv2(graph, x)
        targets.append(x)

    return torch.cat(targets, dim=1).mean(1).squeeze()


@torch.no_grad()
def generate_synthetic_features(
    graph: dgl.DGLGraph,
    n_features: int,
    seed: int,
) -> Tensor:
    """Generate node features correlated with graph structure.

    Features are generated as:
    1. Random features -> MLP -> normalize
    2. Add normalized neighbor aggregation (message passing)

    This creates features that depend on local graph structure.
    """
    torch.manual_seed(seed)

    device = graph.device
    n_nodes = graph.num_nodes()

    # Random features through MLP
    features = torch.randn(n_nodes, n_features, device=device)
    features = _apply_random_mlp(
        features,
        output_dim=n_features,
        hidden_dim=n_features,
        n_layers=1,
    )
    features = (features - features.mean(0)) / features.std(0)

    # Add neighbor aggregation (normalized by sqrt of degree)
    degrees = graph.in_degrees()[:, None].float()
    message = dgl.ops.copy_u_sum(graph, features)  # type: ignore
    message = torch.where(degrees > 0, message / (degrees**0.5), 0.0)

    # Combine original and aggregated features
    features = (0.5**0.5) * (features + message)

    return features


@torch.no_grad()
def generate_synthetic_targets(
    graph: dgl.DGLGraph,
    features: Tensor,
    feat_coef: float,
    struc_coef: float,
    nfa_coef: float,
    noise_coef: float,
    seed: int,
    *,
    struc_n_samples: int = 10_000,
    struc_hidden_dim: int = 32,
) -> Tensor:
    """Generate synthetic regression targets.

    Target is a weighted combination of:
    - Feature-based component: MLP(features)
    - Structure-based component: random GNN on random inputs
    - NFA-based component: MLP(mean neighbor features)
    - Noise component: random noise

    All components are normalized before combining.
    """
    torch.manual_seed(seed)

    device = graph.device
    n_nodes = graph.num_nodes()
    hidden_dim = features.shape[-1] * 4

    # Feature-based target
    feat_target = _apply_random_mlp(
        features, output_dim=1, hidden_dim=hidden_dim, n_layers=3
    )
    feat_target = _normalize(feat_target)

    # Structure-based target (random GNN)
    struc_target = _compute_structure_target(
        graph, n_samples=struc_n_samples, hidden_dim=struc_hidden_dim
    )
    struc_target = _normalize(struc_target)

    # NFA-based target (neighbor feature aggregation)
    nfa = dgl.ops.copy_u_mean(graph, features)  # type: ignore
    nfa_target = _apply_random_mlp(nfa, output_dim=1, hidden_dim=hidden_dim, n_layers=3)
    nfa_target = _normalize(nfa_target)

    # Noise
    noise = torch.randn(n_nodes, device=device)

    # Combine all components
    target = (
        feat_coef * feat_target
        + struc_coef * struc_target
        + nfa_coef * nfa_target
        + noise_coef * noise
    )

    return _normalize(target)


def generate_synthetic_splits(
    n_nodes: int,
    seed: int,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate random train/val/test splits for synthetic data.

    Args:
        n_nodes: Number of nodes in the graph.
        seed: Random seed for reproducibility.
        train_ratio: Fraction of nodes for training (default: 0.1).
        val_ratio: Fraction of nodes for validation (default: 0.1).

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to boolean masks.
    """
    rng = np.random.default_rng(seed)

    indices = np.arange(n_nodes)
    rng.shuffle(indices)

    n_train = int(n_nodes * train_ratio)
    n_val = int(n_nodes * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    masks: dict[str, np.ndarray] = {
        "train": np.zeros(n_nodes, dtype=bool),
        "val": np.zeros(n_nodes, dtype=bool),
        "test": np.zeros(n_nodes, dtype=bool),
    }
    masks["train"][train_indices] = True
    masks["val"][val_indices] = True
    masks["test"][test_indices] = True

    return masks
