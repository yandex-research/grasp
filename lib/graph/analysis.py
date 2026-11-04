import typing as tp

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold
from torch import Tensor

TargetType = tp.Literal["features", "edges"]
ModelType = tp.Literal["linear", "mlp"]
ModeType = tp.Literal["regression", "classification"]


class ProbingConfig(tp.TypedDict):
    """Configuration for probing model training and evaluation."""

    n_splits: int
    model_type: ModelType
    lr: float
    alpha: float
    max_iter: int
    seed: int


class AnalysisConfig(tp.TypedDict):
    """Configuration for probing analysis."""

    targets: list[TargetType]
    seed: tp.NotRequired[int]  # Seed for edge sampling (separate from probing seed)
    probing_config: tp.NotRequired[ProbingConfig]


DEFAULT_PROBING_CONFIG: ProbingConfig = {
    "n_splits": 3,
    "model_type": "mlp",
    "lr": 1e-3,
    "alpha": 1e-3,
    "max_iter": 1000,
    "seed": 42,
}


# >>> Probe model


def get_probe_model(d_in: int, d_out: int, model_type: ModelType) -> nn.Module:
    if model_type == "linear":
        return nn.Linear(d_in, d_out)
    elif model_type == "mlp":
        d_hidden = 2 * d_in
        return nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
    else:
        raise ValueError(f"Unknown {model_type=}")


def fit_probe(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    criterion: tp.Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    lr: float = 1e-3,
    alpha: float = 1e-3,
    max_iter: int = 1000,
) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)  # type: ignore

    x = x.to(device).float()
    y = y.to(device).float()

    model.train()
    for _ in range(max_iter):
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    return model


# >>> Probing score


def _filter_unrepresented_targets(
    target_train: Tensor, target_test: Tensor
) -> tuple[Tensor, Tensor]:
    """Filter out target columns with inconsistent classes between train/test."""
    unrepresented_mask = torch.zeros(target_train.shape[-1], dtype=torch.bool)

    for target_id in range(unrepresented_mask.shape[0]):
        train_unique = target_train[:, target_id].unique().tolist()
        test_unique = target_test[:, target_id].unique().tolist()

        if len(train_unique) == 1 or set(train_unique) != set(test_unique):
            unrepresented_mask[target_id] = True

    return target_train[:, ~unrepresented_mask], target_test[:, ~unrepresented_mask]


def _normalize_regression_targets(
    target_train: Tensor,
    target_test: Tensor,
    eps: float = 1e-9,
) -> tuple[Tensor, Tensor]:
    """Normalize regression targets and filter zero-variance columns."""
    mean = target_train.mean(0)
    std = target_train.std(0)
    mask = std >= eps

    target_train = target_train[:, mask]
    target_test = target_test[:, mask]
    mean, std = mean[mask], std[mask]

    return (target_train - mean) / std, (target_test - mean) / std


def compute_probing_score(
    hidden: Tensor,
    target: Tensor,
    mode: ModeType,
    device: torch.device,
    n_splits: int = 3,
    model_type: ModelType = "mlp",
    lr: float = 1e-3,
    alpha: float = 1e-3,
    max_iter: int = 1000,
    seed: int = 42,
) -> float:
    """Compute probing score via K-Fold cross-validation.

    Args:
        hidden: Node embeddings [n_nodes, d_hidden]
        target: Target values [n_nodes] or [n_nodes, n_targets]
        mode: "regression" (R2) or "classification" (ROC-AUC)
        device: Torch device
        n_splits: Number of K-Fold splits
        model_type: "linear" or "mlp"
        lr: Learning rate
        alpha: Weight decay
        max_iter: Max training iterations
        seed: Random seed for K-Fold

    Returns:
        Mean score across folds (R2 for regression, ROC-AUC for classification)
    """
    if target.ndim == 1:
        target = target.unsqueeze(-1)

    # L2-normalize embeddings
    hidden = hidden / torch.linalg.norm(hidden, dim=-1, keepdim=True)

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in folds.split(hidden):  # type: ignore
        hidden_train, hidden_test = hidden[train_idx], hidden[test_idx]
        target_train, target_test = target[train_idx], target[test_idx]

        if mode == "classification":
            target_train, target_test = _filter_unrepresented_targets(
                target_train, target_test
            )
            criterion = F.binary_cross_entropy_with_logits
            metric = roc_auc_score
        elif mode == "regression":
            target_train, target_test = _normalize_regression_targets(
                target_train, target_test
            )
            criterion = F.mse_loss
            metric = r2_score
        else:
            raise ValueError(f"Unknown {mode=}")

        if target_train.shape[-1] == 0:
            continue

        model = get_probe_model(
            d_in=hidden_train.shape[-1],
            d_out=target_train.shape[-1],
            model_type=model_type,
        )
        model = fit_probe(
            model=model,
            x=hidden_train,
            y=target_train,
            criterion=criterion,
            device=device,
            lr=lr,
            alpha=alpha,
            max_iter=max_iter,
        )

        with torch.no_grad():
            preds_test = model(hidden_test.to(device).float()).cpu().numpy()

        target_test_np = target_test.cpu().numpy()
        fold_scores = [
            metric(target_test_np[:, i], preds_test[:, i])
            for i in range(target_test.shape[-1])
        ]
        scores.append(np.mean(fold_scores))  # type: ignore

    return float(np.mean(scores)) if scores else 0.0


# >>> Target generation


def get_features_target(hidden: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
    """Return features as probing target."""
    return hidden, features


def get_edges_target(
    hidden: Tensor,
    graph: dgl.DGLGraph,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Generate edge prediction target by sampling positive and negative edges.

    Args:
        graph: DGL graph
        hidden: Node embeddings [n_nodes, d_hidden]
        seed: Random seed for sampling

    Returns:
        edge_hidden: Edge representations [2 * n_samples, d_hidden]
        edge_labels: Binary labels (1=positive, 0=negative) [2 * n_samples]
    """
    # Set global seed (DGL sampler uses global state)
    # TODO: thats a really bad pattern, but analysis is only called in the end,
    #       so tecnhically everyting is fine.
    torch.manual_seed(seed)
    dgl.random.seed(seed)

    edges_src, edges_dst = graph.edges()
    n_samples = min(edges_src.shape[0], graph.num_nodes())

    positive_ids = torch.randperm(edges_src.shape[0])[:n_samples]
    positive_src, positive_dst = edges_src[positive_ids], edges_dst[positive_ids]
    positive_hidden = hidden[positive_src] * hidden[positive_dst]

    negative_sampler = dgl.dataloading.negative_sampler.GlobalUniform(1)
    negative_src, negative_dst = negative_sampler(graph, positive_ids)
    negative_hidden = hidden[negative_src] * hidden[negative_dst]

    edge_hidden = torch.cat([positive_hidden, negative_hidden], dim=0)
    edge_labels = torch.cat([torch.ones(n_samples), torch.zeros(n_samples)], dim=0)

    return edge_hidden, edge_labels


# >>> Analysis config and runner


def run_probing_analysis(
    hidden: Tensor,
    graph: dgl.DGLGraph,
    features: Tensor,
    targets: list[TargetType],
    seed: int = 42,
    probing_config: ProbingConfig = DEFAULT_PROBING_CONFIG,
) -> dict[str, float]:
    """Run probing analysis on embeddings.

    Args:
        hidden: Node embeddings [n_nodes, d_hidden]
        graph: DGL graph
        features: Node features [n_nodes, n_features]
        device: Torch device
        config: Probing analysis configuration

    Returns:
        Dict of probing scores: {"features": R2, "edges": ROC-AUC}
    """
    scores: dict[str, float] = {}

    for target_name in targets:
        if target_name == "features":
            probe_hidden, probe_target = get_features_target(hidden, features)
            mode = "regression"
        elif target_name == "edges":
            probe_hidden, probe_target = get_edges_target(hidden, graph, seed=seed)
            mode = "classification"
        else:
            raise ValueError(f"Unknown target: {target_name}")

        scores[target_name] = compute_probing_score(
            hidden=probe_hidden,
            target=probe_target,
            mode=mode,
            device=graph.device,
            **probing_config,
        )

    return scores
