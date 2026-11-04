import abc
import copy
import math
from enum import Enum
from functools import partial

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import normalize
from torch import Tensor
from torch.optim.swa_utils import AveragedModel

from lib.graph.data import GraphDataset
from lib.graph.deep import get_activation_module
from lib.util import KWArgs


class PretrainMethod(Enum):
    GRACE = "grace"
    BGRL = "bgrl"
    MASK_GAE = "mask_gae"
    GRAPH_MAE = "graph_mae"
    GRAPH_MAE2 = "graph_mae2"
    GRASP = "grasp"
    GRASP_NAIVE = "grasp_naive"
    EPAGCL = "epagcl"


# >>> Edge Samplers


class EdgeSamplerBase(abc.ABC):
    def __init__(self):
        super().__init__()

    def find_reversed_edges(self, graph: dgl.DGLGraph, eids: Tensor) -> Tensor:
        src, dst = graph.find_edges(eids.to(torch.int32))
        return graph.edge_ids(dst, src)

    @abc.abstractmethod
    def __call__(self, graph: dgl.DGLGraph) -> Tensor:
        raise NotImplementedError()


class SimpleEdgeSampler(EdgeSamplerBase):
    def __init__(self, p: float, directed: bool, n_max_samples: int = 1_000_000):
        super().__init__()
        assert directed is not None
        self.p = p
        self.directed = directed
        self.n_max_samples = n_max_samples

    def __call__(self, graph: dgl.DGLGraph) -> Tensor:
        edges = torch.arange(graph.num_edges(), device=graph.device)

        if not self.directed:
            src, dst = graph.edges()
            edges = edges[src < dst]

        sample_size = int(len(edges) * self.p)
        sample_size = min(sample_size, self.n_max_samples)
        sample_ids = torch.randperm(len(edges), device=edges.device)[:sample_size]
        sample = edges[sample_ids]

        if not self.directed:
            sample = torch.cat([sample, self.find_reversed_edges(graph, sample)], dim=0)

        return sample


class RandomWalkEdgeSampler(EdgeSamplerBase):
    def __init__(
        self,
        p_anchor: float,
        n_walks_per_anchor: int,
        walk_len: int,
        directed: bool,
        n_max_samples: int = 1_000_000,
    ):
        super().__init__()
        assert directed is not None
        self.p_anchor = p_anchor
        self.n_walks_per_anchor = n_walks_per_anchor
        self.walk_len = walk_len
        self.directed = directed
        self.n_max_samples = n_max_samples

    def __call__(self, graph: dgl.DGLGraph) -> Tensor:
        n_nodes = graph.num_nodes()
        n_anchors = int(n_nodes * self.p_anchor)
        n_anchors = min(n_anchors, self.n_max_samples // self.n_walks_per_anchor)

        anchors = torch.randperm(n_nodes, device=graph.device)[:n_anchors]

        anchors = anchors.repeat_interleave(self.n_walks_per_anchor)
        _, sample, _ = dgl.sampling.random_walk(
            g=graph,
            nodes=anchors.to(torch.int32),
            length=self.walk_len,
            return_eids=True,
        )

        sample = torch.unique(sample.reshape(-1))
        sample = sample[sample >= 0]

        if not self.directed:
            sample = torch.cat([sample, self.find_reversed_edges(graph, sample)], dim=0)

        return sample


# <<<


# >>> Augmentations


class AugmentationBase(abc.ABC, nn.Module):
    def __init__(self):
        abc.ABC.__init__(self)
        nn.Module.__init__(self)

    @abc.abstractmethod
    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> tuple[dgl.DGLGraph, Tensor | None, Tensor | None]:
        raise NotImplementedError()


class DropEdge(AugmentationBase):
    def __init__(self, p_drop_edge: float, directed_edge_sampling: bool):
        super().__init__()
        self.sampler = SimpleEdgeSampler(p=p_drop_edge, directed=directed_edge_sampling)

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> tuple[dgl.DGLGraph, Tensor | None, Tensor | None]:
        edges = self.sampler(graph).to(torch.int32).to(graph.device)
        augmented_graph = dgl.remove_edges(graph, edges)
        return augmented_graph, x_num, x_other


class MaskFeatures(AugmentationBase):
    def __init__(self, p_mask_feat):
        super().__init__()
        self.p_mask_feat = p_mask_feat

    def mask(self, features: Tensor | None) -> Tensor | None:
        if features is None:
            return features
        n_features = features.shape[-1]
        mask = torch.rand(n_features, device=features.device) < self.p_mask_feat
        features = torch.where(mask.unsqueeze(0), 0.0, features)
        return features

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> tuple[dgl.DGLGraph, Tensor | None, Tensor | None]:
        return graph, self.mask(x_num), self.mask(x_other)


class ComposeAugmentation(AugmentationBase):
    def __init__(self, augmentations: list[AugmentationBase]):
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> tuple[dgl.DGLGraph, Tensor | None, Tensor | None]:
        for augmentation in self.augmentations:
            graph, x_num, x_other = augmentation(graph, x_num, x_other)
        return graph, x_num, x_other


# <<<


# >>> Utils


class GraphMAEDecoder(nn.Module):
    def __init__(self, d: int, d_out: int, activation: str):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d, d),
            get_activation_module(activation),
            nn.Linear(d, d_out),
        )

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        x = dgl.ops.copy_u_mean(graph, x)  # type: ignore
        x = self.ffn(x)
        return x


class GAEHead(nn.Module):
    def __init__(self, d_hidden: int, activation: str):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            get_activation_module(activation),
            nn.Linear(d_hidden, 1),
        )

    def __call__(self, x: Tensor, edges: tuple[Tensor, Tensor]) -> Tensor:
        edges_src, edges_dst = edges
        preds = self.mlp(x[edges_src] * x[edges_dst])
        return preds


def normalize_l2_safe(x: Tensor) -> Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    return torch.where(norm > 0, x / norm, x)


def ema_multi_avg_fn(
    averaged_model_param: Tensor,
    model_param: Tensor,
    n_averaged: int,
    decay_base: float,
    n_total_steps: int,
) -> Tensor:
    decay = 1.0 - 0.5 * (1 - decay_base) * (
        torch.cos(n_averaged * np.pi / n_total_steps) + 1  # type: ignore
    )
    return decay * averaged_model_param + (1.0 - decay) * model_param


def compute_linear_probing_preds(
    hidden: np.ndarray,
    dataset: GraphDataset,
    n_splits: int = 5,
    max_iter: int = 10_000,
    verbose: int = 0,
) -> np.ndarray | None:
    if np.isnan(hidden).any():
        return None

    hidden = normalize(hidden, norm="l2")

    train_mask = dataset.data["masks"]["train"].cpu().numpy()  # type: ignore
    hidden_train = hidden[train_mask, :]
    target_train = dataset.data["labels"][train_mask].cpu().numpy()

    regularization_strengths = 2.0 ** np.arange(-10, 11)

    if dataset.task.is_regression:
        cv = KFold(n_splits=n_splits)
        linear = RidgeCV(alphas=regularization_strengths, cv=cv)
        linear.fit(hidden_train, target_train)
        preds = linear.predict(hidden)
    else:
        cv = StratifiedKFold(n_splits=n_splits)
        linear = LogisticRegressionCV(
            Cs=(1.0 / regularization_strengths),
            cv=cv,
            max_iter=max_iter,
            verbose=verbose,
        )
        linear.fit(hidden_train, target_train)
        preds = linear.predict_proba(hidden)

    return preds


# <<<


# >>> Pretrain Methods


class PretrainCriterionBase(abc.ABC, nn.Module):
    def __init__(self, backbone: nn.Module):
        abc.ABC.__init__(self)
        nn.Module.__init__(self)
        self.backbone = backbone

    @abc.abstractmethod
    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        raise NotImplementedError()


class GRACE(PretrainCriterionBase):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        p_drop_edge_first: float,
        p_drop_edge_second: float,
        p_mask_feat_first: float,
        p_mask_feat_second: float,
        temperature: float,
        d_hidden: int,
        activation: str,
        batch_size: int,
        directed_edge_sampling: bool = True,
    ):
        super().__init__(backbone)

        self.temperature = temperature
        self.batch_size = batch_size

        self.proj_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            get_activation_module(activation),
            nn.Linear(d_hidden, d_hidden),
        )

        self.augmentation_first = ComposeAugmentation(
            [
                DropEdge(
                    p_drop_edge=p_drop_edge_first,
                    directed_edge_sampling=directed_edge_sampling,
                ),
                MaskFeatures(p_mask_feat=p_mask_feat_first),
            ]
        )

        self.augmentation_second = ComposeAugmentation(
            [
                DropEdge(
                    p_drop_edge=p_drop_edge_second,
                    directed_edge_sampling=directed_edge_sampling,
                ),
                MaskFeatures(p_mask_feat=p_mask_feat_second),
            ]
        )

    ### Based on https://arxiv.org/pdf/2205.03169
    def compute_batch_nt_xent_loss(
        self,
        hidden_first: Tensor,
        hidden_second: Tensor,
    ) -> Tensor:
        n_nodes = hidden_first.shape[0]

        hidden = torch.cat([hidden_first, hidden_second], dim=0)
        hidden = normalize_l2_safe(hidden)
        sim = hidden[:n_nodes, :] @ hidden.T
        sim = sim / self.temperature

        aligment_loss = -torch.sum(
            sim[torch.arange(n_nodes), torch.arange(n_nodes) + n_nodes]
        )

        mask = torch.zeros_like(sim, dtype=torch.bool)
        mask[torch.arange(n_nodes), torch.arange(n_nodes)] = True
        sim = torch.where(mask, -torch.inf, sim)
        distribution_loss = torch.sum(torch.logsumexp(sim, dim=1))

        loss = aligment_loss + distribution_loss
        return loss

    def compute_nt_xent_loss(
        self,
        hidden_first: Tensor,
        hidden_second: Tensor,
    ) -> Tensor:
        n_nodes = hidden_first.shape[0]
        permutation = torch.randperm(n_nodes)
        hidden_first = hidden_first[permutation, :]
        hidden_second = hidden_second[permutation, :]

        loss = torch.zeros(1).to(hidden_first.device)

        for batch_start in range(0, n_nodes, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_nodes)
            if batch_end - batch_start < self.batch_size:
                continue
            loss = loss + self.compute_batch_nt_xent_loss(
                hidden_first[batch_start:batch_end, :],
                hidden_second[batch_start:batch_end, :],
            )

        return loss / n_nodes

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        graph_first, x_num_first, x_other_first = self.augmentation_first(
            graph, x_num, x_other
        )
        graph_second, x_num_second, x_other_second = self.augmentation_second(
            graph, x_num, x_other
        )

        _, hidden_first = self.backbone(graph_first, x_num_first, x_other_first)
        _, hidden_second = self.backbone(graph_second, x_num_second, x_other_second)

        hidden_first = self.proj_head(hidden_first)
        hidden_second = self.proj_head(hidden_second)

        loss = 0.5 * (
            self.compute_nt_xent_loss(hidden_first, hidden_second)
            + self.compute_nt_xent_loss(hidden_second, hidden_first)
        ).squeeze(0)

        return loss


class BGRL(PretrainCriterionBase):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        p_drop_edge_first: float,
        p_drop_edge_second: float,
        p_mask_feat_first: float,
        p_mask_feat_second: float,
        ema_decay: int,
        d_hidden: int,
        activation: str,
        n_steps: int,
        device: torch.device | str,
        directed_edge_sampling: bool = True,
    ):
        super().__init__(backbone)

        self.target_backbone = self.get_target_backbone(
            backbone=backbone,
            ema_decay=ema_decay,
            n_steps=n_steps,
            device=device,
        )

        self.predictor = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            get_activation_module(activation),
            nn.Linear(d_hidden, d_hidden),
        )

        self.augmentation_first = ComposeAugmentation(
            [
                DropEdge(
                    p_drop_edge=p_drop_edge_first,
                    directed_edge_sampling=directed_edge_sampling,
                ),
                MaskFeatures(p_mask_feat=p_mask_feat_first),
            ]
        )

        self.augmentation_second = ComposeAugmentation(
            [
                DropEdge(
                    p_drop_edge=p_drop_edge_second,
                    directed_edge_sampling=directed_edge_sampling,
                ),
                MaskFeatures(p_mask_feat=p_mask_feat_second),
            ]
        )

    @staticmethod
    def get_target_backbone(
        backbone: nn.Module,
        ema_decay: float,
        n_steps: int,
        device: torch.device | str,
    ) -> AveragedModel:
        avg_fn = partial(
            ema_multi_avg_fn,
            decay_base=ema_decay,
            n_total_steps=n_steps,
        )

        target_backbone = AveragedModel(
            model=backbone,
            device=torch.device(device),
            avg_fn=avg_fn,  # type: ignore
        )

        for module in target_backbone.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        for param in target_backbone.parameters():
            param.requires_grad = False

        return target_backbone

    def compute_bgrl_loss(
        self,
        graph_online: dgl.DGLGraph,
        x_num_online: Tensor | None,
        x_other_online: Tensor | None,
        graph_target: dgl.DGLGraph,
        x_num_target: Tensor | None,
        x_other_target: Tensor | None,
    ) -> Tensor:
        _, hidden_online = self.backbone(graph_online, x_num_online, x_other_online)
        hidden_online = self.predictor(hidden_online)

        with torch.no_grad():
            _, hidden_target = self.target_backbone(
                graph_target,
                x_num_target,
                x_other_target,
            )
            hidden_target = hidden_target.detach()

        hidden_online = normalize_l2_safe(hidden_online)
        hidden_target = normalize_l2_safe(hidden_target)

        return 1 - (hidden_online * hidden_target).sum(-1).mean()

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        graph_first, x_num_first, x_other_first = self.augmentation_first(
            graph, x_num, x_other
        )
        graph_second, x_num_second, x_other_second = self.augmentation_second(
            graph, x_num, x_other
        )

        loss = 0.5 * (
            self.compute_bgrl_loss(
                graph_first,
                x_num_first,
                x_other_first,
                graph_second,
                x_num_second,
                x_other_second,
            )
            + self.compute_bgrl_loss(
                graph_second,
                x_num_second,
                x_other_second,
                graph_first,
                x_num_first,
                x_other_first,
            )
        )

        self.target_backbone.update_parameters(self.backbone)
        return loss


class MaskGAE(PretrainCriterionBase):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        p_anchor: float,
        n_walks_per_anchor: int,
        walk_len: int,
        coef_degree: float,
        d_hidden: int,
        activation: str,
        directed_edge_sampling: bool = True,
        n_negatives_per_positive: int = 1,
    ):
        super().__init__(backbone)
        self.coef_degree = coef_degree

        self.edge_head = GAEHead(d_hidden, activation)

        self.degree_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            get_activation_module(activation),
            nn.Linear(d_hidden, 1),
        )

        self.edge_sampler = RandomWalkEdgeSampler(
            p_anchor=p_anchor,
            n_walks_per_anchor=n_walks_per_anchor,
            walk_len=walk_len,
            directed=directed_edge_sampling,
        )

        self.negative_sampler = dgl.dataloading.negative_sampler.GlobalUniform(
            n_negatives_per_positive,
        )

    def mask_edges(
        self,
        graph: dgl.DGLGraph,
    ) -> tuple[dgl.DGLGraph, dgl.DGLGraph, Tensor]:
        mask = self.edge_sampler(graph).to(graph.device).to(torch.int32)
        visible_graph = dgl.remove_edges(graph, mask)
        masked_graph = dgl.graph(graph.find_edges(mask), num_nodes=graph.num_nodes())
        return masked_graph, visible_graph, mask

    def compute_edge_loss(
        self,
        full_graph: dgl.DGLGraph,
        hidden: Tensor,
        mask: Tensor,
    ) -> Tensor:
        negatives = self.negative_sampler(full_graph, mask)
        positive_logits = self.edge_head(hidden, full_graph.find_edges(mask))
        negative_logits = self.edge_head(hidden, negatives)

        loss = F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits)
        ) + F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        )

        return loss

    def compute_degree_loss(
        self,
        masked_graph: dgl.DGLGraph,
        hidden: Tensor,
    ) -> Tensor:
        target = 0.5 * (masked_graph.in_degrees() + masked_graph.out_degrees())
        preds = self.degree_head(hidden).squeeze(-1)
        return (target - preds).pow(2).mean()

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        masked_graph, visible_graph, edge_mask = self.mask_edges(graph)

        _, hidden = self.backbone(visible_graph, x_num, x_other)

        edge_loss = self.compute_edge_loss(
            full_graph=graph, hidden=hidden, mask=edge_mask
        )
        degree_loss = self.compute_degree_loss(masked_graph=masked_graph, hidden=hidden)

        return edge_loss + self.coef_degree * degree_loss


class GraphMAE(PretrainCriterionBase):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        p_mask_feat: float,
        p_replace: float,
        scaling_factor: float,
        n_features: int,
        d_hidden: int,
        activation: str,
    ):
        super().__init__(backbone)

        self.decoder = GraphMAEDecoder(
            d=d_hidden,
            d_out=n_features,
            activation=activation,
        )

        self.token = nn.Parameter(torch.zeros(n_features))
        self.dtoken = nn.Parameter(torch.zeros(d_hidden))

        self.p_mask_feat = p_mask_feat
        self.p_replace = p_replace
        self.scaling_factor = scaling_factor

    def mask_features(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
        n_nodes = graph.num_nodes()
        n_token_nodes = int(self.p_mask_feat * (1.0 - self.p_replace) * n_nodes)
        n_replaced_nodes = int(self.p_mask_feat * self.p_replace * n_nodes)

        random_permutation = torch.randperm(n_nodes, device=graph.device)
        token_mask = random_permutation < n_token_nodes
        replace_mask = torch.logical_and(
            random_permutation >= n_token_nodes,
            random_permutation < n_token_nodes + n_replaced_nodes,
        )

        mask_full = torch.logical_or(token_mask, replace_mask)

        x_original = torch.cat([x for x in [x_num, x_other] if x is not None], dim=1)
        x_masked = x_original

        x_masked = torch.where(token_mask[:, None], self.token[None, :], x_masked)
        x_masked = torch.where(
            replace_mask[:, None],
            x_original[torch.randperm(n_nodes), :],
            x_masked,
        )

        x_num_masked = x_masked[:, : x_num.shape[1]] if x_num is not None else None
        x_other_masked = (
            x_masked[:, -x_other.shape[1] :] if x_other is not None else None
        )
        return x_num_masked, x_other_masked, mask_full, x_original

    def compute_feat_loss(
        self,
        graph: dgl.DGLGraph,
        hidden: Tensor,
        mask: Tensor,
        x_original: Tensor,
    ) -> Tensor:
        hidden = torch.where(mask[:, None], self.dtoken[None, :], hidden)
        x_pred = self.decoder(graph, hidden)[mask, :]
        x_original = x_original[mask, :]

        x_pred = normalize_l2_safe(x_pred)
        x_original = normalize_l2_safe(x_original)
        cos_sim = (x_pred * x_original).sum(-1)

        loss = (1 - cos_sim).pow(self.scaling_factor).mean()
        return loss

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        x_num_masked, x_other_masked, mask_full, x_original = self.mask_features(
            graph,
            x_num,
            x_other,
        )
        _, hidden = self.backbone(graph, x_num_masked, x_other_masked)
        loss = self.compute_feat_loss(graph, hidden, mask_full, x_original)

        return loss


class GrASP(PretrainCriterionBase):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        p_corrupt_feat: float,
        p_mask_edge: float,
        coef_degree: float,
        coef_feat: float,
        n_features: int,
        d_hidden: int,
        activation: str,
        directed_edge_sampling: bool = True,
    ):
        super().__init__(backbone=backbone)
        self.coef_feat = float(coef_feat)

        # Structure-based component
        self.edge_sampler = SimpleEdgeSampler(
            p_mask_edge, directed=directed_edge_sampling
        )

        self.edge_head = GAEHead(d_hidden, activation)
        self.degree_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            get_activation_module(activation),
            nn.Linear(d_hidden, 1),
        )

        self.negative_sampler = dgl.dataloading.negative_sampler.GlobalUniform(1)
        self.coef_degree = coef_degree

        # Feature-based component
        self.p_corrupt_feat = p_corrupt_feat
        self.feature_decoder = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            get_activation_module(activation),
            nn.Linear(d_hidden, n_features),
        )

    def mask_features(
        self,
        features: Tensor | None,
    ) -> tuple[Tensor, Tensor] | tuple[None, None]:
        if features is None:
            return None, None
        assert len(features.shape) == 2

        # Get mask
        mask = (
            torch.distributions.bernoulli.Bernoulli(
                torch.tensor(self.p_corrupt_feat, device=features.device)
            )
            .sample(features.shape)
            .bool()
        )
        # Get permutation
        n_nodes = features.shape[0]
        permutation = torch.randint(
            low=0, high=n_nodes, size=features.shape, device=features.device
        )
        # Apply corruption
        features_permuted = torch.gather(input=features, dim=0, index=permutation)
        corrupted_features = torch.where(mask, features_permuted, features)

        return corrupted_features, mask

    def mask_edges(
        self,
        graph: dgl.DGLGraph,
    ) -> tuple[dgl.DGLGraph, dgl.DGLGraph, Tensor]:
        mask = self.edge_sampler(graph).to(graph.device).to(torch.int32)
        visible_graph = dgl.remove_edges(graph, mask)
        masked_graph = dgl.graph(graph.find_edges(mask), num_nodes=graph.num_nodes())
        return masked_graph, visible_graph, mask

    def compute_feat_loss(
        self, hidden: Tensor, x_num: Tensor | None, x_other: Tensor | None
    ) -> Tensor:
        pred = self.feature_decoder(hidden)

        loss = []

        if x_num is not None:
            num_pred = pred[:, : x_num.shape[-1]]
            num_loss = F.mse_loss(input=num_pred, target=x_num, reduction="none")
            loss.append(num_loss)

        if x_other is not None:
            other_pred = pred[:, -x_other.shape[-1] :]
            other_loss = F.binary_cross_entropy_with_logits(
                input=other_pred, target=x_other, reduction="none"
            )
            loss.append(other_loss)

        return torch.cat(loss, dim=1).mean()

    def compute_edge_loss(
        self,
        full_graph: dgl.DGLGraph,
        hidden: Tensor,
        mask: Tensor,
    ) -> Tensor:
        negatives = self.negative_sampler(full_graph, mask)
        positive_logits = self.edge_head(hidden, full_graph.find_edges(mask))
        negative_logits = self.edge_head(hidden, negatives)

        loss = F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits)
        ) + F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        )

        return loss

    def compute_degree_loss(
        self,
        masked_graph: dgl.DGLGraph,
        hidden: Tensor,
    ) -> Tensor:
        target = 0.5 * (masked_graph.in_degrees() + masked_graph.out_degrees())
        preds = self.degree_head(hidden).squeeze(-1)
        return (target - preds).pow(2).mean()

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        masked_graph, visible_graph, edge_mask = self.mask_edges(graph)
        x_num_corrupted, _ = self.mask_features(x_num)
        x_other_corrupted, _ = self.mask_features(x_other)

        _, hidden = self.backbone(visible_graph, x_num_corrupted, x_other_corrupted)

        edge_loss = self.compute_edge_loss(
            full_graph=graph, hidden=hidden, mask=edge_mask
        )
        degree_loss = self.compute_degree_loss(masked_graph=masked_graph, hidden=hidden)
        feat_loss = self.compute_feat_loss(hidden, x_num, x_other)

        return edge_loss + self.coef_degree * degree_loss + self.coef_feat * feat_loss


class GrASPNaive(PretrainCriterionBase):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feat_config: KWArgs,
        edge_config: KWArgs,
        #
        coef_degree: float,
        coef_feat: float,
        n_features: int,
        d_hidden: int,
        activation: str,
        directed_edge_sampling: bool = True,
    ):
        super().__init__(backbone=backbone)
        self.coef_feat = float(coef_feat)
        self.feat_type = feat_config["type"]

        # Structure-based component
        edge_sampler_cls = {
            "edge": SimpleEdgeSampler,
            "path": RandomWalkEdgeSampler,
        }[edge_config["sampler"]]
        self.edge_sampler = edge_sampler_cls(
            **edge_config["sampler_params"],
            directed=directed_edge_sampling,
        )

        self.edge_head = GAEHead(d_hidden, activation)
        self.degree_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            get_activation_module(activation),
            nn.Linear(d_hidden, 1),
        )

        self.negative_sampler = dgl.dataloading.negative_sampler.GlobalUniform(1)
        self.coef_degree = coef_degree

        # Feature-based component
        if feat_config["type"] == "grasp":
            self.p_corrupt_feat = feat_config["params"]["p_corrupt_feat"]
            self.feature_decoder = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                get_activation_module(activation),
                nn.Linear(d_hidden, n_features),
            )
        elif feat_config["type"] == "graph_mae":
            self.decoder = GraphMAEDecoder(
                d=d_hidden,
                d_out=n_features,
                activation=activation,
            )

            self.token = nn.Parameter(torch.zeros(n_features))
            self.dtoken = nn.Parameter(torch.zeros(d_hidden))

            self.p_mask_feat = feat_config["params"]["p_mask_feat"]
            self.p_replace = feat_config["params"]["p_replace"]
            self.scaling_factor = feat_config["params"]["scaling_factor"]
        else:
            raise ValueError(f"Unknown feat-module type: {feat_config['type']}")

    def mask_edges(
        self,
        graph: dgl.DGLGraph,
    ) -> tuple[dgl.DGLGraph, dgl.DGLGraph, Tensor]:
        mask = self.edge_sampler(graph).to(graph.device).to(torch.int32)
        visible_graph = dgl.remove_edges(graph, mask)
        masked_graph = dgl.graph(graph.find_edges(mask), num_nodes=graph.num_nodes())
        return masked_graph, visible_graph, mask

    def mask_features_grasp(
        self,
        features: Tensor | None,
    ) -> tuple[Tensor, Tensor] | tuple[None, None]:
        if features is None:
            return None, None
        assert len(features.shape) == 2

        # Get mask
        mask = (
            torch.distributions.bernoulli.Bernoulli(
                torch.tensor(self.p_corrupt_feat, device=features.device)
            )
            .sample(features.shape)
            .bool()
        )
        # Get permutation
        n_nodes = features.shape[0]
        permutation = torch.randint(
            low=0, high=n_nodes, size=features.shape, device=features.device
        )
        # Apply corruption
        features_permuted = torch.gather(input=features, dim=0, index=permutation)
        corrupted_features = torch.where(mask, features_permuted, features)

        return corrupted_features, mask

    def mask_features_graphmae(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
        n_nodes = graph.num_nodes()
        n_token_nodes = int(self.p_mask_feat * (1.0 - self.p_replace) * n_nodes)
        n_replaced_nodes = int(self.p_mask_feat * self.p_replace * n_nodes)

        random_permutation = torch.randperm(n_nodes, device=graph.device)
        token_mask = random_permutation < n_token_nodes
        replace_mask = torch.logical_and(
            random_permutation >= n_token_nodes,
            random_permutation < n_token_nodes + n_replaced_nodes,
        )

        mask_full = torch.logical_or(token_mask, replace_mask)

        x_original = torch.cat([x for x in [x_num, x_other] if x is not None], dim=1)
        x_masked = x_original

        x_masked = torch.where(token_mask[:, None], self.token[None, :], x_masked)
        x_masked = torch.where(
            replace_mask[:, None],
            x_original[torch.randperm(n_nodes), :],
            x_masked,
        )

        x_num_masked = x_masked[:, : x_num.shape[1]] if x_num is not None else None
        x_other_masked = (
            x_masked[:, -x_other.shape[1] :] if x_other is not None else None
        )
        return x_num_masked, x_other_masked, mask_full, x_original

    def mask_features(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if self.feat_type == "grasp":
            x_num_corrupted, _ = self.mask_features_grasp(x_num)
            x_other_corrupted, _ = self.mask_features_grasp(x_other)
            return x_num_corrupted, x_other_corrupted, None, None
        elif self.feat_type == "graph_mae":
            return self.mask_features_graphmae(graph, x_num, x_other)
        else:
            raise ValueError(f"Unknown feat-module type: {self.feat_type}")

    def compute_feat_loss_grasp(
        self,
        hidden: Tensor,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        pred = self.feature_decoder(hidden)

        loss = []

        if x_num is not None:
            num_pred = pred[:, : x_num.shape[-1]]
            num_loss = F.mse_loss(input=num_pred, target=x_num, reduction="none")
            loss.append(num_loss)

        if x_other is not None:
            other_pred = pred[:, -x_other.shape[-1] :]
            other_loss = F.binary_cross_entropy_with_logits(
                input=other_pred, target=x_other, reduction="none"
            )
            loss.append(other_loss)

        return torch.cat(loss, dim=1).mean()

    def compute_feat_loss_graphmae(
        self,
        graph: dgl.DGLGraph,
        hidden: Tensor,
        mask: Tensor,
        x_original: Tensor,
    ) -> Tensor:
        hidden = torch.where(mask[:, None], self.dtoken[None, :], hidden)
        x_pred = self.decoder(graph, hidden)[mask, :]
        x_original = x_original[mask, :]

        x_pred = normalize_l2_safe(x_pred)
        x_original = normalize_l2_safe(x_original)
        cos_sim = (x_pred * x_original).sum(-1)

        loss = (1 - cos_sim).pow(self.scaling_factor).mean()
        return loss

    def compute_feat_loss(
        self,
        graph: dgl.DGLGraph,
        hidden: Tensor,
        mask: Tensor | None,
        x_original: Tensor | None,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        if self.feat_type == "grasp":
            assert mask is None
            assert x_original is None
            return self.compute_feat_loss_grasp(hidden, x_num, x_other)
        elif self.feat_type == "graph_mae":
            assert mask is not None
            assert x_original is not None
            return self.compute_feat_loss_graphmae(graph, hidden, mask, x_original)
        else:
            raise ValueError(f"Unknown feat-module type: {self.feat_type}")

    def compute_edge_loss(
        self,
        full_graph: dgl.DGLGraph,
        hidden: Tensor,
        mask: Tensor,
    ) -> Tensor:
        negatives = self.negative_sampler(full_graph, mask)
        positive_logits = self.edge_head(hidden, full_graph.find_edges(mask))
        negative_logits = self.edge_head(hidden, negatives)

        loss = F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits)
        ) + F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        )

        return loss

    def compute_degree_loss(
        self,
        masked_graph: dgl.DGLGraph,
        hidden: Tensor,
    ) -> Tensor:
        target = 0.5 * (masked_graph.in_degrees() + masked_graph.out_degrees())
        preds = self.degree_head(hidden).squeeze(-1)
        return (target - preds).pow(2).mean()

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        masked_graph, visible_graph, edge_mask = self.mask_edges(graph)

        x_num_corrupted, x_other_corrupted, mask_full, x_original = self.mask_features(
            graph,
            x_num,
            x_other,
        )

        _, hidden = self.backbone(visible_graph, x_num_corrupted, x_other_corrupted)

        edge_loss = self.compute_edge_loss(
            full_graph=graph, hidden=hidden, mask=edge_mask
        )
        degree_loss = self.compute_degree_loss(masked_graph=masked_graph, hidden=hidden)
        feat_loss = self.compute_feat_loss(
            graph, hidden, mask_full, x_original, x_num, x_other
        )

        return edge_loss + self.coef_degree * degree_loss + self.coef_feat * feat_loss


class EPAGCL(PretrainCriterionBase):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        p_drop_edge_first: float,
        p_drop_edge_second: float,
        p_add_edge_second: float,
        p_mask_feat_first: float,
        p_mask_feat_second: float,
        temperature: float,
        d_hidden: int,
        activation: str,
        batch_size: int,
        p_drop_cutoff: float,
        p_add_cutoff: float,
        candidate_edge_multiplier: float = 2.0,
        directed_edge_sampling: bool = False,
        eps: float = 1e-8,
        max_candidate_edges: int | None = None,
        cache_preprocessing: bool = True,
        exclude_self_loop_candidates: bool = True,
    ):
        super().__init__(backbone)

        _prob_params = {
            "p_drop_edge_first": p_drop_edge_first,
            "p_drop_edge_second": p_drop_edge_second,
            "p_add_edge_second": p_add_edge_second,
            "p_mask_feat_first": p_mask_feat_first,
            "p_mask_feat_second": p_mask_feat_second,
            "p_drop_cutoff": p_drop_cutoff,
            "p_add_cutoff": p_add_cutoff,
        }
        for name, val in _prob_params.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if candidate_edge_multiplier < 0.0:
            raise ValueError(
                "candidate_edge_multiplier must be >= 0, "
                f"got {candidate_edge_multiplier}"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.p_drop_edge_first = p_drop_edge_first
        self.p_drop_edge_second = p_drop_edge_second
        self.p_add_edge_second = p_add_edge_second
        self.p_mask_feat_first = p_mask_feat_first
        self.p_mask_feat_second = p_mask_feat_second
        self.temperature = temperature
        self.batch_size = batch_size
        self.p_drop_cutoff = p_drop_cutoff
        self.p_add_cutoff = p_add_cutoff
        self.candidate_edge_multiplier = candidate_edge_multiplier
        self.eps = eps
        self.max_candidate_edges = max_candidate_edges
        self.cache_preprocessing = cache_preprocessing
        self.exclude_self_loop_candidates = exclude_self_loop_candidates

        self.proj_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            get_activation_module(activation),
            nn.Linear(d_hidden, d_hidden),
        )

        self._cache: dict | None = None

    def reset_cache(self) -> None:
        self._cache = None

    def _validate_graph(self, graph: dgl.DGLGraph) -> None:
        if not graph.is_homogeneous:
            raise ValueError(
                "EPAGCL requires a homogeneous graph, "
                "but a heterogeneous graph was provided."
            )

        if len(graph.edata) > 0:
            raise ValueError(
                "EPAGCL does not support graphs with edge features, because candidate "
                "non-edges cannot be initialized with edge data. "
                f"Found edge data keys: {list(graph.edata.keys())}."
            )

        src, dst = graph.edges()

        edge_pairs = torch.stack([src, dst], dim=1)
        n_unique = torch.unique(edge_pairs, dim=0).shape[0]
        if n_unique < edge_pairs.shape[0]:
            raise ValueError(
                "EPAGCL expects a simple bidirected graph, "
                "but duplicate edges were found."
            )

        non_self_mask = src != dst
        src_ns, dst_ns = src[non_self_mask], dst[non_self_mask]

        if src_ns.numel() > 0:
            canonical = torch.stack(
                [torch.minimum(src_ns, dst_ns), torch.maximum(src_ns, dst_ns)], dim=1
            )
            n_canonical = torch.unique(canonical, dim=0).shape[0]
            if 2 * n_canonical != src_ns.shape[0]:
                raise ValueError(
                    "EPAGCL expects a simple bidirected graph, "
                    "but non-self edges are not properly bidirected. "
                    "Every non-self edge (u, v) must have a reverse (v, u)."
                )

    def _compute_effective_degrees(
        self, graph: dgl.DGLGraph, src: Tensor, self_loop_mask: Tensor
    ) -> Tensor:
        n_nodes = graph.num_nodes()
        device = graph.device
        self_loop_src = src[self_loop_mask]
        self_loop_count = torch.zeros(n_nodes, device=device)
        if self_loop_src.numel() > 0:
            self_loop_count.scatter_add_(
                0,
                self_loop_src.to(torch.int64),
                torch.ones(self_loop_src.numel(), device=device),
            )
        non_self_degree = graph.out_degrees().float() - self_loop_count
        return non_self_degree + 1.0

    def _build_canonical_edges(
        self, graph: dgl.DGLGraph, src: Tensor, dst: Tensor, non_self_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (canonical_uv [M,2], fwd_eids [M], bwd_eids [M])."""
        n_nodes = graph.num_nodes()
        src_ns = src[non_self_mask]
        dst_ns = dst[non_self_mask]
        eids_ns = torch.where(non_self_mask)[0]

        fwd_mask = src_ns < dst_ns
        fwd_order = torch.argsort(src_ns[fwd_mask] * n_nodes + dst_ns[fwd_mask])
        canonical_u = src_ns[fwd_mask][fwd_order]
        canonical_v = dst_ns[fwd_mask][fwd_order]
        fwd_eids = eids_ns[fwd_mask][fwd_order]

        bwd_mask = ~fwd_mask
        bwd_order = torch.argsort(dst_ns[bwd_mask] * n_nodes + src_ns[bwd_mask])
        bwd_eids = eids_ns[bwd_mask][bwd_order]

        return torch.stack([canonical_u, canonical_v], dim=1), fwd_eids, bwd_eids

    def _weight_to_drop_probs(self, weights: Tensor, p_drop: float) -> Tensor:
        w_min = weights.min()
        denom = weights.mean() - w_min
        if denom <= self.eps:
            return torch.full_like(weights, p_drop).clamp(0.0, self.p_drop_cutoff)
        return ((weights - w_min) / denom * p_drop).clamp(0.0, self.p_drop_cutoff)

    def _weight_to_add_probs(self, weights: Tensor) -> Tensor:
        w_max = weights.max()
        denom = w_max - weights.mean()
        if denom <= self.eps:
            return torch.full_like(weights, self.p_add_edge_second).clamp(
                0.0, self.p_add_cutoff
            )
        return ((w_max - weights) / denom * self.p_add_edge_second).clamp(
            0.0, self.p_add_cutoff
        )

    def _build_candidate_edges(
        self,
        n_nodes: int,
        m: int,
        canonical_uv: Tensor,
        degrees: Tensor,
        device: torch.device,
    ) -> Tensor:
        if m == 0 or n_nodes < 2:
            return torch.empty((0, 2), dtype=torch.int32, device=device)

        k = min(math.ceil(math.sqrt(self.candidate_edge_multiplier * m)), n_nodes)

        # Top-k by descending degree; ties broken by ascending node ID via stable sort
        degree_order = torch.argsort(degrees, descending=True, stable=True)
        top_k = degree_order[:k].sort().values

        i_idx, j_idx = torch.triu_indices(k, k, offset=1, device=device)
        u_cand = top_k[i_idx]
        v_cand = top_k[j_idx]

        # Remove pairs that are already existing edges
        existing_keys = (
            (canonical_uv[:, 0] * n_nodes + canonical_uv[:, 1]).sort().values
        )
        candidate_keys = u_cand * n_nodes + v_cand
        pos = torch.searchsorted(existing_keys, candidate_keys).clamp(
            max=existing_keys.shape[0] - 1
        )
        non_existing = existing_keys[pos] != candidate_keys
        candidate_uv = torch.stack([u_cand[non_existing], v_cand[non_existing]], dim=1)

        if (
            self.max_candidate_edges is not None
            and candidate_uv.shape[0] > self.max_candidate_edges
        ):
            candidate_uv = candidate_uv[: self.max_candidate_edges]

        return candidate_uv

    def _preprocess(self, graph: dgl.DGLGraph) -> dict:
        self._validate_graph(graph)
        src, dst = graph.edges()

        self_loop_mask = src == dst
        non_self_mask = ~self_loop_mask

        canonical_uv, fwd_eids, bwd_eids = self._build_canonical_edges(
            graph, src, dst, non_self_mask
        )
        m = canonical_uv.shape[0]
        degrees = self._compute_effective_degrees(graph, src, self_loop_mask)

        if m > 0:
            d_u = degrees[canonical_uv[:, 0]]
            d_v = degrees[canonical_uv[:, 1]]
            drop_weights = 2.0 / torch.sqrt(d_u * d_v)
            drop_probs_v1 = self._weight_to_drop_probs(
                drop_weights, self.p_drop_edge_first
            )
            drop_probs_v2 = self._weight_to_drop_probs(
                drop_weights, self.p_drop_edge_second
            )
        else:
            drop_probs_v1 = torch.empty(0, device=graph.device)
            drop_probs_v2 = torch.empty(0, device=graph.device)

        candidate_uv = self._build_candidate_edges(
            graph.num_nodes(), m, canonical_uv, degrees, graph.device
        )

        if candidate_uv.shape[0] > 0:
            d_cu = degrees[candidate_uv[:, 0]]
            d_cv = degrees[candidate_uv[:, 1]]
            add_weights = 2.0 / torch.sqrt((d_cu + 1.0) * (d_cv + 1.0))
            add_probs = self._weight_to_add_probs(add_weights)
        else:
            add_probs = torch.empty(0, device=graph.device)

        return {
            "fwd_eids": fwd_eids,
            "bwd_eids": bwd_eids,
            "drop_probs_v1": drop_probs_v1,
            "drop_probs_v2": drop_probs_v2,
            "candidate_uv": candidate_uv,
            "add_probs": add_probs,
        }

    def _drop_edges(
        self,
        graph: dgl.DGLGraph,
        fwd_eids: Tensor,
        bwd_eids: Tensor,
        drop_probs: Tensor,
    ) -> dgl.DGLGraph:
        if drop_probs.numel() == 0:
            return graph
        drop_mask = torch.bernoulli(drop_probs).bool()
        if not drop_mask.any():
            return graph
        dropped = torch.cat([fwd_eids[drop_mask], bwd_eids[drop_mask]]).to(torch.int32)
        return dgl.remove_edges(graph, dropped)

    def _add_edges(
        self,
        graph: dgl.DGLGraph,
        candidate_uv: Tensor,
        add_probs: Tensor,
    ) -> dgl.DGLGraph:
        if add_probs.numel() == 0:
            return graph
        add_mask = torch.bernoulli(add_probs).bool()
        if not add_mask.any():
            return graph
        added = candidate_uv[add_mask]
        u, v = added[:, 0], added[:, 1]
        new_src = torch.cat([u, v]).to(graph.idtype)
        new_dst = torch.cat([v, u]).to(graph.idtype)
        return dgl.add_edges(graph, new_src, new_dst)

    def _mask_feat(self, x: Tensor | None, p: float) -> Tensor | None:
        if x is None:
            return None
        mask = torch.rand(x.shape[-1], device=x.device) < p
        return torch.where(mask.unsqueeze(0), 0.0, x)

    def _batch_nt_xent(self, h_a: Tensor, h_b: Tensor) -> Tensor:
        n = h_a.shape[0]
        h_cat = normalize_l2_safe(torch.cat([h_a, h_b], dim=0))
        sim = h_cat[:n] @ h_cat.T / self.temperature

        idx = torch.arange(n, device=h_a.device)
        alignment_loss = -sim[idx, idx + n].sum()

        mask = torch.zeros(n, 2 * n, dtype=torch.bool, device=h_a.device)
        mask[idx, idx] = True
        distribution_loss = torch.logsumexp(
            sim.masked_fill(mask, float("-inf")), dim=1
        ).sum()

        return alignment_loss + distribution_loss

    def _directional_nt_xent(self, h_anchor: Tensor, h_positive: Tensor) -> Tensor:
        n_nodes = h_anchor.shape[0]
        if n_nodes <= 1:
            return torch.zeros(1, device=h_anchor.device).squeeze(0)

        bs = (
            n_nodes
            if (self.batch_size is None or self.batch_size <= 0)
            else self.batch_size
        )
        perm = torch.randperm(n_nodes, device=h_anchor.device)
        h_a = h_anchor[perm]
        h_p = h_positive[perm]

        total_loss = torch.zeros(1, device=h_anchor.device)
        total_count = 0
        for start in range(0, n_nodes, bs):
            end = min(start + bs, n_nodes)
            if end - start < 2:
                continue
            total_loss = total_loss + self._batch_nt_xent(
                h_a[start:end], h_p[start:end]
            )
            total_count += end - start

        if total_count == 0:
            return torch.zeros(1, device=h_anchor.device).squeeze(0)
        return (total_loss / total_count).squeeze(0)

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        if self.cache_preprocessing:
            if self._cache is None:
                self._cache = self._preprocess(graph)
            data = self._cache
        else:
            data = self._preprocess(graph)

        graph_v1 = self._drop_edges(
            graph, data["fwd_eids"], data["bwd_eids"], data["drop_probs_v1"]
        )
        graph_v2 = self._drop_edges(
            graph, data["fwd_eids"], data["bwd_eids"], data["drop_probs_v2"]
        )
        graph_v2 = self._add_edges(graph_v2, data["candidate_uv"], data["add_probs"])

        _, h1 = self.backbone(
            graph_v1,
            self._mask_feat(x_num, self.p_mask_feat_first),
            self._mask_feat(x_other, self.p_mask_feat_first),
        )
        _, h2 = self.backbone(
            graph_v2,
            self._mask_feat(x_num, self.p_mask_feat_second),
            self._mask_feat(x_other, self.p_mask_feat_second),
        )

        z1 = self.proj_head(h1)
        z2 = self.proj_head(h2)

        return 0.5 * (
            self._directional_nt_xent(z1, z2) + self._directional_nt_xent(z2, z1)
        )


class GraphMAE2(PretrainCriterionBase):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        p_mask_feat: float,
        p_remask: float,
        n_remask: int,
        scaling_factor: float,
        coef_latent: float,
        ema_decay: float,
        n_features: int,
        d_hidden: int,
        activation: str,
        device: torch.device | str,
        reconstruct_all_nodes: bool = False,
        latent_all_nodes: bool = False,
    ):
        super().__init__(backbone)

        if not 0.0 < p_mask_feat <= 1.0:
            raise ValueError(f"p_mask_feat must be in (0, 1], got {p_mask_feat}.")
        if not 0.0 <= p_remask <= 1.0:
            raise ValueError(f"p_remask must be in [0, 1], got {p_remask}.")
        if n_remask < 1:
            raise ValueError(f"n_remask must be >= 1, got {n_remask}.")
        if scaling_factor <= 0.0:
            raise ValueError(f"scaling_factor must be positive, got {scaling_factor}.")
        if coef_latent < 0.0:
            raise ValueError(f"coef_latent must be nonnegative, got {coef_latent}.")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {ema_decay}.")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}.")
        if d_hidden <= 0:
            raise ValueError(f"d_hidden must be positive, got {d_hidden}.")

        self.p_mask_feat = p_mask_feat
        self.p_remask = p_remask
        self.n_remask = n_remask
        self.scaling_factor = scaling_factor
        self.coef_latent = coef_latent
        self.ema_decay = ema_decay
        self.n_features = n_features
        self.reconstruct_all_nodes = reconstruct_all_nodes
        self.latent_all_nodes = latent_all_nodes

        self.input_mask_token = nn.Parameter(torch.zeros(n_features))
        self.decoder_mask_token = nn.Parameter(torch.zeros(d_hidden))

        self.feature_decoder = GraphMAEDecoder(
            d=d_hidden,
            d_out=n_features,
            activation=activation,
        )

        self.online_projector = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            get_activation_module(activation),
            nn.Linear(d_hidden, d_hidden),
        )

        self.target_backbone = copy.deepcopy(self.backbone).to(device)
        self.target_projector = copy.deepcopy(self.online_projector).to(device)

        self._freeze_targets()

    def _freeze_targets(self) -> None:
        self.target_backbone.requires_grad_(False)
        self.target_projector.requires_grad_(False)
        self.target_backbone.eval()
        self.target_projector.eval()

    def train(self, mode: bool = True) -> "GraphMAE2":
        super().train(mode)

        # The target generator must not enter training mode even when the
        # surrounding criterion does.
        self.target_backbone.eval()
        self.target_projector.eval()
        return self

    @staticmethod
    def _named_tensors(
        module: nn.Module,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return dict(module.named_parameters()), dict(module.named_buffers())

    @torch.no_grad()
    def update_target(self) -> None:
        decay = self.ema_decay

        for online, target in (
            (self.backbone, self.target_backbone),
            (self.online_projector, self.target_projector),
        ):
            online_params, online_buffers = self._named_tensors(online)
            target_params, target_buffers = self._named_tensors(target)

            if online_params.keys() != target_params.keys():
                raise RuntimeError("Online and target parameter structures differ.")
            if online_buffers.keys() != target_buffers.keys():
                raise RuntimeError("Online and target buffer structures differ.")

            for name, target_param in target_params.items():
                online_param = online_params[name]
                if online_param.shape != target_param.shape:
                    raise RuntimeError(
                        f"Parameter shape mismatch for {name}: "
                        f"{tuple(online_param.shape)} vs "
                        f"{tuple(target_param.shape)}."
                    )

                target_param.lerp_(
                    online_param.detach(),
                    1.0 - decay,
                )

            # Floating buffers such as normalization statistics are also
            # smoothed. Integer counters cannot be EMA-updated.
            for name, target_buffer in target_buffers.items():
                online_buffer = online_buffers[name].detach()

                if target_buffer.shape != online_buffer.shape:
                    raise RuntimeError(
                        f"Buffer shape mismatch for {name}: "
                        f"{tuple(online_buffer.shape)} vs "
                        f"{tuple(target_buffer.shape)}."
                    )

                if torch.is_floating_point(target_buffer):
                    target_buffer.lerp_(online_buffer, 1.0 - decay)
                else:
                    target_buffer.copy_(online_buffer)

        self.target_backbone.eval()
        self.target_projector.eval()

    @staticmethod
    def _sample_mask(
        n_nodes: int,
        p: float,
        device: torch.device,
        *,
        require_nonempty: bool = False,
    ) -> Tensor:
        if n_nodes < 0:
            raise ValueError(f"n_nodes must be nonnegative, got {n_nodes}.")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Mask probability must be in [0, 1], got {p}.")

        if n_nodes == 0 or p == 0.0:
            if require_nonempty:
                raise ValueError(
                    "Cannot create a nonempty mask with zero nodes or p=0."
                )
            return torch.empty(0, dtype=torch.long, device=device)

        n_masked = int(p * n_nodes)
        if require_nonempty:
            n_masked = max(1, n_masked)

        n_masked = min(n_nodes, n_masked)

        if n_masked == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        return torch.randperm(n_nodes, device=device)[:n_masked]

    @staticmethod
    def _split_features(
        X: Tensor,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if x_num is None:
            return None, X
        if x_other is None:
            return X, None

        n_num_features = x_num.shape[1]
        return X[:, :n_num_features], X[:, n_num_features:]

    def _scaled_cosine_error(
        self,
        prediction: Tensor,
        target: Tensor,
        scaling_factor: float | None = None,
    ) -> Tensor:
        if scaling_factor is None:
            scaling_factor = self.scaling_factor
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target must have identical shapes, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.shape[0] == 0:
            raise ValueError("Scaled cosine error received an empty node set.")

        prediction = normalize_l2_safe(prediction)
        target = normalize_l2_safe(target)

        cosine = (prediction * target).sum(dim=-1).clamp(-1.0, 1.0)
        return (1.0 - cosine).clamp_min(0.0).pow(scaling_factor).mean()

    def __call__(
        self,
        graph: dgl.DGLGraph,
        x_num: Tensor | None,
        x_other: Tensor | None,
    ) -> Tensor:
        parts = [x for x in (x_num, x_other) if x is not None]
        if not parts:
            raise ValueError("At least one of x_num or x_other must be provided.")

        n_nodes = parts[0].shape[0]
        if n_nodes == 0:
            raise ValueError("GraphMAE2 does not support an empty node batch.")

        for part in parts:
            if part.ndim != 2:
                raise ValueError(
                    f"Every feature tensor must be rank 2, got {part.ndim}."
                )
            if part.shape[0] != n_nodes:
                raise ValueError(
                    "x_num and x_other must have the same number of nodes."
                )
            if part.device != parts[0].device:
                raise ValueError("x_num and x_other must be on the same device.")

        X = torch.cat(parts, dim=1)

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} input features, got {X.shape[1]}."
            )

        if graph.num_nodes() != n_nodes:
            raise ValueError(
                f"Graph has {graph.num_nodes()} nodes but features have {n_nodes} rows."
            )

        device = X.device

        # 1. Encode a masked input view.
        masked_nodes = self._sample_mask(
            n_nodes,
            self.p_mask_feat,
            device,
            require_nonempty=True,
        )

        X_masked = X.clone()
        X_masked[masked_nodes] = self.input_mask_token.to(
            dtype=X.dtype,
            device=X.device,
        )

        x_num_masked, x_other_masked = self._split_features(
            X_masked,
            x_num,
            x_other,
        )
        _, H = self.backbone(graph, x_num_masked, x_other_masked)

        # 2. Multi-view random re-mask reconstruction.
        reconstruction_losses: list[Tensor] = []

        for _ in range(self.n_remask):
            remasked_nodes = self._sample_mask(
                n_nodes,
                self.p_remask,
                device,
                require_nonempty=False,
            )

            H_view = H.clone()
            if remasked_nodes.numel() > 0:
                H_view[remasked_nodes] = self.decoder_mask_token.to(
                    dtype=H.dtype,
                    device=H.device,
                )

            X_hat = self.feature_decoder(graph, H_view)

            if self.reconstruct_all_nodes:
                reconstruction_losses.append(self._scaled_cosine_error(X_hat, X))
            else:
                reconstruction_losses.append(
                    self._scaled_cosine_error(
                        X_hat[masked_nodes],
                        X[masked_nodes],
                    )
                )

        # Average so n_remask does not implicitly rescale the objective.
        input_loss = torch.stack(reconstruction_losses).mean()

        # 3. Latent representation prediction.
        Z_online = self.online_projector(H)

        with torch.no_grad():
            _, H_target = self.target_backbone(graph, x_num, x_other)
            Z_target = self.target_projector(H_target)

        if self.latent_all_nodes:
            latent_loss = self._scaled_cosine_error(
                Z_online,
                Z_target,
                scaling_factor=1.0,
            )
        else:
            latent_loss = self._scaled_cosine_error(
                Z_online[masked_nodes],
                Z_target[masked_nodes],
                scaling_factor=1.0,
            )

        if self.training:
            self.update_target()

        return input_loss + self.coef_latent * latent_loss


# <<<


def build_criterion(
    backbone: nn.Module,
    method: PretrainMethod | str,
    params: KWArgs,
) -> PretrainCriterionBase:
    method = PretrainMethod(method)
    cls = {
        PretrainMethod.GRACE: GRACE,
        PretrainMethod.BGRL: BGRL,
        PretrainMethod.MASK_GAE: MaskGAE,
        PretrainMethod.GRAPH_MAE: GraphMAE,
        PretrainMethod.GRAPH_MAE2: GraphMAE2,
        PretrainMethod.GRASP: GrASP,
        PretrainMethod.GRASP_NAIVE: GrASPNaive,
        PretrainMethod.EPAGCL: EPAGCL,
    }[method]
    return cls(backbone, **params)
