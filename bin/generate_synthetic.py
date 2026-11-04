#!/usr/bin/env python3
"""Generate synthetic datasets that share graph structure with real datasets."""

import argparse
from pathlib import Path

import torch
import yaml
from loguru import logger

from lib.graph.data import build_dataset
from lib.graph.synthetic import (
    generate_synthetic_features,
    generate_synthetic_splits,
    generate_synthetic_targets,
)

DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "lastfm-asia",
    "facebook",
    "amazon-photo",
    "amazon-computers",
    "amazon-ratings",
    "tolokers-2",
    "artnet-exp",
    "city-roads-M",
    "artnet-views",
]


def generate_synthetic(
    dataset: str,
    n_datasets: int,
    n_features: int,
    feat_coef: float,
    struc_coef: float,
    nfa_coef: float,
    noise_coef: float,
    train_ratio: float,
    val_ratio: float,
    data_dir: Path,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # >>> Load real dataset (only need graph structure)
    dataset_path = data_dir / dataset
    logger.info(f"Loading graph from {dataset_path}")
    ds = build_dataset(dataset_path, setting="transductive", internal_split_name="RL")
    graph = ds.data["graph"].to(device)
    n_nodes = graph.num_nodes()
    logger.info(f"Graph: {n_nodes} nodes, {graph.num_edges()} edges")

    # >>> Create output directory
    output_dir = dataset_path / "synthetic"
    output_dir.mkdir(exist_ok=True)

    # >>> Generate synthetic datasets
    for seed in range(n_datasets):
        logger.info(f"Generating synthetic dataset {seed:02d}/{n_datasets}")

        features = generate_synthetic_features(graph, n_features, seed=seed)
        targets = generate_synthetic_targets(
            graph,
            features,
            feat_coef=feat_coef,
            struc_coef=struc_coef,
            nfa_coef=nfa_coef,
            noise_coef=noise_coef,
            seed=seed,
        )
        splits = generate_synthetic_splits(
            n_nodes,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        # Save to .pt file
        output_path = output_dir / f"synt-{seed:02d}.pt"
        torch.save(
            {
                "features": features,
                "targets": targets,
                "splits": splits,
            },
            output_path,
        )

    # >>> Save generation config
    config = {
        "dataset": dataset,
        "n_datasets": n_datasets,
        "n_features": n_features,
        "target_coefficients": {
            "feat_coef": feat_coef,
            "struc_coef": struc_coef,
            "nfa_coef": nfa_coef,
            "noise_coef": noise_coef,
        },
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": 1.0 - train_ratio - val_ratio,
        },
        "graph_info": {
            "n_nodes": n_nodes,
            "n_edges": graph.num_edges(),
        },
    }
    config_path = output_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Saved {n_datasets} synthetic datasets to {output_dir}")
    logger.info(f"Config saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets from real graph structures"
    )
    parser.add_argument("--n-datasets", type=int, default=10)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--feat-coef", type=float, default=1.0)
    parser.add_argument("--struc-coef", type=float, default=1.0)
    parser.add_argument("--nfa-coef", type=float, default=1.0)
    parser.add_argument("--noise-coef", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    for dataset in DATASETS:
        generate_synthetic(dataset=dataset, **vars(args))


if __name__ == "__main__":
    main()
