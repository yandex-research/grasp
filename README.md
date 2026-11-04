# Benchmarking Graph Self-Supervised Learning for Node-Level Tasks: Insights and Strong Baseline

This is the official repository for the paper "Benchmarking Graph Self-Supervised Learning for Node-Level Tasks: Insights and Strong Baseline" (NeurIPS 2026 E&D Track). It provides code to reproduce our experiments and an implementation of our proposed GrASP method.

## Reproducing Experiments

**Prerequisites**

1. [Install uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
2. Install dependencies
```
uv sync
```
3. For experiments on [GraphLand](https://github.com/yandex-research/graphland), download datasets and place them in the `data` directory

**Running the evaluation**

You can execute a minimal evaluation run (supervised GCN on `tolokers-2` dataset) with the following command:

```
uv run bin/go.py exp/main/sup/finetune/gcn/tolokers-2/tuning.toml --force
```

## Project Structure

- `bin/` - Training and evaluation scripts
- `exp/` - Experiment configurations and results
- `data/` - Dataset directory
- `lib/` - Common utilities and tools

## Configuration

Experiments are configured using TOML files located in the `exp/` directory. Each configuration specifies:
- Dataset path and preprocessing
- Model hyperparameters
- Training settings
- Evaluation metrics

## Results

Evaluation results are saved in the same directory as the configuration file:
- `report.json` - Evaluation metrics
- Model checkpoints
- Training logs

## Reference Implementations

Reference repositories for the graph self-supervised learning methods implemented here:

- [GRACE](https://github.com/CRIPAC-DIG/GRACE)
- [BGRL](https://github.com/nerdslab/bgrl)
- [MaskGAE](https://github.com/EdisonLeeeee/MaskGAE)
- [GraphMAE](https://github.com/THUDM/GraphMAE)
- [EPAGCL](https://github.com/coinrainy/EPAGCL)
- [GraphMAE2](https://github.com/THUDM/GraphMAE2)

