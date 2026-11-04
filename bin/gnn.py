import copy
import datetime
import math
import os
import warnings
from collections import namedtuple
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import delu
import dgl
import numpy as np
import scipy
import scipy.special
import torch
import torch.nn as nn
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

import lib
import lib.data
import lib.deep
import lib.env
import lib.graph.data
import lib.graph.deep
import lib.graph.util
from lib import JSONDict, KWArgs, PartKey
from lib.graph.analysis import AnalysisConfig, run_probing_analysis
from lib.graph.data import GraphDataset, RegressionLabelStats
from lib.graph.pretrain import (
    PretrainCriterionBase,
    PretrainMethod,
    compute_linear_probing_preds,
)

ModelOut = namedtuple("ModelOut", ["predictions", "hidden"])
EvalOut = namedtuple("EvalOut", ["metrics", "predictions"])


class PretrainConfig(TypedDict):
    method: PretrainMethod | str
    params: KWArgs
    optimizer: KWArgs
    max_steps: int
    epoch_size: int
    patience: int


class FinetuneConfig(TypedDict):
    optimizer: KWArgs
    max_steps: int
    epoch_size: int
    patience: int


class Config(TypedDict):
    seed: int
    data: KWArgs
    model: KWArgs
    pretrain: NotRequired[PretrainConfig]
    finetune: NotRequired[FinetuneConfig]
    amp_dtype: NotRequired[Literal["bfloat16", "float16"]]
    synthetic_data: NotRequired[bool]
    analysis: NotRequired[AnalysisConfig]


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        n_num_features: int,
        n_classes: None | int,
        backbone: dict,
        num_features: None | dict = None,
        inp_out_type: str = "mlp",
    ):
        super().__init__()

        # >>> Continuous (numerical) features
        if num_features is None:
            self.num_embedding = None
            d_flat = n_features

        else:
            self.num_embedding = lib.graph.deep.make_module(
                **num_features,
                n_features=n_num_features,
            )
            d_embed = num_features["d_embed"]
            d_flat = n_features + n_num_features * (d_embed - 1)

        # >>> Input
        d_hidden = backbone["d_hidden"]
        dropout = backbone["dropout"]

        self.input = {
            "mlp": (
                nn.Sequential(
                    nn.Linear(d_flat, d_hidden),
                    nn.Dropout(dropout),
                    lib.graph.deep.get_activation_module(backbone["activation"]),
                    nn.Linear(d_hidden, d_hidden),
                )
            ),
            "linear": (
                nn.Sequential(
                    nn.Linear(d_flat, d_hidden),
                    nn.Dropout(dropout),
                    lib.graph.deep.get_activation_module(backbone["activation"]),
                )
            ),
        }[inp_out_type]

        # >>> Backbone
        self.backbone = lib.graph.deep.make_module(
            **backbone,
        )
        self.output_norm = lib.graph.deep.NORM_MODULES[backbone["norm_name"]](d_hidden)

        # >>> Output
        d_out = 1 if n_classes is None else n_classes
        self.output = {
            "mlp": (
                nn.Sequential(
                    nn.Linear(d_hidden, d_hidden),
                    lib.graph.deep.get_activation_module(backbone["activation"]),
                    nn.Linear(d_hidden, d_out),
                )
            ),
            "linear": (
                nn.Sequential(
                    nn.Linear(d_hidden, d_out),
                )
            ),
        }[inp_out_type]

    def forward(
        self,
        graph: dgl.DGLGraph,
        x_num: None | Tensor = None,
        x_other: None | Tensor = None,
    ) -> ModelOut:
        x = []
        if x_num is not None:
            x.append(x_num if self.num_embedding is None else self.num_embedding(x_num))

        if x_other is not None:
            x.append(x_other)

        x = torch.cat(x, dim=1)
        x = self.input(x)
        h = self.backbone(graph, x)
        y = self.output(self.output_norm(h))
        y = y.squeeze(-1).float()
        return ModelOut(predictions=y, hidden=h)


def train_loop(
    model: nn.Module,
    dataset: GraphDataset,
    step_fn: Callable[[], Tensor],
    eval_fn: Callable[[], EvalOut],
    config: PretrainConfig | FinetuneConfig,
    report: JSONDict,
    output: Path,
    timer: delu.tools.Timer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> None:
    optimizer = lib.deep.make_optimizer(
        **config["optimizer"], params=lib.deep.make_parameter_groups(model)
    )
    print("Num params:", sum([p.numel() for p in model.parameters()]))

    step = 0
    training_log = []
    early_stopping = delu.tools.EarlyStopping(config["patience"], mode="max")
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore

    grad_scaler = GradScaler(device.type) if amp_dtype is torch.float16 else None
    logger.info(f"AMP dtype: {amp_dtype}")

    def save_checkpoint() -> None:
        lib.dump_checkpoint(
            output,
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "random_state": delu.random.get_state(),
                "early_stopping": early_stopping,
                "report": report,
                "timer": timer,
                "training_log": training_log,
            },
        )

    print()
    timer.run()
    while step < config["max_steps"]:
        step_start_time = timer.elapsed()

        optimizer.zero_grad()
        loss = step_fn()

        if grad_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        step += 1
        step_end_time = timer.elapsed()
        step_loss = loss.detach().item()

        if step % config["epoch_size"] == 0:
            metrics, predictions = eval_fn()
            val_score_improved = (
                "metrics" not in report
                or metrics["val"]["score"] > report["metrics"]["val"]["score"]
            )

            training_log.append({"metrics": metrics, "time": timer.elapsed()})
            print(
                f"{'+' if val_score_improved else ' '}"
                f" [step] {step:>4}"
                f" [val] {metrics['val']['score']:.3f}"
                f" [test] {metrics['test']['score']:.3f}"
                f" [loss] {step_loss:.4f}"
                f" [time] {datetime.timedelta(seconds=math.trunc(timer.elapsed()))}"
                f" [s/it] {(step_end_time - step_start_time):.4f}"
            )
            writer.add_scalars("loss", {"train": step_loss}, step, timer.elapsed())

            if val_score_improved:
                report["best_step"] = step
                report["metrics"] = metrics
                save_checkpoint()
                lib.dump_report(output, report)
                lib.dump_summary(output, lib.summarize(report))

            early_stopping.update(metrics["val"]["score"])
            if (
                predictions is None
                or not lib.are_valid_predictions(predictions)
                or early_stopping.should_stop()
            ):
                break


def apply_model(
    model: Model,
    dataset: GraphDataset,
) -> ModelOut:
    graph = dataset.data["graph"]
    x_other = dataset.data["cat_features"]
    x_num = []
    for data_key in ["num_features", "ratio_features"]:
        if dataset.data[data_key] is not None:
            x_num.append(dataset.data[data_key])
    x_num = torch.cat(x_num, dim=1) if x_num else None
    return model(graph, x_num, x_other)


def supervised_loss(
    model: Model,
    dataset: GraphDataset[Tensor],
    amp_dtype: torch.dtype | None,
    device: torch.device,
) -> Tensor:
    model.train()

    targets: Tensor = dataset.data["labels"]  # type: ignore
    train_mask = dataset.data["masks"]["train"]

    loss_fn = (
        nn.functional.mse_loss
        if dataset.task.is_regression
        else nn.functional.cross_entropy
    )

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        preds = apply_model(model, dataset).predictions

    return loss_fn(
        preds[train_mask],
        targets[train_mask],
    )


def pretrain_loss(
    criterion: PretrainCriterionBase,
    dataset: GraphDataset,
    amp_dtype: torch.dtype | None,
    device: torch.device,
) -> Tensor:
    criterion.train()

    graph = dataset.data["graph"]
    x_other = dataset.data["cat_features"]
    x_num = []
    for data_key in ["num_features", "ratio_features"]:
        if dataset.data[data_key] is not None:
            x_num.append(dataset.data[data_key])
    x_num = torch.cat(x_num, dim=1) if x_num else None

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        return criterion(graph, x_num, x_other)  # type: ignore


# I mean, evaluatation part for this function and LP is the same
# So a lot of code is duplicated
@torch.inference_mode()
def evaluate(
    model: Model,
    dataset: GraphDataset,
    report: JSONDict,
    regression_label_stats: RegressionLabelStats | None,
    amp_dtype: torch.dtype | None,
    device: torch.device,
) -> EvalOut:
    model.eval()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        outputs = apply_model(model, dataset).predictions.cpu().numpy()

    if dataset.task.is_regression:
        assert regression_label_stats is not None
        _predictions: np.ndarray = (
            outputs * regression_label_stats.std + regression_label_stats.mean
        )
    else:
        _predictions: np.ndarray = scipy.special.softmax(outputs, axis=-1)
        if dataset.task.is_binclass:
            _predictions = _predictions[..., 1]

    predictions: dict[PartKey, np.ndarray] = {
        part: _predictions[dataset.task.masks[part]] for part in lib.DATA_PARTS
    }
    metrics = (
        dataset.task.calculate_metrics(predictions, report["prediction_type"])
        if lib.are_valid_predictions(predictions)
        else lib.get_default_metrics()
    )
    return EvalOut(metrics, predictions)


@torch.inference_mode()
def evaluate_linear_probing(
    model: Model,
    dataset: GraphDataset,
    report: JSONDict,
    regression_label_stats: RegressionLabelStats | None,
    amp_dtype: torch.dtype | None,
    device: torch.device,
) -> EvalOut:
    model.eval()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        hidden = apply_model(model, dataset).hidden.cpu().numpy()

    outputs = compute_linear_probing_preds(hidden, dataset)
    if outputs is None:
        return EvalOut(
            lib.get_default_metrics(),
            None,
        )

    if dataset.task.is_regression:
        assert regression_label_stats is not None
        _predictions: np.ndarray = (
            outputs * regression_label_stats.std + regression_label_stats.mean
        )
    else:
        _predictions: np.ndarray = scipy.special.softmax(outputs, axis=-1)
        if dataset.task.is_binclass:
            _predictions = _predictions[..., 1]

    predictions: dict[PartKey, np.ndarray] = {
        part: _predictions[dataset.task.masks[part]] for part in lib.DATA_PARTS
    }
    metrics = (
        dataset.task.calculate_metrics(predictions, report["prediction_type"])
        if lib.are_valid_predictions(predictions)
        else lib.get_default_metrics()
    )
    return EvalOut(metrics, predictions)


def main(
    config: Config | str | Path,
    output: None | str | Path = None,
    *,
    force: bool = False,
) -> None | lib.JSONDict:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"dgl\.backend\.pytorch\.sparse",
    )

    # >>> Start
    config, output = lib.check(config, output, config_type=Config)
    if not lib.start(main, output, force=force):
        return None

    lib.print_config(config)  # type: ignore
    print()
    delu.random.seed(config["seed"])
    device = lib.get_device()
    logger.info(f"Device: {device}")
    report = lib.create_report(main, config)  # type: ignore

    # >>> Data
    if config.get("synthetic_data", False):
        synthetic_seed = config["seed"]
    else:
        synthetic_seed = None

    dataset = lib.graph.data.build_dataset(
        synthetic_seed=synthetic_seed,
        **config["data"],
    )
    assert dataset.task.is_transductive

    if dataset.task.is_regression:
        dataset.data["labels"], regression_label_stats = (
            lib.graph.data.standardize_labels(
                dataset.data["labels"], dataset.data["masks"]
            )
        )
    else:
        regression_label_stats = None

    dataset = dataset.to_torch(device)
    dataset.data["graph"] = dataset.data["graph"].to(device)
    dataset.data["labels"] = (
        dataset.data["labels"].to(torch.long)  # type: ignore
        if dataset.task.is_classification
        else dataset.data["labels"].to(torch.float)  # type: ignore
    )

    # >>> Model
    model = Model(
        n_features=dataset.n_features,
        n_num_features=dataset.n_num_features,
        n_classes=dataset.task.try_compute_n_classes(),
        **config["model"],
    )
    report["n_parameters"] = lib.deep.get_n_parameters(model)
    logger.info(f"Number of parameters: {report['n_parameters']}")
    report["prediction_type"] = "labels" if dataset.task.is_regression else "probs"
    model.to(device)
    print(model)

    # >>> Utils
    timer = delu.tools.Timer()

    amp_dtype = config.get("amp_dtype")
    if amp_dtype == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                'amp_dtype is set to "bfloat16" in the config.'
                f" However, the current {device.type.upper()} device"
                " does not support bfloat16"
            )
        amp_dtype = torch.bfloat16

    elif amp_dtype == "float16":
        amp_dtype = torch.float16

    # >>> Pretrain
    if "pretrain" in config:
        if "finetune" in config:
            pretrain_report = copy.deepcopy(report)
        else:
            pretrain_report = report

        params = config["pretrain"]["params"] | {
            "d_hidden": config["model"]["backbone"]["d_hidden"],
            "activation": config["model"]["backbone"]["activation"],
        }

        if config["pretrain"]["method"] in [
            "graph_mae",
            "graph_mae2",
            "grasp",
            "grasp_naive",
        ]:
            params |= {
                "n_features": dataset.n_features,
            }

        if config["pretrain"]["method"] == "bgrl":
            params |= {
                "n_steps": config["pretrain"]["max_steps"],
                "device": device,
            }
        elif config["pretrain"]["method"] == "graph_mae2":
            params |= {"device": device}

        criterion = lib.graph.pretrain.build_criterion(
            backbone=model,
            method=config["pretrain"]["method"],
            params=params,
        ).to(device)

        step_fn = partial(
            pretrain_loss,
            criterion=criterion,
            dataset=dataset,
            amp_dtype=amp_dtype,
            device=device,
        )
        eval_fn = partial(
            evaluate_linear_probing,
            model=model,
            dataset=dataset,
            report=pretrain_report,
            regression_label_stats=regression_label_stats,
            amp_dtype=amp_dtype,
            device=device,
        )
        train_loop(
            model=criterion,
            dataset=dataset,
            step_fn=step_fn,
            eval_fn=eval_fn,
            config=config["pretrain"],
            report=pretrain_report,
            output=output,
            timer=timer,
            device=device,
            amp_dtype=amp_dtype,
        )
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=False)
        criterion.load_state_dict(checkpoint["model"])

    # >>> Finetune
    if "finetune" in config:
        step_fn = partial(
            supervised_loss,
            model=model,
            dataset=dataset,
            amp_dtype=amp_dtype,
            device=device,
        )
        eval_fn = partial(
            evaluate,
            model=model,
            dataset=dataset,
            report=report,
            regression_label_stats=regression_label_stats,
            amp_dtype=amp_dtype,
            device=device,
        )
        train_loop(
            model=model,
            dataset=dataset,
            step_fn=step_fn,
            eval_fn=eval_fn,
            config=config["finetune"],
            report=report,
            output=output,
            timer=timer,
            device=device,
            amp_dtype=amp_dtype,
        )
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=False)
        model.load_state_dict(checkpoint["model"])

    # >>> Analysis
    if "analysis" in config:
        model.eval()

        with torch.inference_mode():
            with torch.autocast(
                device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                hidden = apply_model(model, dataset).hidden

        features = dataset.data["num_features"]
        assert features is not None, "Analysis requires numerical features"
        features = torch.tensor(features)

        analysis_config = config["analysis"]
        report["analysis"] = run_probing_analysis(
            hidden=hidden,
            graph=dataset.data["graph"],
            features=features,
            **analysis_config,
        )
        logger.info(f"Analysis results: {report['analysis']}")

    # >>> Finish
    # to free-up space
    if config["seed"] > 0:
        os.remove(output / "checkpoint.pt")
    report["time"] = timer.elapsed()
    lib.finish(output, report)
    return report


if __name__ == "__main__":
    lib.init()
    lib.run(main)
