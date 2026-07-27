"""Data loading: CNN Zoo weights (canonical split) and test-set images.

The zoo-loading logic is copied near-verbatim from
cnn_surgery.utils.load_dataset so the canonical seed-123 train/test/val split
is preserved exactly. Image data comes from torchvision instead of
keras/tfds; preprocessing matches the original (greyscale via channel mean,
normalize to [-1, 1]).
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEFAULT_SEED = 123  # canonical seed for this project's splits
DATAFRAME_CONFIG_COLS = [
    "config.w_init",
    "config.activation",
    "config.learning_rate",
    "config.init_std",
    "config.l2reg",
    "config.train_fraction",
    "config.dropout",
    "config.optimizer",
]
DATAFRAME_METRIC_COLS = [
    "test_accuracy",
    "test_loss",
    "train_accuracy",
    "train_loss",
]
DATAFRAME_CLASS_ACCURACY_COLS = ["accuracy_class_" + str(i) for i in range(10)]
TRAIN_SIZE = 15_000


def find_zoo_dir() -> Path:
    """Locate the model_zoo directory by searching upward from this file and cwd."""
    candidates = [Path(__file__).resolve()] + [Path.cwd().resolve()]
    for start in candidates:
        for parent in [start] + list(start.parents):
            if (parent / "model_zoo").is_dir():
                return parent / "model_zoo"
    raise FileNotFoundError(
        "Could not find a 'model_zoo' directory above slimdown/ or the current working directory. Pass --zoo-dir explicitly."
    )


def filter_checkpoints(weights, dataframe, stage="final", load_class_acc=False):
    """Take one checkpoint per run (same logic as cnn_surgery.utils.load_dataset)."""
    if load_class_acc:
        return_cols = DATAFRAME_METRIC_COLS + DATAFRAME_CLASS_ACCURACY_COLS
    else:
        return_cols = DATAFRAME_METRIC_COLS

    ids_to_take = []
    current_uid = dataframe.axes[0][0].split("/")[-2]
    steps = []
    for i in range(len(dataframe.axes[0])):
        ckpt = dataframe.axes[0][i]
        parts = ckpt.split("/")
        if parts[-2] == current_uid:
            steps.append(int(parts[-1].split("-")[-1]))
        else:
            steps_sort = sorted(steps)
            if stage == "final":
                target_step = steps_sort[-1]
            elif stage == "early":
                target_step = steps_sort[0]
            else:  # middle
                target_step = steps_sort[int(len(steps) / 2)]
            offset = [j for (j, el) in enumerate(steps) if el == target_step][0]
            ids_to_take.append(i - len(steps) + offset)
            current_uid = parts[-2]
            steps = [int(parts[-1].split("-")[-1])]

    hyperparams = dataframe[DATAFRAME_CONFIG_COLS]
    hyperparams = hyperparams.iloc[ids_to_take]

    return (
        weights[ids_to_take, :],
        dataframe[return_cols].values[ids_to_take, :].astype(np.float32),
        hyperparams,
    )


def load_zoo(
    dataset: str,
    zoo_dir: str | os.PathLike | None = None,
    train_size: int = TRAIN_SIZE,
    stage: str = "final",
    metrics_file: str = "metrics_merged_final.csv",
    seed: int = DEFAULT_SEED,
):
    """Load CNN Zoo weights, per-class metrics, and configs with the canonical split.

    Returns ((weights, metrics, configs) for train, test, val) — same order and
    content as cnn_surgery.utils.load_dataset.load_dataset(load_class_acc=True).
    """
    zoo_dir = Path(zoo_dir) if zoo_dir is not None else find_zoo_dir()
    dirname = zoo_dir / dataset
    if not dirname.is_dir():
        raise FileNotFoundError(f"No zoo data for dataset '{dataset}' at {dirname}")

    weights = np.load(dirname / "weights.npy", mmap_mode="r")
    metrics_df = pd.read_csv(dirname / metrics_file, index_col=0)

    weights_flt, metrics_flt, configs_flt = filter_checkpoints(weights, metrics_df, stage=stage, load_class_acc=True)

    # Filter out DNNs with NaNs/Infs in their weights
    idx_valid = np.isfinite(weights_flt).mean(1) == 1.0
    inputs = np.asarray(weights_flt[idx_valid], dtype=np.float32)
    outputs = np.asarray(metrics_flt[idx_valid], dtype=np.float32)
    configs = configs_flt.iloc[idx_valid]

    random_idx = np.arange(inputs.shape[0])
    if seed == DEFAULT_SEED:
        logging.info(f"Using canonical seed {DEFAULT_SEED}: splits match the original codebase.")
    np.random.seed(seed)
    np.random.shuffle(random_idx)

    test_size = inputs.shape[0] - train_size
    val_size = test_size // 2
    test_size -= val_size

    def split(sl):
        return inputs[random_idx[sl]], outputs[random_idx[sl]], configs.iloc[random_idx[sl]]

    train = split(slice(0, train_size))
    test = split(slice(train_size, train_size + test_size))
    val = split(slice(train_size + test_size, None))
    return train, test, val


def load_testset_data(dataset: str, root: str | os.PathLike | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Load test images and labels via torchvision.

    Returns:
        x_test: (N, 1, H, W) float32 tensor, normalized to [-1, 1] (greyscale).
        y_test: (N,) int64 tensor.
    """
    from torchvision import datasets  # local import: keeps torchvision optional for zoo-only use

    root = str(root) if root is not None else str(Path.home() / ".cache" / "slimdown_datasets")

    if dataset == "mnist":
        ds = datasets.MNIST(root, train=False, download=True)
        x = ds.data.numpy().astype(np.float32)  # (N, 28, 28)
        y = ds.targets.numpy()
    elif dataset == "fashion_mnist":
        ds = datasets.FashionMNIST(root, train=False, download=True)
        x = ds.data.numpy().astype(np.float32)
        y = ds.targets.numpy()
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(root, train=False, download=True)
        x = ds.data.astype(np.float32).mean(axis=-1)  # (N, 32, 32, 3) -> greyscale
        y = np.asarray(ds.targets)
    elif dataset == "svhn_cropped":
        ds = datasets.SVHN(root, split="test", download=True)
        x = ds.data.astype(np.float32).mean(axis=1)  # (N, 3, 32, 32) -> greyscale
        y = ds.labels
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    x = x / 127.5 - 1.0
    x_test = torch.from_numpy(x).float().unsqueeze(1)  # (N, 1, H, W)
    y_test = torch.from_numpy(np.asarray(y)).long().flatten()
    return x_test, y_test
