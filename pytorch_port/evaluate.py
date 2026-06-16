from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from keras.datasets import cifar10, fashion_mnist, mnist


def evaluate_classifier(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    batch_size: int = 256,
    num_classes: int | None = None,
) -> Tuple[float, List[float]]:
    """Evaluate a PyTorch SmallCNN classifier.

    Args:
        model: A SmallCNN (or compatible) nn.Module in eval mode.
        x_test: Test images in NHWC format, shape (N, H, W, C).
        y_test: Integer labels, shape (N,) or (N, 1).
        batch_size: Batch size for inference.
        num_classes: If provided, forces that many classes; otherwise
            inferred from y_test.

    Returns:
        overall_acc: Top-1 accuracy on the whole test set.
        per_class_acc: Accuracy for every class 0..num_classes-1.
    """
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("x_test and y_test must have the same length")

    if num_classes is None:
        num_classes = int(np.max(y_test) + 1)

    device = next(model.parameters()).device

    y_test_flat = y_test.flatten()

    all_logits = []
    for start in range(0, len(x_test), batch_size):
        batch = x_test[start : start + batch_size]
        # PyTorch expects NCHW format, so we need to transpose from NHWC TF format
        x = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            logits = model(x)
        all_logits.append(logits.cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    y_pred = np.argmax(logits_arr, axis=1)

    overall_acc = float(np.mean(y_pred == y_test_flat))

    per_class_acc: list[float] = []
    for cls in range(num_classes):
        idx = y_test_flat == cls
        if idx.any():
            per_class_acc.append(float(np.mean(y_pred[idx] == y_test_flat[idx])))
        else:
            per_class_acc.append(float("nan"))

    return overall_acc, per_class_acc


# functionaly identical to dataset loader in src/utils
def load_testset_data(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load test set images and labels for the specified dataset.

    Uses the same data sources and preprocessing as the TF version
    (keras.datasets for MNIST/Fashion-MNIST/CIFAR-10, tfds for SVHN).

    Args:
        dataset: One of {'mnist', 'fashion_mnist', 'cifar10', 'svhn_cropped'}.

    Returns:
        (x_test, y_test) where x_test is in NHWC format, float32,
        normalized to [-1, 1], with channel dimension.
    """
    if dataset == "fashion_mnist":
        data = fashion_mnist.load_data()
    elif dataset == "mnist":
        data = mnist.load_data()
    elif dataset == "cifar10":
        data = cifar10.load_data()
        x_test, y_test = data[1]
        x_test = np.mean(x_test, axis=-1, keepdims=False)
        data = (data[0], (x_test, y_test))
    elif dataset == "svhn_cropped":
        import tensorflow_datasets as tfds

        svhn_test = tfds.load("svhn_cropped", split="test", as_supervised=True, batch_size=-1)
        x_test, y_test = tfds.as_numpy(svhn_test)
        x_test = np.mean(x_test, axis=-1, keepdims=False)
        data = ((None, None), (x_test, y_test))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    (_, _), (x_test, y_test) = data
    x_test = x_test.astype("float32") / 127.5 - 1.0
    x_test = x_test[..., None]

    return x_test, y_test
