"""
Benchmarking suite for evaluating unlearning methods.

This module provides functions to compute standard unlearning metrics:
- Forget Accuracy (FA): Accuracy on the forget set (should be low after unlearning)
- Retain Accuracy (RA): Accuracy on the retain set (should be preserved)
- Jensen-Shannon Distance: Behavioral similarity between models

The JS distance metric follows Rangel et al. (2024) and Chundawat et al. (2023),
computing per-sample JSD between softmax outputs and averaging across the test set.

Example usage:
    from benchmark_suite import benchmark_unlearning, js_similarity_score

    scores = benchmark_unlearning(
        unlearned_weights=edited_weights,
        baseline_weights={'finetune': ft_weights, 'gradascent': ga_weights},
        test_data=(x_test, y_test),
        forget_class=3,
        activation='relu'
    )
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

from cnn_surgery.utils.reconstruct_network import reconstruct_network
from cnn_surgery.utils.evaluate_per_class_accuracy import evaluate_classifier


def get_softmax_outputs(
    weights: np.ndarray,
    activation: str,
    x_test: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Get softmax probability outputs from a model.

    Parameters
    ----------
    weights : np.ndarray
        Flattened CNN weight vector.
    activation : str
        Activation function name (e.g., 'relu', 'tanh').
    x_test : np.ndarray
        Test images.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    np.ndarray
        Softmax probabilities of shape (n_samples, n_classes).
    """
    model = reconstruct_network(weights, activation)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Get raw logits
    logits = model.predict(x_test, batch_size=batch_size, verbose=0)

    # Apply softmax to get probabilities
    # Subtract max for numerical stability
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return softmax


def js_similarity_score(
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    activation: str,
    x_test: np.ndarray,
    batch_size: int = 256,
) -> float:
    """
    Compute Jensen-Shannon similarity between two models.

    Following Rangel et al. (2024), we compute:
        φ = 1 - mean_x[JSD(softmax_a(x), softmax_b(x))]

    where φ ≈ 1 indicates identical behavior.

    Parameters
    ----------
    weights_a : np.ndarray
        Flattened CNN weight vector for first model.
    weights_b : np.ndarray
        Flattened CNN weight vector for second model.
    activation : str
        Activation function name.
    x_test : np.ndarray
        Test images.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float
        Similarity score φ in [0, 1]. Higher means more similar.
    """
    softmax_a = get_softmax_outputs(weights_a, activation, x_test, batch_size)
    softmax_b = get_softmax_outputs(weights_b, activation, x_test, batch_size)

    # Compute per-sample JSD and average
    n_samples = softmax_a.shape[0]
    jsd_values = np.array([jensenshannon(softmax_a[i], softmax_b[i]) for i in range(n_samples)])

    # Handle any NaN values (can occur with degenerate distributions)
    jsd_values = np.nan_to_num(jsd_values, nan=0.0)

    mean_jsd = np.mean(jsd_values)

    # flip metrics so that higher is better - turn js distance into js similarity
    return float(1.0 - mean_jsd)


def js_similarity_to_population(
    weights: np.ndarray,
    population_reference: np.ndarray,
    activation: str,
    x_test: np.ndarray,
    batch_size: int = 256,
) -> float:
    """
    Compute JS similarity between a model and a population reference distribution.

    Used for comparing to retrain-from-scratch baselines where no 1-to-1
    model correspondence exists.

    Parameters
    ----------
    weights : np.ndarray
        Flattened CNN weight vector for the model to evaluate.
    population_reference : np.ndarray
        Pre-computed reference distribution of shape (n_samples, n_classes),
        typically the mean softmax outputs across a population of retrained models.
    activation : str
        Activation function name.
    x_test : np.ndarray
        Test images.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float
        Similarity score φ in [0, 1].
    """
    softmax = get_softmax_outputs(weights, activation, x_test, batch_size)

    n_samples = softmax.shape[0]
    jsd_values = np.array([jensenshannon(softmax[i], population_reference[i]) for i in range(n_samples)])

    jsd_values = np.nan_to_num(jsd_values, nan=0.0)
    mean_jsd = np.mean(jsd_values)
    return float(1.0 - mean_jsd)


def compute_population_reference(
    population_weights: List[np.ndarray],
    activation: str,
    x_test: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute reference distribution by averaging softmax outputs across a population.

    Parameters
    ----------
    population_weights : list of np.ndarray
        List of flattened weight vectors for the population of models.
    activation : str
        Activation function name.
    x_test : np.ndarray
        Test images.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    np.ndarray
        Mean softmax distribution of shape (n_samples, n_classes).
    """
    all_softmax = []
    for weights in population_weights:
        softmax = get_softmax_outputs(weights, activation, x_test, batch_size)
        all_softmax.append(softmax)

    # Stack and average across models
    stacked = np.stack(all_softmax, axis=0)  # (n_models, n_samples, n_classes)
    mean_softmax = np.mean(stacked, axis=0)  # (n_samples, n_classes)

    return mean_softmax


def forget_accuracy(
    weights: np.ndarray,
    activation: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    forget_class: int,
    batch_size: int = 256,
) -> float:
    """
    Compute accuracy on the forget set (samples of the forget class).

    For successful unlearning, this should be low (~0).

    Parameters
    ----------
    weights : np.ndarray
        Flattened CNN weight vector.
    activation : str
        Activation function name.
    x_test : np.ndarray
        Test images.
    y_test : np.ndarray
        Test labels.
    forget_class : int
        The class to forget.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float
        Accuracy on forget class samples (= recall for that class).
    """
    model = reconstruct_network(weights, activation)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    _, per_class_acc = evaluate_classifier(model, x_test, y_test, batch_size=batch_size)
    return float(per_class_acc[forget_class])


def retain_accuracy(
    weights: np.ndarray,
    activation: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    forget_class: int,
    batch_size: int = 256,
) -> float:
    """
    Compute accuracy on the retain set (samples NOT of the forget class).

    For successful unlearning, this should remain high.

    Parameters
    ----------
    weights : np.ndarray
        Flattened CNN weight vector.
    activation : str
        Activation function name.
    x_test : np.ndarray
        Test images.
    y_test : np.ndarray
        Test labels.
    forget_class : int
        The class to forget (excluded from retain set).
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float
        Accuracy on retain set samples.
    """
    model = reconstruct_network(weights, activation)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    _, per_class_acc = evaluate_classifier(model, x_test, y_test, batch_size=batch_size)

    # Simple average of retain class accuracies (assumes balanced dataset)
    retain_accs = [per_class_acc[c] for c in range(len(per_class_acc)) if c != forget_class]
    return float(np.nanmean(retain_accs))


def benchmark_unlearning(
    unlearned_weights: np.ndarray,
    baseline_weights: Dict[str, np.ndarray],
    test_data: np.ndarray,
    test_labels: np.ndarray,
    forget_class: int,
    activation: str,
    population_reference: Optional[np.ndarray] = None,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Compute all benchmark metrics for an unlearned model.

    Parameters
    ----------
    unlearned_weights : np.ndarray
        Flattened weight vector of the unlearned model.
    baseline_weights : dict
        Dictionary mapping baseline names to their weight vectors.
        E.g., {'finetune': ft_weights, 'gradascent': ga_weights}
    test_data : tuple
        (x_test, y_test) test images and labels.
    forget_class : int
        The class being unlearned.
    activation : str
        Activation function name.
    population_reference : np.ndarray, optional
        Pre-computed population reference for retrain-from-scratch comparison.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    dict
        Dictionary of metric names to values.
    """
    x_test = test_data
    y_test = test_labels

    results = {
        "forget_accuracy": forget_accuracy(unlearned_weights, activation, x_test, y_test, forget_class, batch_size),
        "retain_accuracy": retain_accuracy(unlearned_weights, activation, x_test, y_test, forget_class, batch_size),
    }

    # Compute JS similarity to each baseline
    for name, weights in baseline_weights.items():
        js_key = f"js_similarity_{name}"
        results[js_key] = js_similarity_score(unlearned_weights, weights, activation, x_test, batch_size)

    # Compute JS similarity to population reference if provided
    if population_reference is not None:
        results["js_similarity_retrain_population"] = js_similarity_to_population(
            unlearned_weights, population_reference, activation, x_test, batch_size
        )

    return results


def benchmark_baseline(
    baseline_weights: np.ndarray,
    unlearned_weights: np.ndarray,
    test_data: Tuple[np.ndarray, np.ndarray],
    forget_class: int,
    activation: str,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Compute metrics for a baseline model (for table comparison).

    Parameters
    ----------
    baseline_weights : np.ndarray
        Flattened weight vector of the baseline model.
    unlearned_weights : np.ndarray
        Flattened weight vector of our unlearned model (for JS comparison).
    test_data : tuple
        (x_test, y_test) test images and labels.
    forget_class : int
        The class being unlearned.
    activation : str
        Activation function name.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    dict
        Dictionary with 'forget_accuracy', 'retain_accuracy', and 'js_with_ours'.
    """
    x_test, y_test = test_data

    return {
        "forget_accuracy": forget_accuracy(baseline_weights, activation, x_test, y_test, forget_class, batch_size),
        "retain_accuracy": retain_accuracy(baseline_weights, activation, x_test, y_test, forget_class, batch_size),
        "js_with_ours": js_similarity_score(baseline_weights, unlearned_weights, activation, x_test, batch_size),
    }


if __name__ == "__main__":
    # Example usage / sanity check
    import numpy as np

    print("Benchmark suite loaded successfully.")
    print("Available functions:")
    print("  - js_similarity_score(weights_a, weights_b, activation, x_test)")
    print("  - js_similarity_to_population(weights, population_ref, activation, x_test)")
    print("  - compute_population_reference(population_weights, activation, x_test)")
    print("  - forget_accuracy(weights, activation, x_test, y_test, forget_class)")
    print("  - retain_accuracy(weights, activation, x_test, y_test, forget_class)")
    print("  - benchmark_unlearning(...)")
    print("  - benchmark_baseline(...)")
