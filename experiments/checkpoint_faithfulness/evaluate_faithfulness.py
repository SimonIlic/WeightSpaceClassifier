"""
Faithfulness evaluation module for checkpoint faithfulness experiment.

This module provides:
- FaithfulnessCallback: Track mean_diff at every step during unlearning
- FaithfulnessResult: Dataclass for storing faithfulness metrics
- Utility functions for computing faithfulness metrics
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class FaithfulnessResult:
    """Results from a single unlearning run with faithfulness tracking."""

    model_idx: int
    target_class: int
    dataset: str
    condition: str
    seed: int

    # Original and final accuracies
    original_accuracy: List[float]
    accuracy_after: List[float]

    # Meta-network predictions
    init_pred: List[float]
    final_pred: List[float]

    # Faithfulness metrics
    initial_mae: float  # Mean |init_pred - original_accuracy|
    final_mae: float  # Mean |final_pred - accuracy_after|
    mean_diff_trajectory: List[float]  # mean_diff at every step

    # Unlearning metadata
    total_steps: int
    distance_travelled: float
    l2_distance: float

    # Optional: per-class errors
    target_initial_error: float = 0.0
    target_final_error: float = 0.0


class FaithfulnessCallback:
    """
    Callback for tracking mean_diff at every step during unlearning.

    This callback tracks the mean absolute error between the meta-network's
    predictions and the original accuracy at each step. This allows post-hoc
    analysis of faithfulness with different thresholds.

    Usage:
        callback = FaithfulnessCallback(original_accuracy)
        state = unlearn(..., step_callback=callback)
        trajectory = callback.mean_diff_trajectory
    """

    def __init__(self, original_accuracy: np.ndarray):
        """
        Initialize the callback.

        Args:
            original_accuracy: Array of original per-class accuracies (10,)
        """
        self.original_accuracy = np.asarray(original_accuracy)
        self.mean_diff_trajectory: List[float] = []
        self.pred_history: List[np.ndarray] = []

    def __call__(self, step: int, pred: torch.Tensor, weights: torch.Tensor) -> None:
        """
        Called at each unlearning step.

        Args:
            step: Current step number
            pred: Meta-network predictions (10,)
            weights: Current CNN weights (not used for mean_diff, but available)
        """
        pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred
        mean_diff = np.abs(pred_np - self.original_accuracy).mean()
        self.mean_diff_trajectory.append(float(mean_diff))
        self.pred_history.append(pred_np.copy())

    def get_unfaithfulness_step(self, threshold: float = 0.1) -> int:
        """
        Find the first step where mean_diff exceeds a threshold.

        Args:
            threshold: Unfaithfulness threshold

        Returns:
            Step number where unfaithfulness was first detected, or -1 if never
        """
        for step, mean_diff in enumerate(self.mean_diff_trajectory):
            if mean_diff > threshold:
                return step
        return -1

    def reset(self) -> None:
        """Reset the callback for a new run."""
        self.mean_diff_trajectory = []
        self.pred_history = []


def compute_faithfulness_metrics(
    init_pred: np.ndarray,
    final_pred: np.ndarray,
    original_accuracy: np.ndarray,
    accuracy_after: np.ndarray,
    target_class: int,
) -> dict:
    """
    Compute faithfulness metrics from predictions and actual accuracies.

    Args:
        init_pred: Initial meta-network predictions (10,)
        final_pred: Final meta-network predictions (10,)
        original_accuracy: Original per-class accuracies (10,)
        accuracy_after: Per-class accuracies after unlearning (10,)
        target_class: Target class being unlearned

    Returns:
        Dictionary with computed metrics
    """
    initial_mae = np.abs(init_pred - original_accuracy).mean()
    final_mae = np.abs(final_pred - accuracy_after).mean()
    target_initial_error = abs(init_pred[target_class] - original_accuracy[target_class])
    target_final_error = abs(final_pred[target_class] - accuracy_after[target_class])

    return {
        "initial_mae": float(initial_mae),
        "final_mae": float(final_mae),
        "target_initial_error": float(target_initial_error),
        "target_final_error": float(target_final_error),
    }


def create_faithfulness_result(
    model_idx: int,
    target_class: int,
    dataset: str,
    condition: str,
    seed: int,
    original_accuracy: np.ndarray,
    accuracy_after: np.ndarray,
    init_pred: np.ndarray,
    final_pred: np.ndarray,
    callback: FaithfulnessCallback,
    total_steps: int,
    distance_travelled: float,
    l2_distance: float,
) -> FaithfulnessResult:
    """
    Create a FaithfulnessResult from unlearning outputs.

    Args:
        model_idx: Index of the model in the dataset
        target_class: Class being unlearned
        dataset: Dataset name
        condition: Training condition ('final-only' or 'multi-stage')
        seed: Random seed used for meta-network
        original_accuracy: Original per-class accuracies
        accuracy_after: Per-class accuracies after unlearning
        init_pred: Initial meta-network predictions
        final_pred: Final meta-network predictions
        callback: FaithfulnessCallback with trajectory data
        total_steps: Number of unlearning steps taken
        distance_travelled: Cumulative L2 distance in weight space
        l2_distance: Final L2 distance from original weights

    Returns:
        FaithfulnessResult dataclass instance
    """
    metrics = compute_faithfulness_metrics(
        init_pred, final_pred, original_accuracy, accuracy_after, target_class
    )

    return FaithfulnessResult(
        model_idx=model_idx,
        target_class=target_class,
        dataset=dataset,
        condition=condition,
        seed=seed,
        original_accuracy=list(original_accuracy),
        accuracy_after=list(accuracy_after),
        init_pred=list(init_pred),
        final_pred=list(final_pred),
        initial_mae=metrics["initial_mae"],
        final_mae=metrics["final_mae"],
        mean_diff_trajectory=callback.mean_diff_trajectory,
        total_steps=total_steps,
        distance_travelled=distance_travelled,
        l2_distance=l2_distance,
        target_initial_error=metrics["target_initial_error"],
        target_final_error=metrics["target_final_error"],
    )
