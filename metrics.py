# a collection of metrics to use in evaluating unlearning performance
# all functions take two iterables (any-type) and compute unlearning metric for the target class index

def mean_difference(acc_before, acc_after, target_idx: int):
    """
    Calculate mean unlearned metric.

    Defined as the difference in accuracy drop for the target class compared to the average accuracy drop for all other classes.
    """
    delta_target = acc_before[target_idx] - acc_after[target_idx]
    delta_others = [acc_before[i] - acc_after[i] for i in range(len(acc_before)) if i != target_idx]
    mean_delta_others = sum(delta_others) / len(delta_others)
    return delta_target - mean_delta_others

def clipped_negative_mean_difference(acc_before, acc_after, target_idx, proportional=False):
    """
    Calculate clipped mean unlearned metric.

    Similar to mean unlearned but clips negative differences to zero.
    """
    if proportional:
        delta_target = (acc_before[target_idx] - acc_after[target_idx]) / acc_before[target_idx]
        delta_others = [(acc_before[i] - acc_after[i]) / acc_before[i] for i in range(len(acc_before)) if i != target_idx]
    else:
        delta_target = acc_before[target_idx] - acc_after[target_idx]
        delta_others = [acc_before[i] - acc_after[i] for i in range(len(acc_before)) if i != target_idx]
    return max(0.0, delta_target) - sum(max(0.0, d) for d in delta_others) / len(delta_others)

def max_difference(acc_before, acc_after, target_idx):
    """
    Calculate max unlearned metric.

    Defined as the difference in accuracy drop for the target class compared to the maximum accuracy drop for all other classes.
    """
    delta_target = acc_before[target_idx] - acc_after[target_idx]
    delta_others = [acc_before[i] - acc_after[i] for i in range(len(acc_before)) if i != target_idx]
    return delta_target - max(delta_others)

def min_difference(acc_before, acc_after, target_idx):
    """
    Calculate min unlearned metric.

    Defined as the difference in accuracy drop for the target class compared to the minimum accuracy drop for all other classes.
    """
    delta_target = acc_before[target_idx] - acc_after[target_idx]
    delta_others = [acc_before[i] - acc_after[i] for i in range(len(acc_before)) if i != target_idx]
    return delta_target - min(delta_others)

if __name__ == "__main__":
    # Example usage
    import numpy as np
    acc_before = np.array([0.5, 0.5, 0.5])
    acc_after = np.array([0.25, 0, 0.25])
    acc_before2 = np.array([1, 1, 1])
    acc_after2 = np.array([0.5, 0, 0.5])
    target_idx = 1

    metric_1 = clipped_negative_mean_difference(acc_before, acc_after, target_idx, proportional=True)
    metric_2 = clipped_negative_mean_difference(acc_before2, acc_after2, target_idx, proportional=True)
    print(f"Clipped Mean Unlearned 1: {metric_1:.2f}")
    print(f"Clipped Mean Unlearned 2: {metric_2:.2f}")
    assert metric_1 == metric_2, "Metrics should be equal for proportional drops"

    acc_before = np.array([0.8, 0.8, 0.85, 0.95])
    acc_after = np.array([0.9, 0.75, 0.5, 0.9])
    target_idx = 2

    print("Unlearning index", target_idx)
    print("Before:", acc_before)
    print("After:", acc_after)
    print("Delta:", acc_after - acc_before)

    print(f"Mean Unlearned: {mean_difference(acc_before, acc_after, target_idx):.2f}")
    print(f"Clipped Mean Unlearned: {clipped_negative_mean_difference(acc_before, acc_after, target_idx):.2f}")
    print(f"Max Unlearned: {max_difference(acc_before, acc_after, target_idx):.2f}")
    print(f"Min Unlearned: {min_difference(acc_before, acc_after, target_idx):.2f}")
