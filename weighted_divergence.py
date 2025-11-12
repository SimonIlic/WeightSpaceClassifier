import numpy as np

def weighted_divergence(before, after, weights=None, eps=1e-9):
    before = np.clip(before, eps, 1)
    after  = np.clip(after,  eps, 1)
    if weights is None:
        weights = np.ones_like(before) / len(before)
    else:
        weights = weights / weights.sum()
    return np.sum(weights * np.log(before / after))

# Example: weight by initial accuracy
before = np.array([0.9, 0.7, 0.3])
after  = np.array([0.8, 0.8, 0.8])
weights = before / before.sum()

print(weighted_divergence(before, after, weights))
