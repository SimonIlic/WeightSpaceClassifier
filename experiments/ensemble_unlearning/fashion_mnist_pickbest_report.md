# Fashion-MNIST Pick-Best Ensemble Unlearning

## Overview
We ran the per-class pick-best unlearning experiment using 5 update meta-nets (0–4) and 5 hold-out meta-nets (5–9). For each target class, 100 validation CNNs were sampled with base target accuracy ≥0.6. Updates optimized predicted target-class accuracy via update nets; edits were selected by lowest hold-out target accuracy. Results are per-class and aggregated across all samples.

Key idea: gradients come only from the update set; selection uses the hold-out set to avoid overfitting to a single meta-net’s bias. Filtering to higher-base models (≥0.6 target accuracy) forces the procedure to operate on non-trivial models and reduces “already-low” edge cases.

## Configuration
- Dataset: `fashion_mnist`
- Samples per class: `100` (validation split), filtered by base target acc ≥ `0.6`
- Update nets: `0 1 2 3 4`; Hold-out nets: `5 6 7 8 9`
- Hyperparameters: `lr=0.01`, `max_steps=2000`, `stop_threshold=0.3`, no L2
- Loss: `simple`
- Outputs per class: `experiments/ensemble_unlearning/data/ensemble_holdout/fashion_mnist_class{c}.csv`

## Aggregate Metrics (all classes, 1000 runs)
- Mean target_drop: **0.204**; success rate (drop>0): **80.1%**
- Clipped negative mean difference: mean **0.159**; positive in **70.4%** of runs
- Zero-step fraction: **14%** (vs ~36% before filtering), indicating many runs now perform genuine updates rather than stopping immediately.
- Non-target drop remains small on average; positive clipped metric in most runs indicates target suppression with limited collateral damage.

## Per-Class Highlights
- Strongest medians (target_drop / clipped metric):
  - Class 4: **0.304 / 0.273**
  - Class 2: **0.221 / 0.177**
  - Class 6: **0.204 / 0.194**
  - Class 7: **0.165 / 0.092**
- All classes show non-negative median clipped metric; upper quartiles reach ~0.33–0.48 target_drop for several classes. Even weaker classes (0,1,8,9) still show positive medians, but spreads are wider.

### Spread (boxplot summary)
- Overall quartiles (target_drop): 25% `0.006`, median `0.086`, 75% `0.373`.
- Overall quartiles (clipped metric): 25% `0.000`, median `0.056`, 75% `0.277`.
- Per-class quartiles: see boxplot figure; classes 2/4/6/7 have both higher medians and higher upper quartiles, indicating more consistent, stronger unlearning.

### Success and steps
- Success rates per class range ~70–89%. Classes 2,4,7 lead (~86–89%), with class 6 close behind (~80%).
- Mean steps are typically 730–1100, reflecting the higher `max_steps=2000` and fewer immediate stops. Residual 0-step runs (14%) likely correspond to update nets already predicting target < threshold.

## Figures
- Mean summary (target_drop, clipped metric, success rate): `experiments/ensemble_unlearning/plots/fashion_mnist_pickbest_summary_min0.6.png`
- Mean summary with spread (±1 std as error bars): `experiments/ensemble_unlearning/plots/fashion_mnist_pickbest_summary_min0.6_std.png`
- Distribution boxplots (per-class spread across 100 CNNs): `experiments/ensemble_unlearning/plots/fashion_mnist_pickbest_boxplots_min0.6.png`

## Discussion
- **Effect of filtering**: Requiring base target accuracy ≥0.6 substantially improved both target_drop and clipped metric and reduced the zero-step rate. This confirms that the method benefits from starting on non-trivial models rather than already-low target accuracy cases.
- **Balance of target vs. non-target**: Positive clipped metrics in ~70% of runs indicate that target suppression generally outweighs any collateral drops. Non-target drops remain modest on average, though the boxplots show some variance—especially in harder classes—so per-class tuning might help.
- **Meta-net ensemble role**: Using disjoint update/eval sets appears to produce robust selections; high success rates suggest hold-out scoring aligns with observed drops. The remaining failures may stem from models whose predicted target accuracy is already low or whose gradients stagnate despite higher max steps.
- **Headroom**: Several classes show 75th-percentile target drops around 0.33–0.48, suggesting there is remaining room to push median performance up by adjusting loss/thresholds or adding a mild step-based stopping floor.

## Notes / Next Steps
- Consider a minimum-step rule (e.g., require ≥K updates) or a step-based stop to reduce the last 14% zero-step cases.
- Per-class loss tuning (e.g., boost loss with small beta) might lift weaker classes (0,1,8,9) without harming non-targets.
- If available, log full per-class post-edit predictions to compute exact clipped metrics (not just mean non-target drops).
