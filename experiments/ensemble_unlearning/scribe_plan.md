# Ensemble Metanet Unlearning (Pick-Best) – Notes

## What was done (MNIST pilot)
- Loaded validation weights/accuracies via `load_multi_stage_dataset(dataset="mn
ist")`, sampled 20 models.
- Loaded update meta-nets `{0..4}` and hold-out eval meta-nets `{5..9}` from `me
tanetworks/`.
- Implemented per-meta-net unlearning, then picked the edited weights that minim
ize hold-out mean target-class accuracy.
- Settings: target_class=3, lr=0.01, max_steps=60, stop when hold-out mean targe
t accuracy < 0.2, l2_penalty=1e-6.
- Results (hold-out eval): 
  - Single meta-net: target drop ~0.0063
  - Joint ensemble loss: target drop ~0.0063
  - Per-net, pick-best: target drop ~0.0121 (≈2×), non-target drop ~0.0035, dist
ance ~0.0287.
- Many sampled models already had low target accuracy; filtering to higher-base-
acc models should show stronger effects.

## Why hold-out meta-nets
- Gradients use only the update set; the hold-out set evaluates edits to avoid o
verfitting to a single net’s bias.
- In “pick-best”, each unlearning run is scored by the hold-out mean; we select 
the run with lowest hold-out target accuracy.

## Next experiment (Fashion-MNIST, all classes)
- For each class 0–9:
  - Sample ~50 models (per-class) from validation split.
  - Use 5 update meta-nets (e.g., `meta_network_fashion_mnist_0..4.pkl`) and 5 h
old-out eval meta-nets (e.g., `5..9`).
  - Run per-net unlearning, pick the best per sample by hold-out mean target acc
uracy.
  - Log per-class CSV: base_target_acc, best_target_acc, target_drop, nontarget_
drop, distance, steps, winning_net_id, sample index.
- Hyperparams to start: lr=0.01, max_steps=60, stop_threshold=0.2, l2_penalty=1e
-6. Optionally tune.

## Files to add (when write access is available)
1) **Driver (Python)**: A small script (or function) akin to `evaluate_models.py
` to:
   - Load Fashion-MNIST multi-stage data.
   - Sample ~50 models per class from validation.
   - Load update/eval meta-nets.
   - Run per-net unlearning + pick-best, aggregate per-class results, write CSV 
to `experiments/ensemble_unlearning/fashion_mnist_class{c}.csv`.
2) **Shell runner** (inspired by `per_class_evaluation.sh` and `experiments/good
-bad/fashion_mnist_job.sh`):
   - Loop over classes 0..9.
   - Set seeds and paths to meta-nets, sample size=50, update_nets=5, eval_nets=
5.
   - Call the driver with class id and output path.

## Suggested structure/commands
- Driver invocation example:
  ```
  python -m scripts.run_pickbest_unlearning \\
    --dataset fashion_mnist \\
    --target_class ${CLASS} \\
    --num_samples 50 \\
    --update_nets 0 1 2 3 4 \\
    --eval_nets 5 6 7 8 9 \\
    --lr 0.01 --steps 60 --stop_threshold 0.2 \\
    --out_csv experiments/ensemble_unlearning/fashion_mnist_class${CLASS}.csv
  ```
- Shell loop similar to:
  ```
  for CLASS in 0 1 2 3 4 5 6 7 8 9; do
    python -m scripts.run_pickbest_unlearning ... --target_class ${CLASS} ...
  done
  ```

## Implementation hints
- Reuse loading from `cnn_surgery.utils.load_dataset.load_multi_stage_dataset`.
- Per-sample selection: run each update meta-net separately, evaluate via hold-o
ut mean, pick best.
- Metrics: target_drop, nontarget_drop (mean of other classes), distance, steps,
 winning_net_id.
- Keep update/eval sets disjoint; do not backprop through hold-out.
- Save seeds for reproducibility.

## Open knobs
- Filter to samples with base target accuracy > threshold (e.g., >0.6) to avoid 
trivial cases.
- Try alternative losses (`boost_loss_factory(beta)`, `simple_loss`) and stop ru
les.
- Optional: track variance across hold-out nets (std) for robustness.
