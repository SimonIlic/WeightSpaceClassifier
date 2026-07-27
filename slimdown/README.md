# slimdown

Slim, TensorFlow-free, **batched** reimplementation of the unlearning core
(`cnn_surgery.evaluate_models` + `cnn_surgery.unlearning`). Models are
unlearned and evaluated in batches instead of one-by-one.

**Scope**: unlearning + evaluation only. Baselines (random vector, finetune
ascent/retain) and JS similarity are intentionally out of scope for now.

**Dependencies**: `torch`, `numpy`, `pandas`, `torchvision`, `tqdm`. No
TensorFlow, Keras, tfds, scipy, or sklearn.

## Why it's fast

- **Batched unlearning.** The meta-network is a row-wise MLP, so a batch of B
  weight vectors is optimized simultaneously: one forward `(B, 4970) → (B, 10)`,
  one backward for the summed per-model losses. Per-model gradients are exactly
  the same as in the sequential loop. Per-model stopping is handled by removing
  finished rows from the batch, so late steps only pay for models still active.
- **Batched evaluation.** The Small CNN forward is vectorized across models
  (grouped by activation) with batched matmuls: the first conv shares one
  im2col across all models, and the deeper convs use shift-and-matmul (9
  strided slices + batched einsum), so B edited CNNs are evaluated on the full
  test set in one pass instead of B Keras build/compile/predict cycles.

## Usage

One-time setup — convert the legacy pickled meta-networks (needs the old
`cnn_surgery` env because unpickling imports it):

```bash
cd ~/Documents/WSL/WeightSpaceClassifier
PYTHONPATH=src uv run python slimdown/convert_metanetworks.py
# writes metanetworks/converted/*.pt
```

Run the experiment (same CLI as `cnn_surgery.evaluate_models`, plus
`--batch-size`, `--device`, `--zoo-dir`):

```bash
uv run --with torchvision python -m slimdown.run \
    -d mnist -c 5 \
    --max-steps 10000 --lr 0.1 \
    --loss-fn simple \
    --stopping-criterium acc_pred --stop-threshold 0.1 \
    --meta-network-path metanetworks/converted/meta_network_mnist_5.pt
```

Device is auto-detected (`cuda` > `mps` > `cpu`); batch size defaults to 256 on
CUDA and 64 otherwise. The output CSV uses the same column names as the
original script for all unlearning-core columns (baseline columns are absent).

## Equivalence with the original

`tests/test_equivalence.py` (34 checks, all passing) verifies against the old
codebase:

- Zoo loading reproduces the canonical seed-123 train/test/val split **exactly**.
- torchvision test images equal the keras-loaded images exactly.
- Converted meta-networks produce identical outputs to the pickled originals.
- `unlearn_batch` with batch size 1 is **bit-exact** vs the original `unlearn()`.
- Full batches match the original per model: identical stopping steps; weights,
  predictions, losses within ~1e-5 (float32 rounding from batched BLAS kernels
  accumulating over hundreds of steps — same class of noise as changing
  hardware).
- Batched evaluation matches the TF evaluation (identical accuracies on the
  tested models).

End-to-end (16 mnist models, class 5, acc_pred < 0.1): identical `steps` on all
models; per-class accuracies within 0.3% (a handful of borderline test images
after ~300 gradient steps).

Run it from the main repo root:

```bash
PYTHONPATH=src uv run --with torchvision python <worktree>/slimdown/tests/test_equivalence.py
```

## Benchmark

`tests/benchmark.py`, 64 mnist models, `max_steps=300`, `acc_pred < 0.1`
(Apple M-series laptop):

| implementation | unlearn (ms/model) | eval (ms/model) |
|---|---|---|
| original (sequential, CPU) | 50.8 | 75.6 |
| slimdown (batch 64, CPU) | **2.9** | 71.0 |
| slimdown (batch 64, MPS) | 19.5 | **49.6** |

Unlearning — the part that scales with `max_steps` — is **~17× faster** on CPU;
evaluation is at parity with TF on CPU and faster on MPS. For reference, the
full original loop (`evaluate_models.py`, which also computes baselines) ran at
~1.8 s/model on the same machine vs ~0.1 s/model for slimdown's full loop.
Gains grow with batch size on GPU (Snellius A100/H100: use `--batch-size 256+`).

### Bigger meta-networks (SANE)

Any `nn.Module` mapping `(B, 4970) -> (B, 10)` works as a drop-in meta-network.
The 202M-param SANE transformer wrapper
(`~/Documents/WSL/SANE/model_export/meta_network.pkl`, loaded in `jointenv`
with `PYTHONPATH=~/Documents/WSL/SANE`) with the same protocol as above:

| implementation | unlearn (ms/model) |
|---|---|
| original (sequential, CPU) | 2472.6 |
| slimdown (batch 64, CPU) | 1091.6 (2.3×) |
| slimdown (batch 64, MPS) | **699.8 (3.5×)** |

The gain is smaller than for the FCN because the SANE encoder is compute-bound
(~99% of step time is its forward+backward; batching saturates at B~32).
Batch compaction matters here: models stop at very different steps (median 43,
max 299), and finished models stop consuming encoder compute.

## Files

- `nets.py` — SmallCNN functional forward, flat-weight parsing (verified TF
  layout), FCN meta-network + loader
- `data.py` — zoo loading with canonical split; test images via torchvision
- `unlearn.py` — batched unlearning, vectorized losses and stopping criteria
- `evaluate.py` — vmapped batched per-class accuracy
- `run.py` — CLI / experiment loop
- `convert_metanetworks.py` — one-time `.pkl` → `.pt` conversion
- `tests/test_equivalence.py`, `tests/benchmark.py`
