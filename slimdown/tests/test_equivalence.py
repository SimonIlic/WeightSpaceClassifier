"""Equivalence checks: slimdown (batched) vs the original cnn_surgery code.

Run from the main repo root (needs cnn_surgery + model_zoo + a converted
meta-network):

    uv run --with torchvision python <worktree>/slimdown/tests/test_equivalence.py

Checks:
1. Zoo loading: slimdown's canonical split matches cnn_surgery.load_dataset.
2. Test images: torchvision-loaded data matches the keras/tfds-loaded data.
3. unlearn_batch matches the original sequential unlearn() per model
   (steps, final weights, predictions, loss, distance) for acc_pred,
   cosine_similarity, and step stopping criteria.
4. evaluate_batch matches the original TF evaluation within 1e-3.
"""

import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SLIMDOWN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SLIMDOWN_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

N_MODELS = 8
DATASET = "mnist"
TARGET_CLASS = 5

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name} {detail}")
    else:
        failed += 1
        print(f"  [FAIL] {name} {detail}")


def test_zoo_split():
    from cnn_surgery.utils.load_dataset import load_dataset as orig_load
    from slimdown.data import load_zoo

    orig_train, orig_test, orig_val = orig_load(dataset=DATASET, metrics_file="metrics_merged_final.csv", load_class_acc=True)
    new_train, new_test, new_val = load_zoo(DATASET)

    for name, o, n in [("train", orig_train, new_train), ("test", orig_test, new_test), ("val", orig_val, new_val)]:
        check(f"zoo/{name}_weights", np.array_equal(o[0], n[0]), f"shape {n[0].shape}")
        check(f"zoo/{name}_metrics", np.array_equal(o[1], n[1]))
        check(f"zoo/{name}_activations", (o[2]["config.activation"].values == n[2]["config.activation"].values).all())
    return new_val


def test_test_images():
    from cnn_surgery.utils.evaluate_per_class_accuracy import load_testset_data as orig_load
    from slimdown.data import load_testset_data

    x_orig, y_orig = orig_load(DATASET)  # (N, H, W, 1) NHWC
    x_new, y_new = load_testset_data(DATASET)  # (N, 1, H, W) NCHW tensors

    x_orig_nchw = np.transpose(x_orig, (0, 3, 1, 2))
    check("images/x_equal", np.allclose(x_orig_nchw, x_new.numpy(), atol=1e-6))
    check("images/y_equal", np.array_equal(np.asarray(y_orig).flatten(), y_new.numpy()))
    return x_new, y_new


def load_metanetworks():
    """Original pickled meta-network + slimdown-converted counterpart."""
    from cnn_surgery.evaluate_models import load_meta_network as orig_load
    from slimdown.nets import load_meta_network

    pkl_path = f"metanetworks/meta_network_{DATASET}_{TARGET_CLASS}.pkl"
    pt_path = f"metanetworks/converted/meta_network_{DATASET}_{TARGET_CLASS}.pt"
    orig_meta = orig_load(pkl_path, input_dim=4970, n_outputs=10, device="cpu")
    new_meta = load_meta_network(pt_path, device="cpu")

    w = torch.randn(3, 4970)
    with torch.no_grad():
        diff = (orig_meta(w) - new_meta(w)).abs().max().item()
    check("meta/converted_forward_equal", diff < 1e-7, f"(max_diff={diff:.2e})")
    return orig_meta, new_meta


def run_original_unlearn(weights_batch, meta, loss_fn, stop, max_steps, lr, l2):
    from cnn_surgery.unlearning import unlearn

    states = []
    for i in range(weights_batch.shape[0]):
        states.append(
            unlearn(
                weights_batch[i],
                meta,
                TARGET_CLASS,
                max_steps=max_steps,
                lr=lr,
                l2_penalty=l2,
                loss_fn=loss_fn,
                stopping_criterium=stop,
                device="cpu",
            )
        )
    return states


def test_unlearn_equivalence(weights_batch, orig_meta, new_meta):
    import cnn_surgery.unlearning as orig_ul
    import slimdown.unlearn as new_ul

    cases = {
        "acc_pred": (
            orig_ul.simple_loss,
            orig_ul.acc_pred_stop_factory(0.1),
            new_ul.simple_loss,
            new_ul.acc_pred_stop_factory(0.1),
        ),
        "cosine_similarity": (
            orig_ul.simple_loss,
            orig_ul.cosine_similarity_stop_factory(derivative=False, eps=1 - 0.9),
            new_ul.simple_loss,
            new_ul.cosine_similarity_stop_factory(derivative=False, eps=1 - 0.9),
        ),
        "step": (
            orig_ul.boost_loss_factory(0.1),
            orig_ul.step_stop_factory(50),
            new_ul.boost_loss_factory(0.1),
            new_ul.step_stop_factory(50),
        ),
    }
    max_steps = {"acc_pred": 300, "cosine_similarity": 300, "step": 50}

    for case, (o_loss, o_stop, n_loss, n_stop) in cases.items():
        ms = max_steps[case]
        orig_states = run_original_unlearn(weights_batch, orig_meta, o_loss, o_stop, ms, lr=0.1, l2=1e-6)
        batch_state = new_ul.unlearn_batch(
            weights_batch,
            new_meta,
            TARGET_CLASS,
            max_steps=ms,
            lr=0.1,
            l2_penalty=1e-6,
            loss_fn=n_loss,
            stopping_criterium=n_stop,
            device="cpu",
        )

        steps_ok = all(orig_states[i].step == int(batch_state.steps[i]) for i in range(len(orig_states)))
        check(f"unlearn/{case}/steps", steps_ok, f"orig={[s.step for s in orig_states]} new={batch_state.steps.tolist()}")

        # Full-batch trajectories accumulate float32 rounding from batched BLAS
        # kernels (~1e-5 over 50+ steps); B=1 bit-exactness is checked separately.
        w_diff = max(
            (orig_states[i].weights.squeeze() - batch_state.weights[i]).abs().max().item() for i in range(len(orig_states))
        )
        check(f"unlearn/{case}/weights", w_diff < 1e-4, f"(max_diff={w_diff:.2e})")

        p_diff = max((orig_states[i].pred - batch_state.pred[i]).abs().max().item() for i in range(len(orig_states)))
        check(f"unlearn/{case}/pred", p_diff < 1e-5, f"(max_diff={p_diff:.2e})")

        l_diff = max(abs(orig_states[i].loss - float(batch_state.loss[i])) for i in range(len(orig_states)))
        check(f"unlearn/{case}/loss", l_diff < 1e-5, f"(max_diff={l_diff:.2e})")

        d_diff = max(
            abs(orig_states[i].distance_travelled - float(batch_state.distance_travelled[i])) for i in range(len(orig_states))
        )
        check(f"unlearn/{case}/distance", d_diff < 1e-4, f"(max_diff={d_diff:.2e})")

        ip_diff = max((orig_states[i].init_pred - batch_state.init_pred[i]).abs().max().item() for i in range(len(orig_states)))
        check(f"unlearn/{case}/init_pred", ip_diff < 1e-6, f"(max_diff={ip_diff:.2e})")


def test_unlearn_bitexact_b1(weights_batch, orig_meta, new_meta):
    """With batch size 1 the batched loop must reproduce the original bit-exactly."""
    import cnn_surgery.unlearning as orig_ul
    import slimdown.unlearn as new_ul

    for loss_name, o_loss, n_loss in [
        ("simple", orig_ul.simple_loss, new_ul.simple_loss),
        ("boost", orig_ul.boost_loss_factory(0.1), new_ul.boost_loss_factory(0.1)),
    ]:
        max_diff = 0.0
        for i in range(weights_batch.shape[0]):
            orig_state = orig_ul.unlearn(
                weights_batch[i],
                orig_meta,
                TARGET_CLASS,
                max_steps=50,
                lr=0.1,
                l2_penalty=1e-6,
                loss_fn=o_loss,
                stopping_criterium=orig_ul.step_stop_factory(50),
                device="cpu",
            )
            b1 = new_ul.unlearn_batch(
                weights_batch[i : i + 1],
                new_meta,
                TARGET_CLASS,
                max_steps=50,
                lr=0.1,
                l2_penalty=1e-6,
                loss_fn=n_loss,
                stopping_criterium=new_ul.step_stop_factory(50),
                device="cpu",
            )
            max_diff = max(max_diff, (orig_state.weights.squeeze() - b1.weights[0]).abs().max().item())
        check(f"unlearn/bitexact_b1/{loss_name}", max_diff == 0.0, f"(max_diff={max_diff:.2e})")


def test_eval_equivalence(weights_batch, activations, x_new, y_new):
    from cnn_surgery.evaluate_models import evaluate_network
    from cnn_surgery.utils.evaluate_per_class_accuracy import load_testset_data as orig_load
    from slimdown.evaluate import evaluate_batch

    x_orig, y_orig = orig_load(DATASET)

    overall_new, per_class_new = evaluate_batch(weights_batch, activations, x_new, y_new)

    max_overall_diff = 0.0
    max_class_diff = 0.0
    for i in range(weights_batch.shape[0]):
        overall_tf, per_class_tf = evaluate_network(weights_batch[i], activations[i], x_orig, y_orig)
        max_overall_diff = max(max_overall_diff, abs(overall_tf - overall_new[i]))
        max_class_diff = max(max_class_diff, np.max(np.abs(np.array(per_class_tf) - per_class_new[i])))

    check("eval/overall_acc", max_overall_diff < 1e-3, f"(max_diff={max_overall_diff:.5f})")
    check("eval/per_class_acc", max_class_diff < 1e-3, f"(max_diff={max_class_diff:.5f})")


def main():
    print("=== 1. Zoo loading / canonical split ===")
    val_data = test_zoo_split()
    weights_val, _, config_val = val_data
    weights_batch = np.array(weights_val[:N_MODELS])
    activations = config_val["config.activation"].values[:N_MODELS]

    print("\n=== 2. Test images (torchvision vs keras) ===")
    x_new, y_new = test_test_images()

    print("\n=== 3. Meta-network conversion ===")
    orig_meta, new_meta = load_metanetworks()

    print("\n=== 4. Unlearning: batched vs sequential ===")
    test_unlearn_equivalence(weights_batch, orig_meta, new_meta)
    test_unlearn_bitexact_b1(weights_batch, orig_meta, new_meta)

    print("\n=== 5. Evaluation: batched vs TF ===")
    test_eval_equivalence(weights_batch, activations, x_new, y_new)

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
