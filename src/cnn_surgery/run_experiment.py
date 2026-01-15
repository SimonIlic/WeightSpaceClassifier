import os, sys, subprocess
from datetime import datetime

import tensorflow as tf
import numpy as np

from cnn_surgery.utils.train_network import run, get_dataset

rng = np.random.default_rng()


BATCH_SIZE = 512  # fixed to original default for all experiments
EPOCHS = 86  # fixed to original default for all experiments


def seed_everything(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_workdir(seed):
    current_time = datetime.now().strftime("%H_%M")
    workdir = f"./runs/{current_time}_{seed}/"
    os.makedirs(workdir, exist_ok=True)
    return workdir


def sample_log_uniform(low: float, high: float):
    """Samples log-uniformly between low and high"""
    return np.exp(rng.uniform(np.log(low), np.log(high)))


def sample_hyperparameters():
    """Samples hyperparameters for train_network.py as described in Appendix A.2 of Unterthiner et al. (2021)."""
    optimizer = rng.choice(["sgd", "adam", "rmsprop"])  # note this differs from defaults in train_network
    learning_rate = sample_log_uniform(5e-4, 5e-2)
    l2reg = sample_log_uniform(1e-8, 1e-2)
    dropout = rng.uniform(0, 0.7)
    init_var = sample_log_uniform(1e-3, 0.5)
    init_std = init_var**0.5  # spread of weight initializer is defined in variance in paper, but std in code
    w_init = rng.choice(["glorot_normal", "RandomNormal", "TruncatedNormal", "orthogonal", "he_normal"])
    b_init = "zero"
    activation_fn = rng.choice(["relu", "tanh"])
    train_fraction = rng.choice([0.1, 0.25, 0.5, 1.0])
    seed = rng.integers(0, 2**31)  # most likely a unique seed (p_coll ~ 2.3% for 10k samples)

    return {
        "optimizer_name": optimizer,
        "learning_rate": learning_rate,
        "l2_penalty": l2reg,
        "dropout_rate": dropout,
        "init_stddev": init_std,
        "w_init_name": w_init,
        "b_init_name": b_init,
        "activation": activation_fn,
        "train_fraction": train_fraction,
        "seed": seed,
    }


def main(argv):
    dataset, exclude_class = argv[1], int(argv[2])

    args = sample_hyperparameters()
    workdir = create_workdir(args["seed"])

    script = "./src/cnn_surgery/utils/train_network.py"
    cmd = [
        "python",
        script,
        f"--train_fraction={args['train_fraction']}",
        f"--epochs={EPOCHS}",
        f"--random_seed={args['seed']}",
        f"--dropout={args['dropout_rate']}",
        f"--l2reg={args['l2_penalty']}",
        f"--init_std={args['init_stddev']}",
        f"--learning_rate={args['learning_rate']}",
        f"--optimizer={args['optimizer_name']}",
        f"--activation={args['activation']}",
        f"--w_init={args['w_init_name']}",
        f"--b_init={args['b_init_name']}",
        f"--dataset={dataset}",
        f"--exclude_class={exclude_class}",
        f"--workdir={workdir}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(sys.argv)
