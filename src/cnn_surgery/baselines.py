import keras
from math import prod
import numpy as np
from cnn_surgery.utils.reconstruct_network import reconstruct_network, SHAPES
from cnn_surgery.utils.process_models import _flatten_weights_for_reconstruction


def finetune_ascent(weights, config, data, forget_class, steps, verbose=True) -> np.ndarray:
    """Baseline finetuning using gradient ascent on the forget task.

    As in Ilharco et al., Golatkar et al., Tarun et al.

    Args:
        weights: Flattened CNN weights (numpy array, shape 4970)
        config: Pandas Series with keys like config.activation, config.optimizer, etc.
        data: TensorFlow dataset (unfiltered - filtering happens inside)
        forget_class: Class to forget (int). Data will be filtered to ONLY this class.
        steps: Number of gradient ascent steps
        verbose: Whether to print training progress

    Returns:
        Flattened weights after gradient ascent on forget class
    """
    # Filter to only forget class
    forget_data = data.unbatch().filter(lambda x, y: y == forget_class).batch(512)

    model = reconstruct_network(
        weights, activation=config["config.activation"], l2_penalty=config["config.l2reg"], dropout_rate=config["config.dropout"]
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.get(config["config.optimizer"])
    optimizer.learning_rate = config["config.learning_rate"]  # type: ignore
    # Compile with negated loss for gradient ascent
    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: -loss(y_true, y_pred),  # gradient ASCENT # type: ignore
        metrics=["accuracy"],
    )

    # Single epoch training with fixed steps
    model.fit(forget_data, epochs=1, steps_per_epoch=steps, verbose=verbose)

    # convert model back to raw weight vector
    model_weights = model.get_weights()
    flat_weights = _flatten_weights_for_reconstruction(model_weights)
    return flat_weights


def finetune_retain(weights, config, data, forget_class, epochs=5, steps=None, verbose=True) -> np.ndarray:
    """Baseline finetuning on the retain set (standard supervised learning).

    As described in Golatkar et al. (2020) & Foster et al. (2024): Selective Synaptic Dampening (2024).
    Default 5 epochs follows SSD paper settings.

    Args:
        weights: Flattened CNN weights (numpy array, shape 4970)
        config: Pandas Series with keys like config.activation, config.optimizer, etc.
        data: TensorFlow dataset (unfiltered - filtering happens inside)
        forget_class: Class to forget (int). Data will be filtered to EXCLUDE this class.
        epochs: Number of finetuning epochs (default: 5, per SSD paper)
        steps: If provided, train for this many steps instead of epochs
        verbose: Whether to print training progress

    Returns:
        Flattened weights after finetuning
    """
    # Filter out forget class to create retain set
    retain_data = data.unbatch().filter(lambda x, y: y != forget_class).batch(512)

    model = reconstruct_network(
        weights, activation=config["config.activation"], l2_penalty=config["config.l2reg"], dropout_rate=config["config.dropout"]
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.get(config["config.optimizer"])
    optimizer.learning_rate = config["config.learning_rate"]  # type: ignore
    # compile for standard supervised learning
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    # Train for epochs or fixed steps (TODO: discuss what we want. I like the reasoning of only allowing as many steps as unlearning,
    # but literature does a few epochs)
    if steps is not None:
        model.fit(retain_data, epochs=1, steps_per_epoch=steps, verbose=verbose)
    else:
        model.fit(retain_data, epochs=epochs, verbose=verbose)

    # Convert model back to raw weight vector
    model_weights = model.get_weights()
    flat_weights = _flatten_weights_for_reconstruction(model_weights)
    return flat_weights


def random_vector(original_weights, edit_weights):
    """Baseline generate a random edit where each layer has the same magnitude as the corresponding layer of the proposed edit. As described in Ilharco et al."""
    i = 0
    random_weights = []
    delta_weights = edit_weights - original_weights
    for shape in SHAPES.values():
        length = prod(shape)
        j = i + length
        delta_layer = delta_weights[i:j]
        rand_layer = np.random.randn(length)
        random_weights.append(rand_layer * (np.linalg.norm(delta_layer) / np.linalg.norm(rand_layer)))
        i = j
    random_weights = np.concatenate(random_weights)
    return original_weights + random_weights


if __name__ == "__main__":
    from cnn_surgery.utils.train_network import get_dataset

    # NOTE: Unterthiner does not mention batch size, using default from their codebase
    dataset = get_dataset("mnist", batchsize=512)

    example_weights = np.array([0.1] * sum(prod(shape) for shape in SHAPES.values()))

    data_tr, data_te, dataset_info = dataset

    example_config = {
        "config.activation": "relu",
        "config.optimizer": "adam",
        "config.learning_rate": 0.001,
        "config.l2reg": 0.01,
        "config.dropout": 0.0,
    }

    # Gradient ascent on forget class (class 7)
    ft = finetune_ascent(example_weights, config=example_config, data=data_tr, forget_class=7, steps=100)

    # Finetune on retain set (all except class 7)
    fr = finetune_retain(example_weights, config=example_config, data=data_tr, forget_class=7, epochs=5)

    rd = random_vector(example_weights, example_weights + 0.01)
