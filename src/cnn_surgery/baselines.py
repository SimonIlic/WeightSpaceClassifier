import keras
from math import prod
import numpy as np
from cnn_surgery.utils.reconstruct_network import reconstruct_network, SHAPES
from cnn_surgery.utils.process_models import _flatten_weights_for_reconstruction

def finetune_ascent(weights, config, data, steps):
    """Baseline finetuning using gradient ascent on the forget task. As in Ilharco et al., Golatkar et al., Tarun et al."""
    model = reconstruct_network(weights, activation=config["config.activation"], l2_penalty=config['config.l2reg'])
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.get(config["config.optimizer"])
    optimizer.learning_rate = config["config.learning_rate"]
    # Compile with negated loss for gradient ascent
    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: -loss(y_true, y_pred),  # gradient ASCENT
        metrics=["accuracy"],
    )
        
    # Single epoch training with fixed steps
    model.fit(data, epochs=1, steps_per_epoch=steps, verbose=True)

    # convert model back to raw weight vector
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
    #NOTE: Unterthiner does not mention batch size, using default from their codebase
    dataset = get_dataset("mnist", batchsize=512)

    example_weights = np.array([0.1] * sum(prod(shape) for shape in SHAPES.values()))

    data_tr, data_te, dataset_info = dataset
    # filter data_tr to only include class 7
    data_tr = data_tr.unbatch().filter(lambda x, y: y == 7).batch(512)

    ft = finetune_ascent(example_weights,
                         config={'activation': 'relu', 'optimizer': 'adam', 'learning_rate': 0.001, 'l2_penalty': 0.01},
                         data=data_tr,
                         steps=100)

    rd = random_vector(example_weights, example_weights + 0.01)
