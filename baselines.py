import keras
from cnn_surgery.utils.reconstruct_network import reconstruct_network

def finetune_ascent(weights, config, data, steps):
    """Baseline finetuning using gradient ascent on the forget task."""
    # Rebuild model with provided activation
    model = reconstruct_network(weights, activation=config["activation"])
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Configure optimizer from dict settings
    optimizer = keras.optimizers.get(config["optimizer"])
    optimizer.learning_rate = config["learning_rate"]
    # Compile with negated loss for ascent
    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: -loss(y_true, y_pred),  # gradient ASCENT
        metrics=["accuracy"],
    )
    # force the model to set input shapes and init weights
    for x, _ in data:
        model.predict(x)
        model.summary()
        break
        
    # Single epoch training with fixed steps
    model.fit(data, epochs=1, steps_per_epoch=steps, verbose=True)


def random_vector():
    """Baseline generate a random vector where each layer has the same magnitude as the corresponding layer of the proposed edit. From Ilharco et al."""
    pass


if __name__ == "__main__":
    from cnn_surgery.utils.train_network import get_dataset
    #NOTE: Unterthiner does not mention batch size, using default from their codebase
    dataset = get_dataset("mnist", batchsize=512)

    import numpy as np
    from math import prod
    from cnn_surgery.utils.reconstruct_network import SHAPES
    example_weights = np.array([0.1] * sum(prod(shape) for shape in SHAPES.values()))

    data_tr, data_te, dataset_info = dataset
    finetune_ascent(example_weights, config={'activation': 'relu', 'optimizer': 'adam', 'learning_rate': 0.001}, data=data_tr, steps=100)
