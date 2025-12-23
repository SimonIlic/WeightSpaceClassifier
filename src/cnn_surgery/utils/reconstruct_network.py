from math import prod

import numpy as np

# typing imports
import tensorflow as tf

from cnn_surgery.utils.train_network import build_cnn

SHAPES = {
    "sequential/conv2d/bias:0": (16,),
    "sequential/conv2d/kernel:0": (3, 3, 1, 16),
    "sequential/conv2d_1/bias:0": (16,),
    "sequential/conv2d_1/kernel:0": (3, 3, 16, 16),
    "sequential/conv2d_2/bias:0": (16,),
    "sequential/conv2d_2/kernel:0": (3, 3, 16, 16),
    "sequential/dense/bias:0": (10,),
    "sequential/dense/kernel:0": (16, 10),
}


def reconstruct_network(weights: np.ndarray, activation: str, l2_penalty=0.0, dropout_rate=0.0) -> tf.keras.Model:  # type: ignore  # type: ignore
    """
    Reconstruct a CNN model from the paper with the given weights and activation function.
    Args:
        weights (list or np.ndarray): The flat list of weights to set in the model.
        activation (str): The activation function to use in the model.
    Returns:
        tf.keras.Model: The reconstructed CNN model. (needs to be compiled before use)
    """
    w_reg = tf.keras.regularizers.l2(l2_penalty) if l2_penalty > 0.0 else None
    model = build_cnn(
        n_layers=3,
        n_hidden=16,
        n_outputs=10,
        dropout_rate=dropout_rate,
        activation=activation,
        w_regularizer=w_reg,
        w_init="glorot_uniform",
        b_init="zeros",
        stride=2,
        use_batchnorm=False,
    )
    model.build(input_shape=(None, 28, 28, 1))

    weights_reshaped_reshaped = reshape_weights(weights, SHAPES)
    model.set_weights(weights_reshaped_reshaped)
    return model


def reshape_weights(weights: np.ndarray, shapes: dict) -> list:
    """
    Reshape a flat list of weights into the specified shapes.

    Args:
        weights (list or np.ndarray): The flat list of weights.
        shapes (dict): An ordered dictionary of layer names and their corresponding shapes.

    Returns:
        list: A list of reshaped weights in the correct order.
    """
    reshaped_weights = []
    i = 0
    for shape in shapes.values():
        length = prod(shape)
        layer = weights[i : i + length].reshape(shape)
        i += length
        if len(layer.shape) > 1:  # is a weight layer
            reshaped_weights.insert(-1, layer)
        else:  # is a bias layer
            reshaped_weights.append(layer)
    return reshaped_weights


# Example usage:
if __name__ == "__main__":
    import numpy as np

    # Example weights and activation function
    example_weights = np.array([0.1] * sum(prod(shape) for shape in SHAPES.values()))
    example_activation = "relu"

    # Reconstruct the model
    model = reconstruct_network(example_weights, example_activation, l2_penalty=0.01)

    # Print model summary
    model.summary()
    # Note: You need to compile the model before using it for training or evaluation.
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.fit(...)  # Example training code
