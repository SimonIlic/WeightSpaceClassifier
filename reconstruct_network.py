from math import prod

import notebook
from train_network import build_cnn
import tensorflow as tf
from tensorflow.keras.datasets import mnist


SHAPES = {
    'sequential/conv2d/bias:0': (16,),
    'sequential/conv2d/kernel:0': (3, 3, 1, 16),
    'sequential/conv2d_1/bias:0': (16,),
    'sequential/conv2d_1/kernel:0': (3, 3, 16, 16),
    'sequential/conv2d_2/bias:0': (16,),
    'sequential/conv2d_2/kernel:0': (3, 3, 16, 16),
    'sequential/dense/bias:0': (10,),
    'sequential/dense/kernel:0': (16, 10),
}

# RUN reconstrion and test for the first 10 models
for model_index in range(10):
    weights = notebook.weights_train['mnist'][model_index]
    activation = str(notebook.configs_train['mnist']['config.activation'][model_index])

    model = build_cnn(
        n_layers=3,
        n_hidden=16,
        n_outputs=10,
        dropout_rate=0.0,
        activation=activation,
        w_regularizer=None,
        w_init='glorot_uniform',
        b_init='zeros',
        stride=2,
        use_batchnorm=False,
    )
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    # rebuild model with correct shapes
    i=0
    reshaped_weights = []
    for shape in SHAPES.values():
        length = prod(shape)
        layer = weights[i:i+length].reshape(shape)
        i += length

        # reorder bias after weights
        if len(layer.shape) > 1:  # is a weight layer
            reshaped_weights.insert(-1, layer)
        else:  # is a bias layer
            reshaped_weights.append(layer)

    model.set_weights(reshaped_weights)

    # Compile the model before evaluation
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Load and preprocess MNIST test set
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32") / 127.5 - 1.0  # rescale data between -1 and 1
    x_test = x_test[..., None]  # Add channel dimension

    # Evaluate model accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    # target test accuracy
    print("Target test accuracy:", notebook.outputs_train['mnist'][model_index][0])
