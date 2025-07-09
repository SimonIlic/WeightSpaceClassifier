import pandas as pd
import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist, mnist, cifar10
import tensorflow_datasets as tfds
from tqdm import tqdm
import os

from utils.reconstruct_network import reconstruct_network
from utils.load_dataset import load_dataset

from typing import Dict, List, Tuple

DATASET = 'mnist'

def evaluate_classifier(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    batch_size: int = 256,
    num_classes: int | None = None,
) -> Tuple[float, List[float]]:
    """
    Evaluate a compiled Keras/TensorFlow classifier.

    Parameters
    ----------
    model
        A *compiled* tf.keras.Model ready for inference.
    x_test
        Test features in the exact preprocessing/shape expected by `model`.
    y_test
        Integer class labels aligned with `x_test`.
    batch_size
        Batch size for inference (default 256).
    num_classes
        If provided, forces that many classes; otherwise it is
        inferred from `y_test.max() + 1`.

    Returns
    -------
    overall_acc : float
        Top-1 accuracy on the whole test set.
    per_class_acc : list[float]
        Accuracy for every class `0 … num_classes-1`.  Classes
        missing from `y_test` are assigned `np.nan`.
    """
    # sanity checks
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("x_test and y_test must have the same length")

    if num_classes is None:
        num_classes = int(np.max(y_test) + 1)

    # overall accuracy 
    _, overall_acc = model.evaluate(x_test, y_test,
                                    batch_size=batch_size,
                                    verbose=0)

    # per-class accuracy
    y_pred = np.argmax(model.predict(x_test, batch_size=batch_size, verbose=0),
                       axis=1)

    per_class_acc: list[float] = []
    for cls in range(num_classes):
        idx = (y_test == cls).flatten()
        if idx.any():
            per_class_acc.append(float(np.mean(y_pred[idx] == y_test[idx])))
        else:
            per_class_acc.append(np.nan)        # class absent in test set

    return overall_acc, per_class_acc

def load_testset_data(dataset: str):
    """
    Load the test set data for the specified dataset.

    Args:
        dataset (str): One of {'mnist', 'fashion_mnist', 'cifar10', 'svhn_cropped'}.

    Returns:
        Tuple: (x_test, y_test) where x_test is the test images and y_test is the test labels.
    """
    if dataset == 'fashion_mnist':
        data = fashion_mnist.load_data()
    elif dataset == 'mnist':
        data = mnist.load_data()
    elif dataset == 'cifar10':
        data = cifar10.load_data()
        # greyscale conversion for CIFAR-10
        x_test, y_test = data[1]
        x_test = np.mean(x_test, axis=-1, keepdims=False)  # convert to greyscale
        data = (data[0], (x_test, y_test))

    elif dataset == 'svhn_cropped':
        svhn_test = tfds.load('svhn_cropped', split='test', as_supervised=True, batch_size=-1) # load all data at once
        x_test, y_test = tfds.as_numpy(svhn_test)
        x_test = np.mean(x_test, axis=-1, keepdims=False)  # convert to greyscale
        data = ((None, None), (x_test, y_test))

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # unpack data
    (_, _), (x_test, y_test) = data
    x_test = x_test.astype("float32") / 127.5 - 1.0  # rescale data between -1 and 1
    x_test = x_test[..., None]  # Add channel dimension for CNNs

    return x_test, y_test


def main(dataset: str = DATASET):
    # if output csv file exists raise error
    if os.path.exists(f'{dataset}_model_results.csv'):
        raise FileExistsError(f"Results file {dataset}_model_results.csv already exists. Please remove it or choose a different dataset.")
    
    # load_dataset
    weights_train, weights_test, outputs_train, outputs_test, configs_train, configs_test = load_dataset(dataset)

    # combine train and test split
    weights = np.concatenate((weights_train, weights_test), axis=0)
    outputs = np.concatenate((outputs_train, outputs_test), axis=0)
    configs = pd.concat((configs_train, configs_test), axis=0)

    # load test set data
    x_test, y_test = load_testset_data(dataset)

    for model_index, (weights_row, output_row, config_row) in tqdm(enumerate(zip(weights, outputs, configs.itertuples())), total=len(weights), desc="Evaluating models"):
        # reconstruct the model
        model = reconstruct_network(weights_row, config_row[2])  # config_row[2] is the activation function
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # evaluate the model
        overall_acc, per_class_acc = evaluate_classifier(model, x_test, y_test)

        # store results
        model_name = config_row[0]
        results = {
            'model_name': model_name,
            'target_accuracy': output_row[0],  # assuming output_row[0] is the target accuracy
            'overall_accuracy': overall_acc,
            **{f'accuracy_class_{cls}': acc for cls, acc in enumerate(per_class_acc)}
        }
        
        # save results to a CSV file or any other format as needed
        results_df = pd.DataFrame([results])
        results_df.to_csv(f'{dataset}_model_results.csv', mode='a', header=model_index==0, index=False)

    print(f"All per-class results for {dataset} models saved to {dataset}_model_results.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate per-class accuracy of CNN models.")
    parser.add_argument('--dataset', type=str, default=DATASET, choices=['mnist', 'fashion_mnist', 'cifar10', 'svhn_cropped'],
                        help='Dataset to use for evaluation (default: mnist)')
    args = parser.parse_args()
    
    main(dataset=args.dataset)
