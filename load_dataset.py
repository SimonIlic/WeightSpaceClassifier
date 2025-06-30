# loads and filters datasets of weights, metrics, and hyperparameters

import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.io import gfile

DATAFRAME_CONFIG_COLS = [
    'config.w_init',
    'config.activation',
    'config.learning_rate',
    'config.init_std',
    'config.l2reg',
    'config.train_fraction',
    'config.dropout']
CATEGORICAL_CONFIG_PARAMS = ['config.w_init', 'config.activation']
CATEGORICAL_CONFIG_PARAMS_PREFIX = ['winit', 'act']
DATAFRAME_METRIC_COLS = [
    'test_accuracy',
    'test_loss',
    'train_accuracy',
    'train_loss']
TRAIN_SIZE = 15_000

# TODO: modify the following lines
CONFIGS_PATH_BASE = './'
MNIST_OUTDIR = "./mnist"
FMNIST_OUTDIR = './fashion_mnist'
CIFAR_OUTDIR = './cifar10'
SVHN_OUTDIR = './svhn_cropped'

def filter_checkpoints(weights, dataframe,
                       target='test_accuracy',
                       stage='final', binarize=True):
  """Take one checkpoint per run and do some pre-processing.

  Args:
    weights: numpy array of shape (num_runs, num_weights)
    dataframe: pandas DataFrame which has num_runs rows. First 4 columns should
      contain test_accuracy, test_loss, train_accuracy, train_loss respectively.
    target: string, what to use as an output
    stage: flag defining which checkpoint out of potentially many we will take
      for the run.
    binarize: Do we want to binarize the categorical hyperparams?

  Returns:
    tuple (weights_new, metrics, hyperparams, ckpts), where
      weights_new is a numpy array of shape (num_remaining_ckpts, num_weights),
      metrics is a numpy array of shape (num_remaining_ckpts, num_metrics) with
        num_metric being the length of DATAFRAME_METRIC_COLS,
      hyperparams is a pandas DataFrame of num_remaining_ckpts rows and columns
        listed in DATAFRAME_CONFIG_COLS.
      ckpts is an instance of pandas Index, keeping filenames of the checkpoints
    All the num_remaining_ckpts rows correspond to one checkpoint out of each
    run we had.
  """

  assert target in DATAFRAME_METRIC_COLS, 'unknown target'
  ids_to_take = []
  # Keep in mind that the rows of the DataFrame were sorted according to ckpt
  # Fetch the unit id corresponding to the ckpt of the first row
  current_uid = dataframe.axes[0][0].split('/')[-2]  # get the unit id
  steps = []
  for i in range(len(dataframe.axes[0])):
    # Fetch the new unit id
    ckpt = dataframe.axes[0][i]
    parts = ckpt.split('/')
    if parts[-2] == current_uid:
      steps.append(int(parts[-1].split('-')[-1]))
    else:
      # We need to process the previous unit
      # and choose which ckpt to take
      steps_sort = sorted(steps)
      target_step = -1
      if stage == 'final':
        target_step = steps_sort[-1]
      elif stage == 'early':
        target_step = steps_sort[0]
      else:  # middle
        target_step = steps_sort[int(len(steps) / 2)]
      offset = [j for (j, el) in enumerate(steps) if el == target_step][0]
      # Take the DataFrame row with the corresponding row id
      ids_to_take.append(i - len(steps) + offset)
      current_uid = parts[-2]
      steps = [int(parts[-1].split('-')[-1])]

  # Fetch the hyperparameters of the corresponding checkpoints
  hyperparams = dataframe[DATAFRAME_CONFIG_COLS]
  hyperparams = hyperparams.iloc[ids_to_take]
  if binarize:
    # Binarize categorical features
    hyperparams = pd.get_dummies(
        hyperparams,
        columns=CATEGORICAL_CONFIG_PARAMS,
        prefix=CATEGORICAL_CONFIG_PARAMS_PREFIX)
  else:
    # Make the categorical features have pandas type "category"
    # Then LGBM can use those as categorical
    hyperparams.is_copy = False
    for col in CATEGORICAL_CONFIG_PARAMS:
      hyperparams[col] = hyperparams[col].astype('category')

  # Fetch the file paths of the corresponding checkpoints
  ckpts = dataframe.axes[0][ids_to_take]

  return (weights[ids_to_take, :],
          dataframe[DATAFRAME_METRIC_COLS].values[ids_to_take, :].astype(
              np.float32),
          hyperparams,
          ckpts)

def load_dataset(dataset, train_size=TRAIN_SIZE, stage='final', binarize=False):
    """Load weight, metric, and config data for a single dataset.

    Args:
        dataset (str): One of {'mnist', 'fashion_mnist', 'cifar10', 'svhn_cropped'}.
        train_size (int): Number of samples for the training split.
        stage (str): Which checkpoint stage to select when calling
            `filter_checkpoints` ('final', 'early', or 'middle').
        binarize (bool): Whether to binarize categorical hyperparameters in
            `filter_checkpoints`.

    Returns:
        Tuple (weights_train, weights_test,
               outputs_train, outputs_test,
               configs_train, configs_test)
    """
    # Map dataset names to their corresponding output directory
    outdir_map = {
        'mnist': MNIST_OUTDIR,
        'fashion_mnist': FMNIST_OUTDIR,
        'cifar10': CIFAR_OUTDIR,
        'svhn_cropped': SVHN_OUTDIR,
    }
    if dataset not in outdir_map:
        raise ValueError(f"Unknown dataset: {dataset}")

    dirname = outdir_map[dataset]

    # Load the raw weights and metrics
    weights_path = os.path.join(dirname, "weights.npy")
    metrics_path = os.path.join(dirname, "metrics.csv")

    weights = np.load(weights_path, mmap_mode='r')
    with gfile.GFile(metrics_path) as f:
        metrics_df = pd.read_csv(f, index_col=0)

    # Select one checkpoint per run and preprocess
    weights_flt, metrics_flt, configs_flt, ckpts = filter_checkpoints(
        weights, metrics_df, binarize=binarize, stage=stage
    )

    # Filter out DNNs with NaNs/Infs in their weights
    idx_valid = np.isfinite(weights_flt).mean(1) == 1.0
    inputs = np.asarray(weights_flt[idx_valid], dtype=np.float32)
    outputs = np.asarray(metrics_flt[idx_valid], dtype=np.float32)
    configs = configs_flt.iloc[idx_valid]

    # Shuffle and split into train/test
    random_idx = np.arange(inputs.shape[0])
    np.random.shuffle(random_idx)

    weights_train = inputs[random_idx[:train_size]]
    weights_test = inputs[random_idx[train_size:]]
    outputs_train = outputs[random_idx[:train_size]]
    outputs_test = outputs[random_idx[train_size:]]
    configs_train = configs.iloc[random_idx[:train_size]]
    configs_test = configs.iloc[random_idx[train_size:]]

    return (
        weights_train,
        weights_test,
        outputs_train,
        outputs_test,
        configs_train,
        configs_test,
    )

# Example usage:
if __name__ == "__main__":
  # load mnist
  import time
  start_time = time.time()
  weights_train, weights_test, outputs_train, outputs_test, configs_train, configs_test = load_dataset('mnist')
  end_time = time.time()
  print(f"Loaded MNIST dataset in {end_time - start_time:.2f} seconds")
