# loads and filters datasets of weights, metrics, and hyperparameters

import os
import logging
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.io import gfile

SEED = 123  # Seed 123 will be the canonical seed for this project.
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
    'train_loss',
    ]
DATAFRAME_CLASS_ACCURACY_COLS = ['accuracy_class_' + str(i) for i in range(10)]
TRAIN_SIZE = 15_000

# TODO: modify the following lines
CONFIGS_PATH_BASE = './'
MNIST_OUTDIR = "./mnist"
FMNIST_OUTDIR = './fashion_mnist'
CIFAR_OUTDIR = './cifar10'
SVHN_OUTDIR = './svhn_cropped'

def filter_checkpoints(weights, dataframe,
                       stage='final', binarize=True, load_class_acc=False):
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
  if load_class_acc:
    return_cols = DATAFRAME_METRIC_COLS + DATAFRAME_CLASS_ACCURACY_COLS
  else:
    return_cols = DATAFRAME_METRIC_COLS

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
          dataframe[return_cols].values[ids_to_take, :].astype(
              np.float32),
          hyperparams,
          ckpts)

def load_dataset(dataset, train_size=TRAIN_SIZE, stage='final', binarize=False, metrics_file='metrics.csv', load_class_acc=False, shuffle=True, seed=SEED):
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
    metrics_path = os.path.join(dirname, metrics_file)

    weights = np.load(weights_path, mmap_mode='r')
    with gfile.GFile(metrics_path) as f:
        metrics_df = pd.read_csv(f, index_col=0)

    # Select one checkpoint per run and preprocess
    weights_flt, metrics_flt, configs_flt, ckpts = filter_checkpoints(
        weights, metrics_df, binarize=binarize, stage=stage, load_class_acc=load_class_acc
    )

    # Filter out DNNs with NaNs/Infs in their weights
    idx_valid = np.isfinite(weights_flt).mean(1) == 1.0
    inputs = np.asarray(weights_flt[idx_valid], dtype=np.float32)
    outputs = np.asarray(metrics_flt[idx_valid], dtype=np.float32)
    configs = configs_flt.iloc[idx_valid]

    # Shuffle and split into train/test
    random_idx = np.arange(inputs.shape[0])
    if shuffle:
      if seed == SEED:
        logging.warning(f"Using default seed {SEED} for shuffling. This results in the canonical train/test/val splits for this project. If a random split is desired, please use a different seed.")
      np.random.seed(seed)  # Set seed for reproducibility
      np.random.shuffle(random_idx)

    # compute the test size and validation size from what's left after the train split
    test_size = inputs.shape[0] - train_size
    val_size = test_size // 2
    test_size -= val_size  # Adjust test size to account for validation

    weights_train = inputs[random_idx[:train_size]]
    outputs_train = outputs[random_idx[:train_size]]
    configs_train = configs.iloc[random_idx[:train_size]]

    weights_test = inputs[random_idx[train_size:train_size + test_size]]
    outputs_test = outputs[random_idx[train_size:train_size + test_size]]
    configs_test = configs.iloc[random_idx[train_size:train_size + test_size]]

    weights_val = inputs[random_idx[train_size + test_size:]]
    outputs_val = outputs[random_idx[train_size + test_size:]]
    configs_val = configs.iloc[random_idx[train_size + test_size:]]

    return ((
        weights_train,
        weights_test,
        outputs_train,),
        (outputs_test,
        configs_train,
        configs_test,),
        (weights_val,
        outputs_val,
        configs_val)
    )

# Example usage:
if __name__ == "__main__":
  # load mnist
  import time
  start_time = time.time()
  train, test, val = load_dataset('mnist', metrics_file='metrics_merged.csv', load_class_acc=True)
  weights_train, weights_test, outputs_train = train
  outputs_test, configs_train, configs_test = test
  weights_val, outputs_val, configs_val = val
  end_time = time.time()
  print(f"Loaded MNIST dataset in {end_time - start_time:.2f} seconds")
