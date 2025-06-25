# runs relevant imports and defines some constants
# also loads data as dicts
# might take some time to run

from __future__ import division

import time
import os
import json
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
from scipy import stats
from tensorflow import keras
from tensorflow.io import gfile
import lightgbm as lgb

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
TRAIN_SIZE = 15000

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

def build_fcn(n_layers, n_hidden, n_outputs, dropout_rate, activation,
              w_regularizer, w_init, b_init, last_activation='softmax'):
  """Fully connected deep neural network."""
  model = keras.Sequential()
  model.add(keras.layers.Flatten())
  for _ in range(n_layers):
    model.add(
        keras.layers.Dense(
            n_hidden,
            activation=activation,
            kernel_regularizer=w_regularizer,
            kernel_initializer=w_init,
            bias_initializer=b_init))
    if dropout_rate > 0.0:
      model.add(keras.layers.Dropout(dropout_rate))
  if n_layers > 0:
    model.add(keras.layers.Dense(n_outputs, activation=last_activation))
  else:
    model.add(keras.layers.Dense(
        n_outputs,
        activation='sigmoid',
        kernel_regularizer=w_regularizer,
        kernel_initializer=w_init,
        bias_initializer=b_init))
  return model

def extract_summary_features(w, qts=(0, 25, 50, 75, 100)):
  """Extract various statistics from the flat vector w."""
  features = np.percentile(w, qts)
  features = np.append(features, [np.std(w), np.mean(w)])
  return features


def extract_per_layer_features(w, qts=None, layers=(0, 1, 2, 3)):
  """Extract per-layer statistics from the weight vector and concatenate."""
  # Indices of the location of biases/kernels in the flattened vector
  all_boundaries = {
      0: [(0, 16), (16, 160)], 
      1: [(160, 176), (176, 2480)], 
      2: [(2480, 2496), (2496, 4800)], 
      3: [(4800, 4810), (4810, 4970)]}
  boundaries = []
  for layer in layers:
    boundaries += all_boundaries[layer]
  
  if not qts:
    features = [extract_summary_features(w[a:b]) for (a, b) in boundaries]
  else:
    features = [extract_summary_features(w[a:b], qts) for (a, b) in boundaries]
  all_features = np.concatenate(features)
  return all_features

all_dirs = [MNIST_OUTDIR, FMNIST_OUTDIR, CIFAR_OUTDIR, SVHN_OUTDIR]
weights = {'mnist': None,
            'fashion_mnist': None,
            'cifar10': None,
            'svhn_cropped': None}
metrics = {'mnist': None,
            'fashion_mnist': None,
            'cifar10': None,
            'svhn_cropped': None}
for (dirname, dataname) in zip(
    all_dirs, ['mnist', 'fashion_mnist', 'cifar10', 'svhn_cropped']):
  print('Loading %s' % dataname)
  with gfile.GFile(os.path.join(dirname, "weights.npy"), "rb") as f:
    # Weights of the trained models
    weights[dataname] = np.load(f)
  with gfile.GFile(os.path.join(dirname, "metrics.csv")) as f:
    # pandas DataFrame with metrics
    metrics[dataname] = pd.read_csv(f, index_col=0)

weights_train = {}
weights_test = {}
configs_train = {}
configs_test = {}
outputs_train = {}
outputs_test = {}

for dataset in ['mnist', 'fashion_mnist', 'cifar10', 'svhn_cropped']:
  # Take one checkpoint per each run
  # If using GBM as predictor, set binarize=False
  weights_flt, metrics_flt, configs_flt, ckpts = filter_checkpoints(
      weights[dataset], metrics[dataset], binarize=False)

  # Filter out DNNs with NaNs and Inf in the weights
  idx_valid = (np.isfinite(weights_flt).mean(1) == 1.0)
  inputs = np.asarray(weights_flt[idx_valid], dtype=np.float32)
  outputs = np.asarray(metrics_flt[idx_valid], dtype=np.float32)
  configs = configs_flt.iloc[idx_valid]
  ckpts = ckpts[idx_valid]

  # Shuffle and split the data
  random_idx = list(range(inputs.shape[0]))
  np.random.shuffle(random_idx)
  weights_train[dataset], weights_test[dataset] = (
      inputs[random_idx[:TRAIN_SIZE]], inputs[random_idx[TRAIN_SIZE:]])
  outputs_train[dataset], outputs_test[dataset] = (
      1. * outputs[random_idx[:TRAIN_SIZE]],
      1. * outputs[random_idx[TRAIN_SIZE:]])
  configs_train[dataset], configs_test[dataset] = (
      configs.iloc[random_idx[:TRAIN_SIZE]], 
      configs.iloc[random_idx[TRAIN_SIZE:]])
