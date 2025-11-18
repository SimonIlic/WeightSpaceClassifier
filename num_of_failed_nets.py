from cnn_surgery.utils.load_dataset import load_dataset
import numpy as np
(weights_train, outputs_train, configs_train), (weights_test, outputs_test, configs_test), (weights_val, outputs_val, configs_val) = load_dataset('mnist', load_class_acc=True, metrics_file='metrics_merged_final.csv')
accuracies = outputs_val[:, -10:]
failed_nets = np.sum(np.any(accuracies <= 0.1, axis=1))
print(f"Number of failed networks: {failed_nets} out of {len(accuracies)}, fraction: {failed_nets / len(accuracies):.2f}")
