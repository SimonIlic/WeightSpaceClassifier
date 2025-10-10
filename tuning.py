import numpy as np
from cnn_surgery.utils.load_dataset import load_dataset
from cnn_surgery.lenses.regressor_lens import get_regressor_lens, default_config
import optuna

early = load_dataset('mnist', metrics_file='metrics_merged_mnist_early.csv', load_class_acc=True, stage='early')
middle = load_dataset('mnist', metrics_file='metrics_merged_mnist_middle.csv', load_class_acc=True, stage='middle')
final = load_dataset('mnist', metrics_file='metrics_merged.csv', load_class_acc=True, stage='final')

train_early, test_early, val_early = early
train_middle, test_middle, val_middle = middle
train_final, test_final, val_final = final

weights_train = np.concatenate([train_early[0], train_middle[0], train_final[0]])
weights_val = np.concatenate([val_early[0], val_middle[0], val_final[0]])

accuracies_train = np.concatenate([train_early[1][:, -10:], train_middle[1][:, -10:], train_final[1][:, -10:]])
accuracies_val = np.concatenate([val_early[1][:, -10:], val_middle[1][:, -10:], val_final[1][:, -10:]])

def objective(trial: optuna.trial.Trial) -> float:
    config = default_config.copy()
    config['batch_size'] = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
    config['learning_rate'] = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    config['n_hidden_layers'] = trial.suggest_int('n_hidden_layers', 1, 8)
    config['hidden_dim'] = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
    config['l2_penalty'] = trial.suggest_float('l2_penalty', 1e-6, 1e-2, log=True)
    config['dropout_rate'] = trial.suggest_float('dropout_rate', 0.0, 0.5)
    config['n_epochs'] = 15

    _, ((mse_train, mae_train), (mse_val, mae_val), r2) = get_regressor_lens(
        weights_train,
        accuracies_train,
        weights_val,
        accuracies_val,
        config=config,
        return_metrics=True,
        verbose=False
    )
    
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
