# PyToch port of tensorflow DNN used by Unterthiner et al. adapted for multi-output regression.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# config – pulled from your best_configs entry
default_config = dict(
    optimizer_name = "Adam",
    learning_rate  = 1e-3,
    w_init_name    = "glorot_uniform",
    init_stddev    = 0.05,
    l2_penalty     = 1e-4,
    n_layers       = 6,
    n_hiddens      = 380,
    dropout_rate   = 0.20,
    batch_size     = 256,
)

# ---------------------------------------------------------------------
# 1. MLP architecture (identical to build_fcn from unterthiner, but last layer = sigmoid)
# ---------------------------------------------------------------------
class FCN(nn.Module):
    def __init__(self, input_dim, n_layers, n_hidden, n_outputs,
                 dropout_p, activation=nn.ReLU, last_activation="sigmoid"):
        super().__init__()
        self.flatten = nn.Flatten()
        blocks, in_f = [], input_dim
        for _ in range(n_layers):
            lin = nn.Linear(in_f, n_hidden)
            blocks += [lin, activation()]
            if dropout_p > 0:
                blocks.append(nn.Dropout(dropout_p))
            in_f = n_hidden # to make sure intermediate layers are (n_hidden, n_hidden)
        self.hidden = nn.Sequential(*blocks)
        self.out = nn.Linear(in_f, n_outputs)
        self.last_activation = last_activation

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.out(x)
        if self.last_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


# ---------------------------------------------------------------------
# 2. Utility: apply the same weight/bias initialisation as TF
# ---------------------------------------------------------------------
def apply_weight_init(module, w_init_name: str, std: float | None):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            name = w_init_name.lower()
            if name in ("glorot_uniform", "glorotuniform", "xavier_uniform"):
                nn.init.xavier_uniform_(m.weight)
            elif name in ("truncatednormal", "randomnormal"):
                sd = 0.05 if std is None else std
                nn.init.normal_(m.weight, mean=0.0, std=sd)
            else:                               # sensible default
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------
# 3. Metric helpers
# ---------------------------------------------------------------------
def mse_mae(model, loader, device="cpu"):
    model.eval()
    mse_tot, mae_tot, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            mse_tot += F.mse_loss(pred, yb, reduction="none").mean(dim=1).sum().item()
            mae_tot += F.l1_loss(pred, yb, reduction="none").mean(dim=1).sum().item()
            n += yb.size(0)
    return mse_tot / n, mae_tot / n


# ---------------------------------------------------------------------
# 4. Trainer ─ replicates fit() + EarlyStopping
# ---------------------------------------------------------------------
def train_torch_dnn(train_x, train_y, test_x, test_y, config):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- data ----------
    train_ds = TensorDataset(torch.as_tensor(train_x, dtype=torch.float32),
                             torch.as_tensor(train_y, dtype=torch.float32))
    test_ds  = TensorDataset(torch.as_tensor(test_x,  dtype=torch.float32),
                             torch.as_tensor(test_y,  dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader   = DataLoader(test_ds,  batch_size=int(config["batch_size"]), shuffle=False)

    # ---------- model ----------
    model = FCN(input_dim=train_x.shape[1],
                n_layers=int(config["n_layers"]),
                n_hidden=int(config["n_hiddens"]),
                n_outputs=train_y.shape[1],
                dropout_p=config["dropout_rate"],
                activation=nn.ReLU,
                last_activation="sigmoid")
    apply_weight_init(model, config["w_init_name"], config.get("init_stddev"))
    model.to(device)

    # ---------- optimiser (with L2) ----------
    opt_class = getattr(torch.optim, config["optimizer_name"])
    optimizer = opt_class(model.parameters(),
                          lr=config["learning_rate"],
                          weight_decay=config["l2_penalty"])

    # ---------- loss ----------
    criterion = nn.MSELoss()

    # ---------- early-stopping _________________
    patience, patience_left = 10, 10
    best_val = float("inf")

    for epoch in range(1, 301):  # epochs = 300
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # ---- validation
        val_mse, val_mae = mse_mae(model, val_loader, device)
        print(f"Epoch {epoch:3d} ─ val MSE {val_mse:.6f} | val MAE {val_mae:.6f}")

        # early-stopping logic (min_delta = 0)
        if val_mse < best_val:               # improvement
            best_val = val_mse
            patience_left = patience
        else:                                # no improvement
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopped after epoch {epoch}")
                break

    # ---------- final evaluation (batch_size = 128) ----------
    eval_loader_train = DataLoader(train_ds, batch_size=128, shuffle=False)
    eval_loader_test  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    mse_train, mae_train = mse_mae(model, eval_loader_train, device)
    mse_test,  mae_test  = mse_mae(model, eval_loader_test,  device)

    var = np.mean((test_y - np.mean(test_y)) ** 2.0)
    r2  = 1.0 - mse_test / var

    print("\n========== FINAL REPORT ==========")
    print(f"Test MSE = {mse_test:.6f}")
    print(f"Test MAD = {mae_test:.6f}")
    print(f"Test R2  = {r2:.6f}")

    return model


# ---------------------------------------------------------------------
# 5. Usage example (replace the dummy arrays with your real Tensors)
# ---------------------------------------------------------------------
# -----------------------------------------------------------------
# Dummy data so the script runs – substitute your own arrays here!
# -----------------------------------------------------------------

def get_regressor_lens(weights_train: np.ndarray, outputs_train: np.ndarray, weights_test: np.ndarray, outputs_test: np.ndarray, config: dict=default_config) -> torch.nn.Module:
    """
    Get a regressor lens for the given training and test weights and outputs.

    Trains a DNN (MLP) regressor model on the weights_train data, which may then be used for attribution analysis or other tasks.

    Args:
        weights_train (np.ndarray): Training weights.
        outputs_train (np.ndarray): Training outputs.
        weights_test (np.ndarray): Test weights.
        outputs_test (np.ndarray): Test outputs.
        config (dict): Configuration dictionary for the regressor MLP.

    Returns:
        torch.nn.Module: The trained regressor model.
    """
    return train_torch_dnn(weights_train, outputs_train, weights_test, outputs_test, config)
