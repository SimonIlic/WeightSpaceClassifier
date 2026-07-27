"""One-time conversion of legacy meta-network pickles to plain .pt files.

The legacy metanetworks/*.pkl files pickle the whole FCN object, which requires
importing cnn_surgery to load. This script (run inside the existing repo
environment) re-saves each as {"state_dict", "arch"} so slimdown can load them
with zero dependency on the old package.

Usage (from the repo root):
    uv run python slimdown/convert_metanetworks.py [--in-dir metanetworks] [--out-dir metanetworks/converted]
"""

import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn


def infer_arch(model: nn.Module) -> dict:
    """Read the FCN architecture off the module structure."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    hidden = linears[:-1]
    out = linears[-1]
    return dict(
        input_dim=hidden[0].in_features,
        n_layers=len(hidden),
        n_hidden=hidden[0].out_features,
        n_outputs=out.out_features,
        dropout_p=dropouts[0].p if dropouts else 0.0,
        last_activation=getattr(model, "last_activation", "sigmoid"),
    )


def convert(pkl_path: Path, out_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    arch = infer_arch(model)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "arch": arch}, out_path)
    return arch


def main():
    parser = argparse.ArgumentParser(description="Convert meta-network .pkl files to .pt state dicts.")
    parser.add_argument("--in-dir", type=str, default="metanetworks", help="Directory with .pkl meta-networks.")
    parser.add_argument("--out-dir", type=str, default="metanetworks/converted", help="Output directory for .pt files.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(in_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found in {in_dir}")

    for pkl_path in pkls:
        out_path = out_dir / (pkl_path.stem + ".pt")
        arch = convert(pkl_path, out_path)
        print(f"{pkl_path.name} -> {out_path}  {arch}")


if __name__ == "__main__":
    main()
