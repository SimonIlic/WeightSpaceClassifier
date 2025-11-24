
## Installation
1. clone repo 
2. install uv
3. from WeightSpaceClassifier directory:
```bash
uv sync
uv pip install -e .
```
4. Make sure to have the ruff extension installed in your code editor. (vscode: `charliemarsh.ruff`)

I recommend to configure the ruff extension to format your code on save.
Feel free to customize the ruff configuration in `pyproject.toml` to fit your coding style.

## Adding packages
Instead of `pip install <package_name>`:
```bash
uv add <package_name>
```

## Structure

Main files for doing the unlearning:
- `notebooks/unlearning.ipynb`: Jupyter notebook with the most up to date general unlearning experiments (mostly used by @moosmiddelkoop)
- `unlearning.py`: Main script for rigorous unlearning experiments (written by @simonilic)