
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
