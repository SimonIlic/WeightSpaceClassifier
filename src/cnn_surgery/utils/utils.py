from pathlib import Path


def find_project_root():
    """Find the project root directory by looking for characteristic files/directories.

    This function searches upward from the current file's location to find the
    project root, identified by the presence of 'model_zoo' directory and
    'pyproject.toml' file.

    Returns:
        Path: The project root directory path.

    Raises:
        FileNotFoundError: If the project root cannot be found.
    """
    current_path = Path(__file__).resolve()

    # Start from the directory containing this file and search upward
    for parent in [current_path] + list(current_path.parents):
        # Check for characteristic project files/directories
        if (parent / "model_zoo").is_dir() and (parent / "pyproject.toml").is_file():
            return parent

    # Fallback: if we can't find the project root, raise an error
    raise FileNotFoundError(
        "Could not find project root. Make sure you're running from within the "
        "WeightSpaceClassifier project and that 'model_zoo' directory and "
        "'pyproject.toml' file exist in the root."
    )
