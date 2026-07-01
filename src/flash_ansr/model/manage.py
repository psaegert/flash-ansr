"""Install and remove pretrained model snapshots from the Hugging Face Hub."""
import shutil
import os

from huggingface_hub import snapshot_download

from flash_ansr.utils.paths import get_path


def install_model(model: str, local_dir: str | None = None, verbose: bool = True) -> None:
    """Download a model snapshot from the Hugging Face Hub.

    Parameters
    ----------
    model : str
        The Hugging Face Hub repo id of the model to install.
    local_dir : str, optional
        Directory to download the model into. Defaults to the package models
        directory for ``model`` (``get_path('models', model)``).
    verbose : bool, optional
        If True (default), print progress messages before and after the download.
    """
    if verbose:
        print(f"Installing model {model} to {get_path('models', model, create=True)}")
    snapshot_download(repo_id=model, repo_type="model", local_dir=local_dir or get_path('models', model))
    if verbose:
        print(f"Model {model} installed successfully!")


def remove_model(path: str, verbose: bool = True, force_remove: bool = False) -> None:
    """Remove an installed model directory.

    Looks for the model both in the package models directory
    (``get_path('models', path)``) and at ``path`` itself, and deletes the first
    one that exists via ``shutil.rmtree``. Unless ``force_remove`` is True, the
    user is interactively prompted to confirm the deletion.

    Parameters
    ----------
    path : str
        Name of the model in the package models directory, or a filesystem path
        to a model directory.
    verbose : bool, optional
        If True (default), print status messages (removal confirmation and the
        'not found' notice).
    force_remove : bool, optional
        If True, skip the interactive confirmation prompt and delete without
        asking. Defaults to False.

    Raises
    ------
    ValueError
        If the model exists both in the package models directory and at ``path``,
        in which case one must be removed manually first.
    """
    path_in_package = get_path('models', path)

    if os.path.exists(path_in_package) and os.path.exists(path):
        raise ValueError(f"Both {path_in_package} and {path} exist. Please remove one of them manually before running this command.")

    for path_to_delete in [path_in_package, path]:
        if os.path.exists(path_to_delete):
            if not force_remove:
                confirm = input(f"Are you sure you want to remove {path_to_delete}? (y/N): ")
                if confirm.lower() != 'y':
                    print("Aborting model removal.")
                    return
            print(f"Removing {path_to_delete}...")
            shutil.rmtree(path_to_delete)
            if verbose:
                print(f"Model {path_to_delete} removed successfully!")
            return

    if verbose:
        print(f"Model {path} not found.")
