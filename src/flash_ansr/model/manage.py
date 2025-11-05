import shutil
import os

from huggingface_hub import snapshot_download

from flash_ansr.utils.paths import get_path


def install_model(model: str, local_dir: str | None = None, verbose: bool = True) -> None:
    if verbose:
        print(f"Installing model {model} to {get_path('models', model, create=True)}")
    snapshot_download(repo_id=model, repo_type="model", local_dir=local_dir or get_path('models', model))
    if verbose:
        print(f"Model {model} installed successfully!")


def remove_model(path: str, verbose: bool = True, force_remove: bool = False) -> None:
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
