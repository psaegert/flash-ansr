import shutil
import os

from huggingface_hub import snapshot_download

from flash_ansr.utils import get_path


def install_model(model: str, local_dir: str | None = None) -> None:
    print(f"Installing model {model} to {get_path('models', model)}")
    snapshot_download(repo_id=model, repo_type="model", local_dir=local_dir or get_path('models', model))
    print(f"Model {model} installed successfully!")


def remove_model(path: str) -> None:
    path_in_package = get_path('models', path)

    if os.path.exists(path_in_package) and os.path.exists(path):
        raise ValueError(f"Both {path_in_package} and {path} exist. Please remove one of them manually before running this command.")

    for path_to_delete in [path_in_package, path]:
        if os.path.exists(path_to_delete):
            confirm = input(f"Are you sure you want to remove {path_to_delete}? (y/N): ")
            if confirm.lower() != 'y':
                print("Aborting model removal.")
                return

            shutil.rmtree(path_to_delete)
            print(f"Model {path_to_delete} removed successfully!")
            return

    print(f"Model {path} not found.")
