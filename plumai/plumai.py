import os
import re
import json
import argparse
import subprocess
import webbrowser
import platform
from pathlib import Path
from colorama import Fore, init

from plumai.app import app

# Initialize colorama
init(autoreset=True)

# Determine the cache directory based on the OS
def get_cache_directory() -> Path:
    if platform.system() == "Windows":
        cache_dir = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming")) / "plumai"
    elif platform.system() == "Darwin":  # macOS
        cache_dir = Path.home() / ".plumai"
    else:  # Assume Linux
        cache_dir = Path.home() / ".plumai"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

CACHE_FILE = get_cache_directory() / "model_cache.json"


class PlumAIDeploymentError(Exception):
    pass


def print_plumai(message: str, color: str = Fore.WHITE) -> None:
    """Prints a message with a [plumai] prefix and optional color."""
    print(color + f"[plumai] {message}")


def _get_cached_model_endpoint(model_name: str) -> str:
    """Retrieve cached model endpoint from the cache file."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            return cache.get(model_name)
    return None

def _cache_model_endpoint(model_name: str, endpoint: str) -> None:
    """Cache the model endpoint."""
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    cache[model_name] = endpoint
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _deploy(model_name: str, gpu_type: str, force: bool) -> str:
    cached_endpoint = _get_cached_model_endpoint(model_name)

    if not force and cached_endpoint:
        print_plumai(f"Model {model_name} is already deployed. Using cached endpoint.", Fore.GREEN)
        return cached_endpoint

    root_dir = Path(__file__).parent
    models_dir = (root_dir / "models").resolve()

    if model_name == "flux.1-dev":
        model_file = str(models_dir / "flux1_dev.py")
    elif model_name == "flux.1-schnell":
        model_file = str(models_dir / "flux1_schnell.py")
    else:
        raise ValueError(f"{model_name} not found")

    command = f"GPU_TYPE={gpu_type} modal deploy {model_file}"
    try:
        print_plumai(f"Deploying model {model_name} gpu: {gpu_type}.", Fore.BLUE)
        print_plumai(f"This may take a few minutes on the first deployment...", Fore.BLUE)
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout

        link_pattern = r'https://[^\s"]+'
        match = re.search(link_pattern, output)
        if match:
            model_endpoint = match.group(0)
        else:
            raise PlumAIDeploymentError("Model endpoint not found.")

        _cache_model_endpoint(model_name, model_endpoint)
        print_plumai(f"Model {model_name} deployed successfully.", Fore.GREEN)
        print_plumai(f"Model endpoint: {model_endpoint}", Fore.GREEN)
        return model_endpoint
    except subprocess.CalledProcessError as e:
        print_plumai(f"Failed to deploy model {model_name}. Return code: {e.returncode}", Fore.RED)
        return
    except PlumAIDeploymentError as e:
        print_plumai(f"Error: {str(e)}", Fore.RED)
        return


def run(args):
    model_endpoint = _deploy(args.model_name, args.gpu_type, args.force)
    if model_endpoint:
        os.environ["MODEL_NAME"] = args.model_name
        os.environ["MODEL_ENDPOINT"] = model_endpoint
        port = 6868
        webbrowser.open(f"http://localhost:{port}")
        app.run(host="0.0.0.0", port=port)


def main():
    parser = argparse.ArgumentParser(prog="plumai")
    subparsers = parser.add_subparsers(dest="command")

    # Define `run` command
    run_parser = subparsers.add_parser("run", help="Deploy and run a model")
    run_parser.add_argument("model_name", type=str,
                            choices=["flux.1-dev", "flux.1-schnell"],
                            help="Model name. Available options: flux.1-dev (default), flux.1-schnell")
    run_parser.add_argument("gpu_type", type=str, nargs="?", default="A10G",
                            choices=["A10G", "A100"],
                            help="GPU type. Available options: A10G (default), A100")
    run_parser.add_argument("-f", "--force", action="store_true",
                            help="Force redeployment of the model, ignoring the cache")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        run(args)


if __name__ == "__main__":
    main()