import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

from apipod.deploy.deployment_manager import DeploymentManager


def input_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for a yes/no response with default value."""
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] " if default else " [y/N] "
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return default
        if choice in valid:
            return valid[choice]
        sys.stdout.write("Please respond with 'yes' or 'no' (or 'y'/'n').\n")


def _parse_bool(value) -> Optional[bool]:
    """Coerce a CLI flag value into a bool. ``None`` (flag absent) stays ``None``."""
    if value is None or isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def select_base_image(manager: DeploymentManager, config_data: dict) -> str:
    """Interactive base image selection process."""
    recommended_image = manager.recommend_image(config_data)
    print(f"Detected configuration: Python {config_data.get('python_version')}, "
          f"PyTorch: {config_data.get('pytorch')}, TensorFlow: {config_data.get('tensorflow')}, "
          f"ONNX: {config_data.get('onnx')}")
    print(f"Recommended Base Image: {recommended_image}")

    if input_yes_no("Is this correct?"):
        return recommended_image

    print("Select a base image:")
    for i, img in enumerate(manager.images, 1):
        print(f"{i}. {img}")
    print(f"{len(manager.images) + 1}. Enter custom image")

    while True:
        try:
            selection = input("Selection: ").strip()
            idx = int(selection) - 1
            if 0 <= idx < len(manager.images):
                return manager.images[idx]
            elif idx == len(manager.images):
                custom_image = input("Enter custom base image: ").strip()
                if custom_image:
                    return custom_image
        except ValueError:
            pass
        print("Invalid selection. Please try again.")


def get_or_create_config(manager: DeploymentManager, target_file: Optional[str] = None) -> Optional[dict]:
    """
    Load existing config or create new one through scanning.
    If target_file is provided, scanning will focus on that file.
    """
    def perform_scan():
        if target_file:
            print(f"Scanning project with target file: {target_file}...")
            return manager.scan(target_file)
        else:
            print("Scanning project...")
            return manager.scan()

    if manager.config_exists:
        if not input_yes_no(f"Found {manager.config_path.name} in {manager.config_path.parent}/. Overwrite?"):
            return manager.load_config()

        print("Rescanning project...")
        config_data = perform_scan()
        manager.save_config(config_data)
        return config_data

    print(f"No {manager.config_path.name} found. Scanning project...")
    config_data = perform_scan()
    manager.save_config(config_data)
    return config_data


def run_scan():
    """Scan the project and generate apipod.json configuration file."""
    manager = DeploymentManager()

    if manager.config_exists and not input_yes_no(f"{manager.config_path.name} already exists in {manager.config_path.parent}/. Overwrite?"):
        print("Scan aborted.")
        return

    config_data = manager.scan()
    manager.save_config(config_data)


def run_build(args):
    """Run the build process for creating a deployment-ready container."""
    manager = DeploymentManager()

    target_file = None
    if args.build is not None and args.build is not True:
        target_file = args.build

        target_path = Path(target_file)
        if not target_path.exists():
            print(f"Error: Target file '{target_file}' does not exist.")
            return

        if not target_path.is_file():
            print(f"Error: '{target_file}' is not a file.")
            return

        if not target_path.suffix == '.py':
            print(f"Warning: '{target_file}' is not a Python file (.py)")
            if not input_yes_no("Continue anyway?", default=False):
                return

        print(f"Using target file: {target_file}")

    if manager.dockerfile_exists and not input_yes_no("Deployment config DOCKERFILE exists. Overwrite your deployment config?"):
        print("Aborting build configuration.")
        return

    config_data = get_or_create_config(manager, target_file)
    if not config_data:
        print("Error: Failed to obtain configuration.")
        return

    service_title = config_data.get("title", "apipod-service")

    final_image = select_base_image(manager, config_data)
    if final_image == "Enter custom base image":
        print("Please write your own Dockerfile and config.")
        return

    if not manager.check_dependencies():
        print("Warning: No pyproject.toml or requirements.txt found.")
        if not input_yes_no("Proceed anyway?", default=False):
            print("Please configure dependencies and try again.")
            return

    print("Generating Dockerfile...")
    dockerfile_content = manager.render_dockerfile(final_image, config_data)
    manager.write_dockerfile(dockerfile_content)

    if input_yes_no(f"Build the application now using docker? (Tag: {service_title})"):
        manager.build_docker_image(service_title)


def _resolve_entrypoint(manager: DeploymentManager, entrypoint: Optional[str]) -> str:
    """Return the service entrypoint file, scanning the project when not provided."""
    if entrypoint:
        return entrypoint

    config = manager.load_config() if manager.config_exists else None
    if not config:
        print("No apipod.json found. Scanning project to locate the entrypoint...")
        config = manager.scan()
        manager.save_config(config)
    return config.get("entrypoint", "main.py")


def _load_app(entrypoint: str):
    """Import the entrypoint module and return its APIPod application instance."""
    from apipod.engine.base_backend import _BaseBackend

    path = Path(entrypoint).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Entrypoint '{entrypoint}' not found.")

    spec = importlib.util.spec_from_file_location("apipod_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    app = getattr(module, "app", None)
    if isinstance(app, _BaseBackend):
        return app
    for value in vars(module).values():  # fall back to the first app in the module
        if isinstance(value, _BaseBackend):
            return value
    raise RuntimeError(f"No APIPod app found in '{entrypoint}'. Expected an APIPod() instance.")


def _run_service(args, simulate: Optional[str], direct: Optional[bool]):
    """Resolve the entrypoint, apply the run intent via env vars, and start the app."""
    # The user's service calls APIPod() with no args; it reads the intent from env.
    # Set the env BEFORE importing the entrypoint (which imports apipod settings).
    if simulate is not None:
        os.environ["APIPOD_SIMULATE"] = simulate
    if direct is not None:
        os.environ["APIPOD_DIRECT"] = "true" if direct else "false"

    manager = DeploymentManager()
    entrypoint = _resolve_entrypoint(manager, args.entrypoint)
    app = _load_app(entrypoint)

    port = args.port or 8000
    host = args.host or "0.0.0.0"

    mode = "development" if simulate is None else f"simulation '{simulate}'{' --direct' if direct else ''}"
    print(f"Starting APIPod ({mode}) from {entrypoint}")
    app.start(port=port, host=host)


def run_start(args):
    """Run the service locally for development (plain FastAPI)."""
    _run_service(args, simulate=None, direct=None)


def run_simulate(args):
    """Run the service locally while emulating a deployment target."""
    target = args.simulate or "serverless"  # bare --simulate defaults to serverless
    _run_service(args, simulate=target, direct=_parse_bool(args.direct))


def run_deploy(args):
    """Placeholder for the upcoming managed deployment command."""
    target = args.deploy or "serverless"
    print(
        f"`apipod --deploy {target}` is not available yet.\n"
        "Deploy through the Socaity dashboard for now: https://www.socaity.ai\n"
        "Tip: validate the target locally first with `apipod --simulate "
        f"{target}`."
    )


def main():
    """Main entry point for the APIPod CLI."""
    parser = argparse.ArgumentParser(
        description="APIPod CLI - build, simulate and deploy AI services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  apipod --start                               Run locally for development (FastAPI)
  apipod --simulate                            Emulate a serverless deployment (FastAPI + job queue)
  apipod --simulate serverless-runpod          Emulate Socaity managed deploy to RunPod
  apipod --simulate serverless-runpod --direct Emulate RunPod's native worker directly
  apipod --simulate dedicated-azure            Emulate a dedicated Azure deployment
  apipod --scan                                Scan project and generate apipod.json
  apipod --build                               Build the deployment container
  apipod --deploy                              Deploy via Socaity (coming soon)
        """
    )

    parser.add_argument(
        "--start",
        action="store_true",
        help="Run the service locally for development (plain FastAPI)."
    )
    parser.add_argument(
        "--simulate",
        nargs="?",
        const="",
        metavar="TARGET",
        help="Emulate a deployment locally. Optional target '{compute}-{provider}' "
             "(e.g. serverless-runpod, dedicated-azure). Defaults to 'serverless'."
    )
    parser.add_argument(
        "--direct",
        nargs="?",
        const=True,
        default=None,
        metavar="BOOL",
        help="Emulate the provider's native serverless worker instead of Socaity's job queue."
    )
    parser.add_argument(
        "--deploy",
        nargs="?",
        const="",
        metavar="TARGET",
        help="Deploy the service via Socaity (coming soon)."
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan project and generate apipod.json configuration file."
    )
    parser.add_argument(
        "--build",
        nargs="?",
        const=True,
        metavar="FILE",
        help="Build the service container. Optionally specify a target Python file."
    )
    parser.add_argument(
        "--entrypoint",
        default=None,
        help="Service entrypoint file for --start / --simulate (auto-detected if omitted)."
    )
    parser.add_argument("--host", default=None, help="Host to bind to (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (default: 8000).")

    args = parser.parse_args()

    if args.scan:
        run_scan()
    elif args.build is not None:
        run_build(args)
    elif args.deploy is not None:
        run_deploy(args)
    elif args.simulate is not None:
        run_simulate(args)
    elif args.start:
        run_start(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
