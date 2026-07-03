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


def select_base_image(manager: DeploymentManager, config_data: dict) -> str:
    """Interactive base image selection process."""
    recommended_image = manager.recommend_image(config_data)
    print(
        f"Detected configuration: profile={config_data.get('profile')}, "
        f"Python {config_data.get('python_version')}, "
        f"PyTorch: {config_data.get('pytorch')}, TensorFlow: {config_data.get('tensorflow')}, "
        f"ONNX: {config_data.get('onnx')}"
    )
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


def run_scan(_args=None):
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

    target_file = args.file
    if target_file:
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


def _looks_like_entrypoint(value: str) -> bool:
    """Return True when a simulate positional arg is likely a Python entrypoint file."""
    return value.endswith(".py") or Path(value).is_file()


def _resolve_simulate_args(args) -> tuple[str, Optional[str]]:
    """Split simulate positionals into deployment target and optional entrypoint."""
    if args.entrypoint is not None:
        return args.target, args.entrypoint

    if args.target is None:
        return "serverless", None

    if _looks_like_entrypoint(args.target):
        return "serverless", args.target

    return args.target, None


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


def _run_service(args, simulate: Optional[str], native: bool = False):
    """Resolve the entrypoint, apply the run intent via env vars, and start the app."""
    # The user's service calls APIPod() with no args; it reads the intent from env.
    # Set the env BEFORE importing the entrypoint (which imports apipod settings).
    if simulate is not None:
        os.environ["APIPOD_SIMULATE"] = simulate
    if native:
        os.environ["APIPOD_NATIVE"] = "true"

    manager = DeploymentManager()
    entrypoint = _resolve_entrypoint(manager, args.entrypoint)
    app = _load_app(entrypoint)

    port = args.port or 8000
    host = args.host or "0.0.0.0"

    if simulate is None:
        mode = "development"
    else:
        mode = f"simulation '{simulate}'{' --native' if native else ''}"
    print(f"Starting APIPod ({mode}) from {entrypoint}")
    app.start(port=port, host=host)


def run_start(args):
    """Run the service locally for development (plain FastAPI)."""
    _run_service(args, simulate=None)


def run_simulate(args):
    """Run the service locally while emulating a deployment target."""
    target, entrypoint = _resolve_simulate_args(args)
    args.entrypoint = entrypoint
    _run_service(args, simulate=target, native=args.native)


def run_deploy(args):
    """Placeholder for the upcoming managed deployment command."""
    target = args.target or "serverless"
    print(
        f"`apipod deploy {target}` is not available yet.\n"
        "Deploy through the Socaity dashboard for now: https://www.socaity.ai\n"
        f"Tip: validate the target locally first with `apipod simulate {target}`."
    )


def run_help(args, parsers: dict):
    """Print top-level or command-specific help."""
    if args.command:
        parser = parsers.get(args.command)
        if parser is None:
            print(f"Unknown command: {args.command}\n")
            parsers["__root__"].print_help()
            sys.exit(1)
        parser.print_help()
    else:
        parsers["__root__"].print_help()


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    """Shared host/port and entrypoint options for start and simulate."""
    parser.add_argument(
        "entrypoint",
        nargs="?",
        default=None,
        metavar="ENTRYPOINT",
        help="Service entrypoint file (auto-detected from apipod.json when omitted).",
    )
    parser.add_argument("--host", default=None, help="Host to bind to (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (default: 8000).")


def _build_parser() -> tuple[argparse.ArgumentParser, dict]:
    """Construct the CLI parser tree and return the root parser plus a lookup map."""
    parser = argparse.ArgumentParser(
        description="APIPod CLI - build, simulate and deploy AI services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  apipod help                                    Show this overview
  apipod start                                   Run locally for development (FastAPI)
  apipod start main.py                           Run a specific entrypoint
  apipod simulate                                Emulate a serverless deployment (FastAPI + job queue)
  apipod simulate serverless-runpod              Emulate Socaity managed deploy to RunPod
  apipod simulate serverless-runpod --native     Emulate RunPod's native worker directly
  apipod simulate dedicated-azure                Emulate a dedicated Azure deployment
  apipod scan                                    Scan project and generate apipod.json
  apipod build                                   Build the deployment container
  apipod build path/to/service.py                Build from a specific Python file
  apipod deploy                                  Deploy via Socaity (coming soon)
        """,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    parsers: dict = {"__root__": parser}

    start_parser = subparsers.add_parser(
        "start",
        help="Run the service locally for development (plain FastAPI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  apipod start\n  apipod start main.py --port 8123",
    )
    _add_run_options(start_parser)
    parsers["start"] = start_parser

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Emulate a deployment target locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  apipod simulate\n"
            "  apipod simulate serverless-runpod\n"
            "  apipod simulate serverless-runpod main.py --native"
        ),
    )
    simulate_parser.add_argument(
        "target",
        nargs="?",
        default=None,
        metavar="TARGET",
        help="Deployment target '{compute}-{provider}' (e.g. serverless-runpod, dedicated-azure). "
             "Defaults to 'serverless'. A .py value is treated as the entrypoint.",
    )
    simulate_parser.add_argument(
        "entrypoint",
        nargs="?",
        default=None,
        metavar="ENTRYPOINT",
        help="Service entrypoint file (auto-detected when omitted).",
    )
    simulate_parser.add_argument(
        "--native",
        action="store_true",
        help="Emulate the provider's native serverless worker instead of Socaity's job queue.",
    )
    simulate_parser.add_argument("--host", default=None, help="Host to bind to (default: 0.0.0.0).")
    simulate_parser.add_argument("--port", type=int, default=None, help="Port to bind to (default: 8000).")
    parsers["simulate"] = simulate_parser

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan project and generate apipod.json configuration file.",
    )
    parsers["scan"] = scan_parser

    build_parser = subparsers.add_parser(
        "build",
        help="Build the service container.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  apipod build\n  apipod build path/to/service.py",
    )
    build_parser.add_argument(
        "file",
        nargs="?",
        default=None,
        metavar="FILE",
        help="Target Python file to scan (project-wide scan when omitted).",
    )
    parsers["build"] = build_parser

    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy the service via Socaity (coming soon).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  apipod deploy\n  apipod deploy serverless-runpod",
    )
    deploy_parser.add_argument(
        "target",
        nargs="?",
        default="serverless",
        metavar="TARGET",
        help="Deployment target '{compute}-{provider}' (default: serverless).",
    )
    parsers["deploy"] = deploy_parser

    help_parser = subparsers.add_parser(
        "help",
        help="Show help for apipod or a specific command.",
    )
    help_parser.add_argument(
        "command",
        nargs="?",
        default=None,
        metavar="COMMAND",
        help="Command to show help for (omit for the overview).",
    )
    parsers["help"] = help_parser

    return parser, parsers


def main():
    """Main entry point for the APIPod CLI."""
    parser, parsers = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "scan":
        run_scan(args)
    elif args.command == "build":
        run_build(args)
    elif args.command == "deploy":
        run_deploy(args)
    elif args.command == "simulate":
        run_simulate(args)
    elif args.command == "start":
        run_start(args)
    elif args.command == "help":
        run_help(args, parsers)


if __name__ == "__main__":
    main()
