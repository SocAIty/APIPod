import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

from apipod.deploy.deployment_manager import DeploymentManager
from socaity_cli.prompts import input_yes_no


def _deployment_manager(args=None) -> DeploymentManager:
    """Build a DeploymentManager, honoring ``-C`` / ``--project-dir`` when set."""
    start_path = getattr(args, "project_dir", None) if args is not None else None
    return DeploymentManager(start_path=start_path)


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


def run_scan(args=None):
    """Scan the project and generate apipod.json configuration file."""
    manager = _deployment_manager(args)

    if manager.config_exists and not input_yes_no(f"{manager.config_path.name} already exists in {manager.config_path.parent}/. Overwrite?"):
        print("Scan aborted.")
        return

    config_data = manager.scan()
    manager.save_config(config_data)


def run_build(args):
    """Run the build process for creating a deployment-ready container."""
    manager = _deployment_manager(args)

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


def _load_app(entrypoint: str, project_root: Optional[Path] = None):
    """Import the entrypoint module and return its APIPod application instance."""
    from apipod.engine.base_backend import _BaseBackend

    path = Path(entrypoint)
    if not path.is_absolute():
        path = (Path(project_root) if project_root else Path.cwd()) / entrypoint
    path = path.resolve()
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

    manager = _deployment_manager(args)
    entrypoint = _resolve_entrypoint(manager, args.entrypoint)
    app = _load_app(entrypoint, project_root=manager.project_root)

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


def _load_or_scan_config(args=None) -> dict:
    """Return the apipod.json config, scanning the project when missing or stale."""
    manager = _deployment_manager(args)
    config = manager.load_config() if manager.config_exists else None
    if not config or "models" not in config:
        print("Scanning project (models + includes)...")
        config = manager.scan()
        manager.save_config(config)
    return config


def run_analyze(args):
    """Analyze the project against the Socaity backend (no draft, nothing persisted)."""
    from socaity_cli.deployment import analyze_deployment

    analyze_deployment(_load_or_scan_config(args))


def run_deploy(args):
    """Full deploy: analyze, draft, build, push to the registry, poll until live."""
    from socaity_cli.deployment import run_full_deploy

    if args.push_only and not args.resume:
        print("--push-only requires --resume DEPLOYMENT_ID (the draft to push into).")
        return

    config = _load_or_scan_config(args)
    local_tag = f"apipod-{config.get('title', 'service').lower()}"

    if not args.skip_build and not args.push_only:
        manager = _deployment_manager(args)
        if not manager.dockerfile_exists:
            print("No DOCKERFILE found. Run 'apipod build' first, or use --skip-build with an existing image.")
            return
        print(f"Building container image ({local_tag})...")
        if not manager.build_docker_image(config.get("title", "service")):
            print("Docker build failed. Fix the build and retry, or use --skip-build.")
            return

    try:
        result = run_full_deploy(
            config,
            local_tag=local_tag,
            resume_deployment_id=args.resume,
            assume_yes=args.yes,
        )
    except Exception as exc:
        from socaity_cli.errors import PrivateSlotLimitError

        if isinstance(exc, PrivateSlotLimitError):
            sys.exit(1)
        raise
    if result is None:
        sys.exit(1)


def run_help(args, parsers: dict):
    """Print top-level or command-specific help."""
    help_command = getattr(args, "help_command", None)
    if help_command:
        parser = parsers.get(help_command)
        if parser is None:
            print(f"Unknown command: {help_command}\n")
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
  apipod analyze                                 Pre-deploy analysis via Socaity (no draft created)
  apipod deploy                                  Analyze + create a Socaity deployment draft
  apipod -C simple_test_service deploy           Deploy a service from a monorepo subdirectory
        """,
    )
    parser.add_argument(
        "-C",
        "--project-dir",
        default=None,
        metavar="DIR",
        help="Service project directory (must contain apipod-deploy/ or pyproject.toml).",
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

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze the project against the Socaity platform (HF checks, catalog match, GPU hint).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  apipod analyze\n\nRequires a Socaity login (socaity login); nothing is created.",
    )
    parsers["analyze"] = analyze_parser

    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy to Socaity: analyze, build, push and provision in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  apipod deploy\n"
            "  apipod deploy serverless-runpod\n"
            "  apipod -C simple_test_service deploy serverless-runpod --yes\n"
            "  apipod deploy --skip-build              Use the already-built local image\n"
            "  apipod deploy --resume DEPLOYMENT_ID    Retry the push for an existing draft"
        ),
    )
    deploy_parser.add_argument(
        "target",
        nargs="?",
        default="serverless",
        metavar="TARGET",
        help="Deployment target '{compute}-{provider}' (default: serverless).",
    )
    deploy_parser.add_argument(
        "--resume",
        default=None,
        metavar="DEPLOYMENT_ID",
        help="Skip analyze/draft and push into an existing deployment attempt.",
    )
    deploy_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not rebuild the container; push the existing local image.",
    )
    deploy_parser.add_argument(
        "--push-only",
        action="store_true",
        help="With --resume: only push and poll, never build or create drafts.",
    )
    deploy_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Non-interactive: skip confirmations (CI / automated runs).",
    )
    parsers["deploy"] = deploy_parser

    help_parser = subparsers.add_parser(
        "help",
        help="Show help for apipod or a specific command.",
    )
    help_parser.add_argument(
        "help_command",
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
    elif args.command == "analyze":
        run_analyze(args)
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
