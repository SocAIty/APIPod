import argparse
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

    # Handle target file argument
    target_file = None
    if args.build is not None and args.build is not True:
        target_file = args.build
        
        # Validate target file exists
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

    # Check if we should create/update the Dockerfile
    if manager.dockerfile_exists and not input_yes_no("Deployment config DOCKERFILE exists. Overwrite your deployment config?"):
        print("Aborting build configuration.")
        return

    should_create_dockerfile = True

    # Load or create configuration
    config_data = get_or_create_config(manager, target_file)
    if not config_data:
        print("Error: Failed to obtain configuration.")
        return

    service_title = config_data.get("title", "apipod-service")

    # Select base image
    final_image = select_base_image(manager, config_data)
    if final_image == "Enter custom base image":
        print("Please write your own Dockerfile and config.")
        return

    # Validate dependencies
    if not manager.check_dependencies():
        print("Warning: No pyproject.toml or requirements.txt found.")
        if not input_yes_no("Proceed anyway?", default=False):
            print("Please configure dependencies and try again.")
            return

    # Generate Dockerfile if requested
    if should_create_dockerfile:
        print("Generating Dockerfile for serverless deployment...")
        dockerfile_content = manager.render_dockerfile(final_image, config_data)
        manager.write_dockerfile(dockerfile_content)

    # Build Docker image
    if input_yes_no(f"Build the application now using docker? (Tag: {service_title})"):
        manager.build_docker_image(service_title)


def main():
    """Main entry point for the APIPod CLI."""
    parser = argparse.ArgumentParser(
        description="APIPod CLI - Build and deploy serverless API containers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  apipod --scan                    Scan project and generate apipod.json
  apipod --build                   Build container using current directory
  apipod --build ./main.py         Build container with specific entry file
  apipod --build ./src/app.py      Build container with file in subdirectory
        """
    )
    parser.add_argument(
        "--build", 
        nargs="?", 
        const=True, 
        metavar="FILE",
        help="Build the service container. Optionally specify a target Python file (e.g., --build main.py)"
    )
    parser.add_argument(
        "--scan", 
        action="store_true", 
        help="Scan project and generate apipod.json configuration file"
    )

    args = parser.parse_args()

    if args.scan:
        run_scan()
    elif args.build is not None:  # Changed from 'elif args.build:' to handle both True and string values
        run_build(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()