from pathlib import Path
from typing import Any, Dict, List, Optional

from apipod.deploy.detectors.project_root import find_project_root
from apipod.deploy.docker_factory import DockerFactory
from apipod.deploy.scanner import Scanner


class DeploymentManager:
    """
    Coordinates scanning, configuration persistence and Docker operations.
    Docker specifics live in DockerFactory; detection lives in Scanner.
    """

    DEPLOY_DIR_NAME = "apipod-deploy"
    CONFIG_FILE_NAME = "apipod.json"
    DOCKERFILE_NAME = "Dockerfile"

    def __init__(self, start_path: Optional[Path] = None):
        start = Path(start_path) if start_path else Path.cwd()
        self.project_root = find_project_root(start)
        self.deploy_dir = self.project_root / self.DEPLOY_DIR_NAME
        self.config_path = self.deploy_dir / self.CONFIG_FILE_NAME
        self.dockerfile_path = self._resolve_file_case_insensitive(
            self.deploy_dir, self.DOCKERFILE_NAME
        )

        self.scanner = Scanner(root_path=self.project_root, config_path=self.config_path)
        self.docker_factory = DockerFactory(
            project_root=self.project_root,
            deploy_dir=self.deploy_dir,
            template_dir=Path(__file__).parent,
        )

    @staticmethod
    def _resolve_file_case_insensitive(directory: Path, filename: str) -> Path:
        """
        Return an existing file path regardless of case, or the expected path if missing.
        """
        directory = Path(directory)
        if directory.exists():
            for candidate in directory.iterdir():
                if candidate.name.lower() == filename.lower():
                    return candidate
        return directory / filename

    @property
    def config_exists(self) -> bool:
        return self.config_path.exists()

    @property
    def dockerfile_exists(self) -> bool:
        return Path(self.dockerfile_path).exists()

    def scan(self, target_file: Optional[str] | None = None) -> Dict[str, Any]:
        return self.scanner.scan(target_file=target_file)

    def save_config(self, config: Dict[str, Any]) -> None:
        self.scanner.save_report(config)

    def load_config(self) -> Optional[Dict[str, Any]]:
        return self.scanner.load_report()

    def recommend_image(self, config: Dict[str, Any]) -> str:
        return self.docker_factory.recommend_image(config)

    @property
    def images(self) -> List[str]:
        return self.docker_factory.images

    def render_dockerfile(self, base_image: str, config: Dict[str, Any]) -> str:
        return self.docker_factory.render_dockerfile(base_image, config)

    def write_dockerfile(self, content: str) -> Path:
        return self.docker_factory.write_dockerfile(content, self.dockerfile_path)

    def build_docker_image(self, service_title: str) -> bool:
        tag = f"apipod-{service_title.lower()}"
        return self.docker_factory.build_image(tag, self.dockerfile_path, self.project_root)

    def check_dependencies(self) -> bool:
        """Check if project has dependency files."""
        has_toml = (self.project_root / "pyproject.toml").exists()
        has_req = (self.project_root / "requirements.txt").exists()
        return has_toml or has_req
