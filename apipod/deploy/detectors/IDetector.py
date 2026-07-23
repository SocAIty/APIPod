import os
from abc import ABC, abstractmethod
from typing import Dict, Any


class Detector(ABC):
    def __init__(self, project_root: str):
        self.project_root = project_root

    def should_ignore(self, path: str) -> bool:
        """
        Check if a directory should be ignored during file scanning.
        """
        # Get relative path from project root
        try:
            rel_path = os.path.relpath(path, self.project_root)
        except ValueError:
            # Path is not within project_root
            return True

        # Ignore common directories
        ignore_dirs = {
            '.vscode', '.idea', '.github',
            '__pycache__', '.git', '.svn', '.hg', '.DS_Store',
            'node_modules', 'venv', 'env', '.env', '.venv',
            'build', 'dist', '.pytest_cache', '.mypy_cache',
            '.tox', '.coverage', 'htmlcov', '.eggs', '*.egg-info',
            'apipod-deploy',
        }

        # Check if any part of the path matches ignore patterns
        parts = rel_path.split(os.sep)
        return any(part in ignore_dirs or part.startswith('.') for part in parts)

    @abstractmethod
    def detect(self) -> Dict[str, Any]:
        """
        Perform detection and return a dictionary of findings.
        """
        pass
