"""Resolve the one authored version from the checkout or distribution metadata."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def resolve_version() -> str:
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if project_file.is_file():
        with project_file.open("rb") as stream:
            project = tomllib.load(stream).get("project", {})
        if project.get("name") == "twhy":
            return str(project["version"])
    try:
        return version("twhy")
    except PackageNotFoundError:
        return "0+unknown"
