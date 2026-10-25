"""Protect the NumPy compatibility range for CLI installs in scientific environments."""

from pathlib import Path
import tomllib

import pytest
from packaging.requirements import Requirement


PROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def numpy_requirement() -> Requirement:
    project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))["project"]
    requirements = (Requirement(item) for item in project["dependencies"])
    matches = [item for item in requirements if item.name.lower() == "numpy"]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("version", ["1.26.4", "2.0.2", "2.2.6"])
def test_numpy_range_accepts_common_scientific_environments(version: str) -> None:
    # Numba 0.60 accepts NumPy 1.26.4 or 2.0.2; newer environments need 2.2 too.
    assert version in numpy_requirement().specifier


@pytest.mark.parametrize("version", ["1.26.3", "3.0.0"])
def test_numpy_range_respects_backend_floor_and_major_cap(version: str) -> None:
    assert version not in numpy_requirement().specifier
