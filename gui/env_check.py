from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import List, Optional


@dataclass
class PackageStatus:
    package: str
    required: str
    installed: Optional[str]


def _parse_requirements(requirements_path: Path) -> List[tuple[str, str]]:
    requirements: List[tuple[str, str]] = []
    for raw_line in requirements_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        package, version = line.split("==", 1)
        requirements.append((package.strip(), version.strip()))
    return requirements


def get_package_mismatches(requirements_path: Path) -> List[PackageStatus]:
    mismatches: List[PackageStatus] = []
    for package, required_version in _parse_requirements(requirements_path):
        try:
            installed_version = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed_version = None
        if installed_version != required_version:
            mismatches.append(
                PackageStatus(
                    package=package,
                    required=required_version,
                    installed=installed_version,
                )
            )
    return mismatches


def build_update_commands(requirements_path: Path, mismatches: List[PackageStatus]) -> str:
    if not mismatches:
        return ""
    package_specs = " ".join(f"{item.package}=={item.required}" for item in mismatches)
    return "\n".join(
        [
            "python -m pip install --upgrade pip",
            f"python -m pip install --upgrade --force-reinstall {package_specs}",
            f"python -m pip install -r \"{requirements_path}\"",
        ]
    )
