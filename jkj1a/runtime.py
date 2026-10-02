"""Shared local/Colab setup for the teaching notebooks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _valid_root(path: Path) -> bool:
    return (path / "requirements-colab.txt").is_file() and (
        path / "jkj1a/__init__.py"
    ).is_file()


def _find_root(drive_project: str) -> Path | None:
    drive_root = Path("/content/drive/MyDrive")
    candidates = [
        drive_root / drive_project,
        drive_root / "Colab Notebooks" / "JKJ1A-test" / "JKJ1A",
        drive_root / "Colab Notebooks" / "JKJ1A",
        drive_root / "JKJ1A",
    ]
    for candidate in candidates:
        if _valid_root(candidate):
            return candidate
    # Avoid scanning an entire large Drive. The usual course locations are small.
    detected = []
    for base in (drive_root / "Colab Notebooks", drive_root / "授業"):
        if base.is_dir():
            detected.extend(
                p.parent
                for p in base.rglob("requirements-colab.txt")
                if _valid_root(p.parent)
            )
    return detected[0] if len(detected) == 1 else None


def prepare(*, drive_project: str = "Colab Notebooks/JKJ1A",
            project_root: str | Path | None = None,
            install_colab: bool = True,
            already_mounted: bool | None = None) -> dict[str, object]:
    """Mount Drive when needed, locate the project, and prepare imports."""
    try:
        import google.colab  # type: ignore  # noqa: F401
        on_colab = True
    except ImportError:
        on_colab = False

    if on_colab:
        if already_mounted is None:
            already_mounted = Path("/content/drive/MyDrive").is_dir()
        if not already_mounted:
            from google.colab import drive

            drive.mount("/content/drive")
        if project_root is not None:
            project_root = Path(project_root)
            if not project_root.is_absolute():
                project_root = Path("/content/drive/MyDrive") / project_root
        else:
            project_root = _find_root(drive_project)
    else:
        if project_root is not None:
            project_root = Path(project_root)
        else:
            candidates = [Path.cwd(), *Path.cwd().parents]
            project_root = next((p for p in candidates if _valid_root(p)), None)

    if project_root is None:
        expected = Path("/content/drive/MyDrive") / drive_project if on_colab else Path.cwd()
        raise FileNotFoundError(
            "JKJ1Aが見つかりません。"
            f"\n期待した場所: {expected}"
            "\nDrive内にリポジトリ全体を置くか、DRIVE_PROJECTを確認してください。"
        )

    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if on_colab and install_colab and not os.environ.get("JKJ1A_COLAB_DEPS_READY"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"]
        )
        os.environ["JKJ1A_COLAB_DEPS_READY"] = "1"

    return {"ON_COLAB": on_colab, "project_root": project_root}
