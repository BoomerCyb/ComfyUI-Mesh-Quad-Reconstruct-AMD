from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


NODE_DIR = Path(__file__).resolve().parent


def _run(command):
    print("[Mesh Quad Installer]", " ".join(map(str, command)), flush=True)
    subprocess.check_call([str(item) for item in command])


def _pip_install(*arguments):
    _run([sys.executable, "-m", "pip", "install", *arguments])


def _has_visualbruno_backend():
    try:
        module = importlib.import_module("cumesh")
        return callable(
            getattr(module.remeshing, "reconstruct_mesh_dc_quad", None)
        )
    except Exception:
        return False


def _torch_folder_name():
    import torch

    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if not match:
        return None
    major, minor = match.groups()
    return f"Torch{major}{minor}0"


def _matching_sibling_wheel():
    """Find the exact wheel already shipped by a sibling ComfyUI-Trellis2."""
    custom_nodes_dir = NODE_DIR.parent
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_folder = _torch_folder_name()
    system = platform.system().lower()
    platform_token = "win_amd64" if system == "windows" else "linux"

    candidates = []
    for sibling in custom_nodes_dir.iterdir():
        if not sibling.is_dir() or "trellis2" not in sibling.name.lower():
            continue
        wheels_dir = sibling / "wheels"
        if not wheels_dir.is_dir():
            continue
        for wheel in wheels_dir.rglob("cumesh-*.whl"):
            path_text = str(wheel)
            if python_tag not in wheel.name:
                continue
            if platform_token not in wheel.name.lower():
                continue
            if torch_folder and torch_folder.lower() not in path_text.lower():
                continue
            candidates.append(wheel)
    return sorted(candidates)[-1] if candidates else None


def _build_visualbruno_cumesh():
    source_dir = NODE_DIR / "_build" / "visualbruno-CuMesh"
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        _run(
            [
                "git",
                "clone",
                "--recursive",
                "https://github.com/visualbruno/CuMesh.git",
                source_dir,
            ]
        )
    _pip_install(str(source_dir), "--no-build-isolation", "--no-cache-dir")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-if-needed",
        action="store_true",
        help="Build VisualBruno/CuMesh if no installed backend or sibling wheel exists.",
    )
    args = parser.parse_args()

    _pip_install("-r", str(NODE_DIR / "requirements.txt"))

    if _has_visualbruno_backend():
        print("[Mesh Quad Installer] VisualBruno CuMesh backend is already ready.")
        return 0

    wheel = _matching_sibling_wheel()
    if wheel is not None:
        print(f"[Mesh Quad Installer] Installing matching sibling wheel: {wheel}")
        _pip_install(str(wheel), "--force-reinstall", "--no-deps")
        importlib.invalidate_caches()
        if _has_visualbruno_backend():
            print("[Mesh Quad Installer] CuMesh backend installed successfully.")
            return 0

    if args.build_if_needed:
        print(
            "[Mesh Quad Installer] No matching wheel found; building VisualBruno/CuMesh.\n"
            "This requires Visual Studio C++ Build Tools and a CUDA Toolkit compatible "
            "with ComfyUI's PyTorch."
        )
        _build_visualbruno_cumesh()
        importlib.invalidate_caches()
        if _has_visualbruno_backend():
            print("[Mesh Quad Installer] CuMesh backend built successfully.")
            return 0

    print(
        "\n[Mesh Quad Installer] The Python node is installed, but the required "
        "VisualBruno CuMesh backend was not found.\n"
        "Either keep ComfyUI-Trellis2 installed so its matching wheel is available, "
        "or run Install-Windows.cmd to build the backend."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

