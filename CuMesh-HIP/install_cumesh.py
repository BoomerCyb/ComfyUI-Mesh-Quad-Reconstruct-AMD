"""Build and install the bundled CuMesh HIP package.

ComfyUI-CuMesh-Decimate-AMD and ComfyUI-Mesh-Quad-Reconstruct-AMD both ship this
directory and install the same `cumesh` package, so the last installer to run
decides which build the other node uses. This script:

- keeps an installed cumesh with a newer version instead of downgrading it,
- skips the rebuild when the identical build is already installed,
- otherwise builds, installs and import-tests the package, restoring the
  previously installed package if any of that fails.

Usage: python install_cumesh.py [--force]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from importlib import metadata
from pathlib import Path

import torch
from packaging.version import Version

ROOT = Path(__file__).resolve().parent
BUILD_INFO = "_build_info.json"
NATIVE_MODULES = ("_C", "_cubvh", "_xatlas")


def bundled_version():
    with open(ROOT / "pyproject.toml", "rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def source_digest():
    """Hash of everything that affects the built package (line endings normalized)."""
    files = sorted(
        path for pattern in ("src/**/*", "cumesh/*.py", "build_hip.py", "setup.py", "pyproject.toml")
        for path in ROOT.glob(pattern) if path.is_file()
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def build_identity():
    return {
        "version": bundled_version(),
        "source_sha256": source_digest(),
        "architectures": os.environ.get("PYTORCH_ROCM_ARCH", ""),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": sys.version.split()[0],
    }


def site_packages():
    return Path(sysconfig.get_paths()["platlib"])


def installed_files():
    """The installed cumesh package and dist-info folders, if any."""
    root = site_packages()
    return [p for p in [root / "cumesh", *root.glob("cumesh-*.dist-info")] if p.exists()]


def installed_build():
    try:
        version = metadata.version("cumesh")
    except metadata.PackageNotFoundError:
        return None
    info = site_packages() / "cumesh" / BUILD_INFO
    build = json.loads(info.read_text(encoding="utf-8")) if info.is_file() else None
    return {"version": version, "build": build}


def ensure_not_loaded():
    """Windows keeps a loaded .pyd locked; reinstalling then fails half-way."""
    for module in NATIVE_MODULES:
        path = site_packages() / "cumesh" / (module + ".pyd")
        if path.is_file():
            try:
                with open(path, "r+b"):
                    pass
            except PermissionError:
                raise SystemExit(f"{path} is in use. Close ComfyUI, then run the installer again.")


def import_test():
    subprocess.check_call([
        sys.executable, "-c",
        "import cumesh._C, cumesh._cubvh, cumesh._xatlas, cumesh; print('CuMesh native modules loaded')",
    ])


def main():
    force = "--force" in sys.argv[1:]
    identity = build_identity()
    installed = installed_build()
    if installed and not force:
        if Version(installed["version"]) > Version(identity["version"]):
            print(f"[CuMesh] Keeping installed cumesh {installed['version']}; it is newer than the "
                  f"bundled {identity['version']} (installed by another AMD mesh node).", flush=True)
            import_test()
            return
        if installed["build"] == identity:
            print(f"[CuMesh] cumesh {identity['version']} is already built from these sources for "
                  f"{identity['architectures'] or 'this GPU'}; nothing to rebuild.", flush=True)
            import_test()
            return

    ensure_not_loaded()
    subprocess.check_call([sys.executable, str(ROOT / "build_hip.py")], cwd=ROOT)
    (ROOT / "cumesh" / BUILD_INFO).write_text(json.dumps(identity, indent=2), encoding="utf-8")

    backup = ROOT / ".build" / "installed-backup"
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir(parents=True)
    previous = installed_files()
    for path in previous:
        shutil.copytree(path, backup / path.name)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", str(ROOT),
                               "--no-build-isolation", "--no-deps"], cwd=ROOT)
        import_test()
    except Exception:
        print("[CuMesh] Installation failed; restoring the previously installed cumesh.", flush=True)
        for path in installed_files():
            shutil.rmtree(path)
        for path in previous:
            shutil.copytree(backup / path.name, path)
        raise
    print(f"[CuMesh] Installed cumesh {identity['version']}.", flush=True)


if __name__ == "__main__":
    main()
