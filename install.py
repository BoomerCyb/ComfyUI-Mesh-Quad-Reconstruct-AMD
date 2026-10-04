from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


NODE_DIR = Path(__file__).resolve().parent


def _run(command, env=None):
    command = [str(item) for item in command]
    if env is not None:
        executable = shutil.which(command[0], path=env.get("PATH"))
        if executable:
            command[0] = executable
    print("[Installer]", subprocess.list2cmdline(command), flush=True)
    subprocess.check_call(command, cwd=NODE_DIR, env=env)


def _gpu_build_architectures(env, torch):
    # Respect an explicit target supplied by the user or their launcher.
    if env.get("PYTORCH_ROCM_ARCH", "").strip():
        print("[Installer] GPU architectures (override):", env["PYTORCH_ROCM_ARCH"])
        return
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        arch = getattr(props, "gcnArchName", "").split(":", 1)[0]
        if not arch:
            raise RuntimeError("Cannot determine the HIP architecture for " + props.name +
                               ". Set PYTORCH_ROCM_ARCH explicitly.")
        integrated = getattr(props, "is_integrated", getattr(props, "integrated", None))
        devices.append((props.name, arch, integrated))
    if not devices:
        raise RuntimeError("No visible ROCm GPU is available for compilation.")
    # Older PyTorch builds may lack the HIP integrated-device property.
    # Keep unclassified devices rather than guessing from names or VRAM.
    selected = [device for device in devices if device[2] != 1]
    if not selected:
        selected = devices
        print("[Installer] Only integrated GPUs are visible; targeting those GPUs.")
    for name, arch, integrated in devices:
        if integrated is None:
            print("[Installer] GPU type unavailable; retaining:", name, arch)
        elif (name, arch, integrated) not in selected:
            print("[Installer] Excluding integrated GPU:", name, arch)
    env["PYTORCH_ROCM_ARCH"] = ";".join(dict.fromkeys(device[1] for device in selected))
    print("[Installer] GPU build architectures:", env["PYTORCH_ROCM_ARCH"])


def _native_environment():
    if os.name != "nt":
        raise RuntimeError("This native build currently supports Windows x64.")
    include = Path(sysconfig.get_paths()['include']) / 'Python.h'
    library = Path(sys.base_prefix) / 'libs' / ('python'+str(sys.version_info.major)+str(sys.version_info.minor)+'.lib')
    if not include.is_file() or not library.is_file():
        raise RuntimeError('ComfyUI Python development files are missing: '+str(include)+' or '+str(library)+'. Native compilation needs matching Python headers and the import library.')
    env = {key.upper(): value for key, value in os.environ.items()}
    sdk = env.get("ROCM_HOME") or env.get("HIP_PATH") or env.get("ROCM_PATH")
    if not sdk:
        for name in ("_rocm_sdk_core", "_rocm_sdk_devel"):
            spec = importlib.util.find_spec(name)
            if spec and spec.origin:
                candidate = Path(spec.origin).parent
                if (candidate / "lib/llvm/bin/clang-cl.exe").is_file():
                    sdk = str(candidate)
                    break
    if not sdk or not (Path(sdk) / "lib/llvm/bin/clang-cl.exe").is_file():
        raise RuntimeError("Set ROCM_HOME to the matching Windows HIP SDK directory.")

    vswhere = Path(env.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        raise RuntimeError("Install Visual Studio C++ Build Tools and the Windows SDK.")
    installation = subprocess.check_output([
        str(vswhere), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath",
    ], text=True).strip()
    if not installation:
        raise RuntimeError("Visual Studio x64 C++ Build Tools were not found.")
    vs = Path(installation)
    script = NODE_DIR / ".build" / "developer-environment.cmd"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('@echo off\ncall "' + str(vs / "Common7/Tools/VsDevCmd.bat") + '" -no_logo -arch=x64 -host_arch=x64 >nul\nif errorlevel 1 exit /b 1\nset\n', encoding="utf-8")
    vcpkg_settings = {key: env[key] for key in ("VCPKG_ROOT", "VCPKG_INSTALLED_DIR") if key in env}
    output = subprocess.check_output(["cmd.exe", "/d", "/c", str(script)], text=True, env=env)
    for line in output.splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            env[key.upper()] = value
    env.update(vcpkg_settings)
    cmake_tools = vs / "Common7/IDE/CommonExtensions/Microsoft/CMake"
    env["PATH"] = os.pathsep.join([
        str(cmake_tools / "Ninja"), str(cmake_tools / "CMake/bin"),
        str(Path(sys.executable).parent / "Scripts"), str(Path(sdk) / "bin"), env.get("PATH", ""),
    ])
    env["ROCM_HOME"] = sdk
    env["HIP_PATH"] = sdk
    env["CXX"] = str(Path(sdk) / "lib/llvm/bin/clang-cl.exe")
    env["DISTUTILS_USE_SDK"] = "1"
    env["MSSDK"] = "1"
    env.setdefault("MAX_JOBS", "2")
    for tool in ("cl.exe", "ninja.exe", "cmake.exe"):
        if not shutil.which(tool, path=env["PATH"]):
            raise RuntimeError("Required build tool is missing: " + tool)
    return env


def _install(env):
    _run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-r", "requirements.txt"], env)
    _run([sys.executable, "CuMesh-HIP/build_hip.py"], env)
    _run([sys.executable, "-m", "pip", "install", "./CuMesh-HIP", "--no-build-isolation", "--no-deps"], env)
    _run([sys.executable, "-c", "import cumesh._C, cumesh._cubvh, cumesh._xatlas; print('CuMesh native modules loaded')"], env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check prerequisites without compiling or installing.")
    parser.add_argument("--build-if-needed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    import torch
    if not torch.version.hip:
        raise RuntimeError("Use the ROCm PyTorch environment that runs ComfyUI.")
    if not torch.cuda.is_available():
        raise RuntimeError("The ROCm GPU is unavailable in this PyTorch environment.")
    print("[Installer] Python:", sys.executable)
    print("[Installer] PyTorch:", torch.__version__, "HIP:", torch.version.hip)
    env = _native_environment()
    _gpu_build_architectures(env, torch)
    if args.check:
        print("[Installer] Prerequisites checked; no modules were compiled or installed.")
        return 0
    _install(env)
    print("[Installer] Installation completed. If installing a node group, wait for all installers before restarting ComfyUI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
