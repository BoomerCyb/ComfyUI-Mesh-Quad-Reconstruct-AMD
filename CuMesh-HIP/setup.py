from pathlib import Path
from setuptools import setup, Distribution
import torch
root = Path(__file__).resolve().parent
if torch.version.hip is None:
    raise RuntimeError("Use ROCm PyTorch to install this HIP backend.")
for name in ("_C", "_cubvh", "_xatlas"):
    if not (root / "cumesh" / (name + ".pyd")).exists():
        raise RuntimeError("Run build_hip.py before packaging or installation.")
class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True
setup(packages=["cumesh"], package_data={"cumesh": ["*.pyd"]},
      distclass=NativeDistribution)
