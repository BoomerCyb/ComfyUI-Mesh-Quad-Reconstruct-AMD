from pathlib import Path
import os, shutil
import torch
if torch.version.hip is None:
    raise RuntimeError('Use the existing ROCm Torch environment; no CUDA fallback is supported.')
from torch.utils.cpp_extension import load
root = Path(__file__).resolve().parent
groups = [('_C', ['src/hash/hash.cu', 'src/atlas.cu', 'src/clean_up.cu', 'src/cumesh.cu', 'src/connectivity.cu', 'src/geometry.cu', 'src/io.cu', 'src/simplify.cu', 'src/shared.cu', 'src/remesh/simple_dual_contour.cu', 'src/remesh/svox2vert.cu', 'src/ext.cpp'], []), ('_cubvh', ['third_party/cubvh/src/bvh.cu', 'third_party/cubvh/src/api_gpu.cu', 'third_party/cubvh/src/bindings.cpp'], ['third_party/cubvh/include', 'third_party/cubvh/third_party/eigen']), ('_xatlas', ['third_party/xatlas/xatlas_mod.cpp', 'third_party/xatlas/binding.cpp'], [])]
# hipCUB, rocPRIM and rocThrust: use the ROCm SDK's own copies, which match its HIP
# version, and the bundled copies only when the SDK does not ship them.
sdk_include = Path(os.environ.get('ROCM_HOME', '')) / 'include'
if all((sdk_include / h).is_file() for h in ('hipcub/hipcub.hpp', 'rocprim/rocprim.hpp', 'thrust/version.h')):
    amd_includes = []
    print('hipCUB/rocPRIM/rocThrust: ROCm SDK headers in', sdk_include, flush=True)
else:
    amd_includes = [str(root / 'third_party/amd' / d) for d in ('hipCUB', 'rocPRIM', 'rocThrust')]
    print('hipCUB/rocPRIM/rocThrust: bundled headers (the ROCm SDK does not include them)', flush=True)
for name, sources, incs in groups:
    sources = [str(Path(s).with_suffix('.hip')) if s.endswith('.cu') else str(Path(s).with_name(Path(s).stem + '_hip.cpp')) if (root / Path(s).with_name(Path(s).stem + '_hip.cpp')).exists() else s for s in sources]
    build = root / '.build' / name
    build.mkdir(parents=True, exist_ok=True)
    m = load(name=name, sources=[str(root / s) for s in sources], extra_include_paths=[str(root / s) for s in incs] + [str(root)] + amd_includes, build_directory=str(build), with_cuda=True, extra_cflags=['-O2'], extra_cuda_cflags=['-O2', '-std=c++20'], verbose=True)
    shutil.copy2(m.__file__, root / 'cumesh' / Path(m.__file__).name)
    print('BUILT', name, flush=True)
