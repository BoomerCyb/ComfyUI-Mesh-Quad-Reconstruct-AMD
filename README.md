# ComfyUI-Mesh-Quad-Reconstruct-AMD

## AMD / ROCm

The installer uses ComfyUI's Python and stops if setup fails. When installing through EZi, wait for the entire node group to complete before restarting.

This fork keeps the original node interface and adds native HIP support.
Use the ROCm PyTorch installation that runs ComfyUI and a matching HIP SDK.
The source does not select a card model or impose a gfx1201 target. Native
extensions target visible discrete AMD GPUs by default, excluding integrated GPUs
when a discrete GPU is available. Set PYTORCH_ROCM_ARCH to override the targets.
Integrated-only systems remain supported; PyTorch supplies the compiler flags.
An installed binary still needs to match its GPU target, Python and Torch runtime.
Hardware support depends on ROCm/PyTorch; validation here covers RX 9070 XT.

For manual installation, close ComfyUI and run `install_requirements.bat` to build/install the native components. ComfyUI-Easy-Install-AMD runs this automatically through its add-on menu.
For prerequisites and manual commands, see [COMFYUI_ROCM_BUILD_GUIDE.md](COMFYUI_ROCM_BUILD_GUIDE.md).

ComfyUI AMD installer: [BoomerCyb/ComfyUI-Easy-Install-AMD](https://github.com/BoomerCyb/ComfyUI-Easy-Install-AMD).


Standalone adaptation of VisualBruno's **Trellis2 - Reconstruct Mesh With Quad**
node for ComfyUI's current native mesh type:

```text
MESH -> MESH
```

It has no dependency on CelloCut or Trellis `MESHWITHVOXEL` objects. It calls the
same HIP function used by VisualBruno:

```python
cumesh.remeshing.reconstruct_mesh_dc_quad(...)
```

## Node

**Mesh - Reconstruct With Quad (512-8K)**

Controls and defaults match VisualBruno where applicable:

| Control | Default | Meaning |
| --- | ---: | --- |
| `remesh_band` | `1.0` | Voxel-band thickness. This adaptation fixes the original wrapper and actually passes it to CuMesh. |
| `resolution` | `512` | Every multiple of 512 from 512 through 8192. |
| `remove_floaters` | `true` | Uses the same VisualBruno/PyMeshLab 0.5% face-component rule on HIP. |
| `remove_inner_faces` | `false` | Requests CuMesh's inner-surface filtering. It can create holes on difficult meshes. |

Reconstruction creates new topology, so UVs, materials, textures, colors, and
custom normals are not preserved.

### Faster floater removal

The default floater stage now stays on HIP. It reproduces both parts of
VisualBruno's PyMeshLab operation: select face-connected components below 0.5%
of the largest component (including MeshLab's integer-threshold truncation),
then remove every vertex touched by that selection and all incident faces. Work
is chunked in 1,048,576-face blocks to keep temporary allocations bounded.

The node unloads inactive ComfyUI models before reconstruction. If the HIP
connectivity stage still cannot allocate enough VRAM, the error is caught and
the node automatically uses the original PyMeshLab implementation. This keeps
the quality rule unchanged instead of crashing or silently disabling floater
removal.

## Installation

### ComfyUI-Easy-Install-AMD

In [ComfyUI-Easy-Install-AMD](https://github.com/BoomerCyb/ComfyUI-Easy-Install-AMD), select **Easy Menu → Add-ons → BoomerCyb WTiVo AMD Nodes**. It downloads the nodes and runs their installers automatically. You do not need to run `install_requirements.bat` separately. Wait for all five nodes to finish; EZi restarts ComfyUI after the group completes successfully.

### Manual installation

1. Place this repository in `ComfyUI/custom_nodes/ComfyUI-Mesh-Quad-Reconstruct-AMD`.
2. Close ComfyUI and run `install_requirements.bat` using ComfyUI's Python.
3. Restart ComfyUI after installation completes.

## Resolutions and memory

The dropdown includes:

```text
512, 1024, 1536, 2048, 2560, 3072, 3584, 4096,
4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192
```

Exposing a resolution does not guarantee that the GPU can finish it. On an 8 GB
GPU, 512-2048 are the realistic range. 4K-8K may require substantially more
VRAM depending on surface area and mesh complexity.

## Source locations

- Wrapper supplied by the user: `nodes(2).py`, class
  `Trellis2ReconstructMeshWithQuad`, around lines 2300-2337.
- Floater removal helpers: `remove_floater2` around line 246 and
  `pymeshlab_remove_floater` around line 292.
- Actual reconstruction backend:
  https://github.com/visualbruno/CuMesh/blob/main/cumesh/remeshing.py
- Original ComfyUI wrapper:
  https://github.com/visualbruno/ComfyUI-Trellis2

## AMD Edition Changes - 2026-10-03

- Uses ComfyUI's Python and reports installation failures before restarting.
- Builds native HIP extensions for the active ROCm environment; matching HIP SDK and Visual Studio C++ Build Tools are required.
- Supports group installation through [ComfyUI-Easy-Install-AMD](https://github.com/BoomerCyb/ComfyUI-Easy-Install-AMD).

Original node by [Mstafa-awad / MostAadTech](https://github.com/Mstafa-awad). AMD fork maintained by [BoomerCyb](https://github.com/BoomerCyb). Original license and third-party credits are retained.
