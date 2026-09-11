# ComfyUI Mesh Quad Reconstruction

Standalone adaptation of VisualBruno's **Trellis2 - Reconstruct Mesh With Quad**
node for ComfyUI's current native mesh type:

## 🚀 SUPPORT MOSTAADTECH

### ❤️ Enjoying this project / workflow?

I’m **MostAadTech**, I create FREE ComfyUI workflows, local AI tools, 3D pipelines, and open-source projects.

If this project or workflow helped you, **please consider following me or supporting my work**. It helps me keep building, testing, and releasing more free tools and workflows.

---

## 💜 Support Me on Patreon

👉 **[Support MostAadTech on Patreon](https://www.patreon.com/cw/MostafaAwad/membership)**

Your support helps me spend more time developing **FREE AI tools, ComfyUI workflows, and 3D pipelines**.

---

## 🌐 Follow MostAadTech

* ▶️ **[YouTube](https://www.youtube.com/@MostAadTech)** — Tutorials, workflows & AI projects
* 📸 **[Instagram](https://www.instagram.com/mostaadtech/)** — Projects, updates & behind the scenes
* 𝕏 **[X / Twitter](https://x.com/MostAadTech)** — Updates, releases & experiments
* 💻 **[GitHub](https://github.com/Mstafa-awad)** — Open-source projects & code

---

### ⭐ One Follow Helps

**Follow • Star • Share • Support**

Every follow, GitHub star, share, and Patreon supporter helps me continue making **FREE tools for the AI community.**

**Thank you for supporting MostAadTech! ❤️**


```text
MESH -> MESH
```

It has no dependency on CelloCut or Trellis `MESHWITHVOXEL` objects. It calls the
same CUDA function used by VisualBruno:

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
| `remove_floaters` | `true` | Uses the same VisualBruno/PyMeshLab 0.5% face-component rule on CUDA. |
| `remove_inner_faces` | `false` | Requests CuMesh's inner-surface filtering. It can create holes on difficult meshes. |

Reconstruction creates new topology, so UVs, materials, textures, colors, and
custom normals are not preserved.

### Faster floater removal

The default floater stage now stays on CUDA. It reproduces both parts of
VisualBruno's PyMeshLab operation: select face-connected components below 0.5%
of the largest component (including MeshLab's integer-threshold truncation),
then remove every vertex touched by that selection and all incident faces. Work
is chunked in 1,048,576-face blocks to keep temporary allocations bounded.

The node unloads inactive ComfyUI models before reconstruction. If the CUDA
connectivity stage still cannot allocate enough VRAM, the error is caught and
the node automatically uses the original PyMeshLab implementation. This keeps
the quality rule unchanged instead of crashing or silently disabling floater
removal.

## Installation

1. Extract `ComfyUI-Mesh-Quad-Reconstruct` into `ComfyUI/custom_nodes/`.
2. If VisualBruno's `ComfyUI-Trellis2` already works, restart ComfyUI. Its
   installed CuMesh backend will be reused automatically.
3. Otherwise run `Install-Windows.cmd`, then restart ComfyUI.

The manual installer first searches a sibling `ComfyUI-Trellis2/wheels` folder
for a wheel matching ComfyUI's Python and PyTorch. If none exists, it builds
VisualBruno/CuMesh from source, which requires Visual Studio C++ Build Tools and
a compatible CUDA Toolkit.

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
