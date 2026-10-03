# Third-party notices

This wrapper adapts the `Trellis2ReconstructMeshWithQuad` node from
VisualBruno's MIT-licensed ComfyUI-Trellis2 project to ComfyUI's native `MESH`
type.

- ComfyUI-Trellis2: https://github.com/visualbruno/ComfyUI-Trellis2 — MIT
- VisualBruno/CuMesh: https://github.com/visualbruno/CuMesh — MIT
- PyMeshLab: https://github.com/cnr-isti-vclab/PyMeshLab — GPL-3.0

Modified CuMesh source is bundled in `CuMesh-HIP/`. PyMeshLab remains an external runtime dependency.


## Bundled HIP backend

CuMesh sources originate from VisualBruno/CuMesh commit d10e54c30ddd03d11472c1431693f985501c7966. The backend retains its MIT license, third-party Eigen, cubvh and xatlas source notices, and AMD hipCUB, rocPRIM and rocThrust headers from rocm-7.0.0 with their licenses. These components retain their own terms; see the license files under CuMesh-HIP/.
