# AMD changes

CuMesh uses HIP with hipCUB/rocPRIM, corrected sorting/header compatibility,
and bounded or flat memory transfers for large meshes. The native backend is
shared with CuMesh Decimate.

Installation uses ComfyUI's ROCm Python through install.py. Native extensions
use PyTorch HIP device properties to target discrete GPUs by default, while
respecting PYTORCH_ROCM_ARCH. Integrated-only systems target their available GPUs.
Tested on RX 9070 XT; other supported ROCm devices require validation.
Original licenses, algorithms and node registrations are preserved.
