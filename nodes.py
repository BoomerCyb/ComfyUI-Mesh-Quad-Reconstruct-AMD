from __future__ import annotations

import numpy as np
import torch

from comfy_api.latest import Types

# Every 512-step resolution requested, from 512 through 8K.
RESOLUTIONS = list(range(512, 8193, 512))
FLOATER_FACE_RATIO = 0.005
CUDA_FILTER_CHUNK_FACES = 1_048_576


def _require_cuda() -> None:
    """Guard only: the CUDA backend is mandatory. No memory cleaning here."""
    if not torch.cuda.is_available():
        raise RuntimeError("Quad Reconstruction requires an available HIP/ROCm or CUDA GPU.")


def _mesh_batch_items(mesh):
    """Extract unpadded items from ComfyUI's native batched MESH type."""
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError(
            "Expected ComfyUI's native MESH object with vertices and faces; "
            f"received {type(mesh).__name__}."
        )
    vertices = mesh.vertices
    faces = mesh.faces
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)
    if faces.ndim == 2:
        faces = faces.unsqueeze(0)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"MESH vertices must be [B,N,3], got {tuple(vertices.shape)}.")
    if faces.ndim != 3 or faces.shape[-1] != 3:
        raise ValueError(f"MESH faces must be [B,F,3], got {tuple(faces.shape)}.")
    if vertices.shape[0] != faces.shape[0]:
        raise ValueError("MESH vertices and faces have different batch sizes.")
    vertex_counts = getattr(mesh, "vertex_counts", None)
    face_counts = getattr(mesh, "face_counts", None)
    items = []
    for index in range(vertices.shape[0]):
        vertex_count = (
            int(vertex_counts[index].item())
            if vertex_counts is not None
            else int(vertices.shape[1])
        )
        face_count = (
            int(face_counts[index].item())
            if face_counts is not None
            else int(faces.shape[1])
        )
        if vertex_count <= 0 or face_count <= 0:
            raise ValueError(f"MESH batch item {index} is empty.")
        items.append(
            (
                vertices[index, :vertex_count].detach(),
                faces[index, :face_count].detach(),
            )
        )
    return items


def _pack_mesh_batch(items):
    """Pack reconstructed tensors back into ComfyUI's native MESH type."""
    vertices = [
        item_vertices.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for item_vertices, _ in items
    ]
    faces = [
        item_faces.detach().to(device="cpu", dtype=torch.int64).contiguous()
        for _, item_faces in items
    ]
    if not vertices or any(len(v) == 0 or len(f) == 0 for v, f in zip(vertices, faces)):
        raise RuntimeError("Quad Reconstruction produced an empty mesh.")
    if len(vertices) == 1:
        return Types.MESH(vertices=vertices[0].unsqueeze(0), faces=faces[0].unsqueeze(0))
    vertex_counts = torch.tensor([len(v) for v in vertices], dtype=torch.int64)
    face_counts = torch.tensor([len(f) for f in faces], dtype=torch.int64)
    packed_vertices = torch.zeros(
        (len(vertices), int(vertex_counts.max().item()), 3), dtype=torch.float32
    )
    packed_faces = torch.zeros(
        (len(faces), int(face_counts.max().item()), 3), dtype=torch.int64
    )
    for index, (item_vertices, item_faces) in enumerate(zip(vertices, faces)):
        packed_vertices[index, : len(item_vertices)] = item_vertices
        packed_faces[index, : len(item_faces)] = item_faces
    return Types.MESH(
        vertices=packed_vertices,
        faces=packed_faces,
        vertex_counts=vertex_counts,
        face_counts=face_counts,
    )


def _remove_floaters_visualbruno(vertices: np.ndarray, faces: np.ndarray):
    """Original PyMeshLab filter, retained as a compatibility/OOM fallback."""
    try:
        import pymeshlab
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "remove_floaters requires PyMeshLab. Run install.py with ComfyUI's Python "
            "or install it using: python -m pip install pymeshlab"
        ) from exc
    print(f"[Mesh Quad] Removing floaters from {len(faces):,} faces...", flush=True)
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces),
        "quad_reconstructed_mesh",
    )
    mesh_set.apply_filter(
        "compute_selection_by_small_disconnected_components_per_face",
        nbfaceratio=0.005,
    )
    mesh_set.apply_filter(
        "compute_selection_transfer_face_to_vertex",
        inclusive=False,
    )
    mesh_set.apply_filter("meshing_remove_selected_vertices_and_faces")
    output = mesh_set.current_mesh()
    out_vertices = output.vertex_matrix().copy()
    out_faces = output.face_matrix().copy()
    print(f"[Mesh Quad] After removing floaters: {len(out_faces):,} faces.", flush=True)
    return out_vertices, out_faces


def _visualbruno_keep_mask(
    num_vertices: int,
    faces: torch.Tensor,
    component_ids: torch.Tensor,
    num_components: int,
    ratio: float = FLOATER_FACE_RATIO,
    chunk_faces: int = CUDA_FILTER_CHUNK_FACES,
):
    """Reproduce VisualBruno/PyMeshLab's face->vertex removal rule in chunks.

    PyMeshLab first selects face-connected components below ``ratio`` of the
    largest component. It then transfers that selection from faces to vertices
    with ``inclusive=False`` and removes selected vertices and all incident
    faces. Keeping this second step matters for components that touch only at a
    non-manifold vertex.
    """
    face_count = int(faces.shape[0])
    if face_count != int(component_ids.shape[0]):
        raise ValueError("Connected-component IDs do not match the face count.")
    if face_count == 0 or num_components <= 1:
        return torch.ones(face_count, dtype=torch.bool, device=faces.device), 0, 0
    chunk_faces = max(1, int(chunk_faces))
    counts = torch.zeros(
        int(num_components), dtype=torch.int64, device=component_ids.device
    )
    for start in range(0, face_count, chunk_faces):
        stop = min(start + chunk_faces, face_count)
        ids = component_ids[start:stop].to(dtype=torch.int64)
        counts.add_(torch.bincount(ids, minlength=int(num_components)))
    largest_faces = int(counts.max().item())
    # VCG stores ``largest * ratio`` in an unsigned integer, truncating the
    # threshold before applying a strict less-than comparison.
    threshold_faces = int(largest_faces * float(ratio))
    small_components = counts < threshold_faces
    small_component_count = int(small_components.sum().item())
    if small_component_count == 0:
        return (
            torch.ones(face_count, dtype=torch.bool, device=faces.device),
            0,
            largest_faces,
        )
    selected_vertices = torch.zeros(
        int(num_vertices), dtype=torch.bool, device=faces.device
    )
    for start in range(0, face_count, chunk_faces):
        stop = min(start + chunk_faces, face_count)
        ids = component_ids[start:stop].to(dtype=torch.int64)
        selected_faces = small_components[ids]
        if bool(selected_faces.any().item()):
            selected_indices = (
                faces[start:stop][selected_faces].reshape(-1).to(dtype=torch.int64)
            )
            selected_vertices.index_fill_(0, selected_indices, True)
    keep_mask = torch.empty(face_count, dtype=torch.bool, device=faces.device)
    for start in range(0, face_count, chunk_faces):
        stop = min(start + chunk_faces, face_count)
        face_indices = faces[start:stop].to(dtype=torch.int64)
        keep_mask[start:stop] = ~selected_vertices[face_indices].any(dim=1)
    return keep_mask.contiguous(), small_component_count, largest_faces


def _remove_floaters_cuda(cumesh_module, vertices, faces):
    """Run the VisualBruno 0.5% floater rule on CUDA using bounded tensors."""
    if not hasattr(cumesh_module, "CuMesh"):
        raise RuntimeError("The installed CuMesh build does not expose cumesh.CuMesh.")
    original_face_count = int(faces.shape[0])
    print(
        f"[Mesh Quad] Removing floaters on GPU from {original_face_count:,} faces...",
        flush=True,
    )
    cuda_mesh = cumesh_module.CuMesh()
    try:
        # CuMesh owns a private CUDA copy. The reconstruction tensors remain
        # untouched, allowing the exact CPU fallback if CUDA connectivity OOMs.
        cuda_mesh.init(vertices.contiguous(), faces.contiguous())
        cuda_mesh.get_connected_components()
        num_components, component_ids = cuda_mesh.read_connected_components()
        # CuMesh releases its own internal edge/adjacency buffers here; this is
        # object lifecycle, not a global memory cleaner.
        cuda_mesh.clear_cache()
        keep_mask, removed_components, largest_faces = _visualbruno_keep_mask(
            num_vertices=int(vertices.shape[0]),
            faces=faces,
            component_ids=component_ids,
            num_components=int(num_components),
        )
        kept_faces = int(keep_mask.sum().item())
        del component_ids
        if removed_components == 0:
            print(
                f"[Mesh Quad] GPU floater filter: {int(num_components):,} component(s), "
                "nothing below the 0.5% threshold.",
                flush=True,
            )
            return vertices, faces
        print(
            f"[Mesh Quad] GPU floater filter: {int(num_components):,} components; "
            f"largest={largest_faces:,} faces; removing {removed_components:,} small "
            f"component(s) and {original_face_count - kept_faces:,} incident faces...",
            flush=True,
        )
        # CuMesh's remove_faces uses True as KEEP and compacts unreferenced
        # vertices, matching PyMeshLab's final removal/compaction stage.
        cuda_mesh.remove_faces(keep_mask)
        del keep_mask
        filtered_vertices, filtered_faces = cuda_mesh.read()
        print(
            f"[Mesh Quad] After GPU floater removal: {len(filtered_faces):,} faces.",
            flush=True,
        )
        return filtered_vertices, filtered_faces
    finally:
        del cuda_mesh


def _is_cuda_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    oom_type = getattr(torch, "OutOfMemoryError", None)
    is_torch_oom = oom_type is not None and isinstance(exc, oom_type)
    return is_torch_oom or any(
        marker in message
        for marker in (
            "out of memory",
            "cudaerrormemoryallocation",
            "cuda error: memory allocation",
            "cuda malloc",
        )
    )


def _remove_floaters_fast(cumesh_module, vertices, faces):
    """Prefer exact CUDA filtering; safely fall back to the original CPU path."""
    try:
        return _remove_floaters_cuda(cumesh_module, vertices, faces)
    except Exception as exc:
        reason = "insufficient VRAM" if _is_cuda_memory_error(exc) else str(exc)
        print(
            f"[Mesh Quad] CUDA floater filter unavailable ({reason}). "
            "Falling back to VisualBruno's exact PyMeshLab filter.",
            flush=True,
        )
        cpu_vertices, cpu_faces = _remove_floaters_visualbruno(
            vertices.detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
        )
        return (
            torch.from_numpy(cpu_vertices).contiguous().float(),
            torch.from_numpy(cpu_faces).contiguous().long(),
        )


def _load_cumesh():
    try:
        import cumesh
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The CuMesh HIP backend is missing. Run install_requirements.bat "
            "using the Python environment that runs ComfyUI."
        ) from exc
    reconstruct = getattr(cumesh.remeshing, "reconstruct_mesh_dc_quad", None)
    if reconstruct is None:
        raise RuntimeError(
            "The installed CuMesh is not VisualBruno's build: "
            "cumesh.remeshing.reconstruct_mesh_dc_quad is missing. "
            "Run install_requirements.bat to build the bundled CuMesh HIP backend."
        )
    return cumesh, reconstruct


def _reconstruct_item(
    source_vertices,
    source_faces,
    remesh_band,
    resolution,
    remove_floaters,
    remove_inner_faces,
):
    cumesh_module, reconstruct = _load_cumesh()
    vertices = source_vertices.to(
        device="cuda", dtype=torch.float32, non_blocking=True
    ).contiguous()
    faces = source_faces.to(
        device="cuda", dtype=torch.int32, non_blocking=True
    ).contiguous()
    if not torch.isfinite(vertices).all():
        raise ValueError("MESH contains NaN or infinite vertices.")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("MESH is empty.")
    if faces.min().item() < 0 or faces.max().item() >= len(vertices):
        raise ValueError("MESH contains out-of-range face indices.")
    resolution = int(resolution)
    if resolution > 2048:
        print(
            f"[Mesh Quad] WARNING: {resolution} resolution is experimental and may "
            "require far more than 8 GB VRAM.",
            flush=True,
        )
    print(
        f"[Mesh Quad] Reconstructing {len(vertices):,} vertices / {len(faces):,} faces "
        f"at {resolution} (band={float(remesh_band):g}, "
        f"remove_inner_faces={bool(remove_inner_faces)})...",
        flush=True,
    )
    # Same VisualBruno algorithm. Unlike the original wrapper, band is passed so
    # the visible remesh_band control genuinely works. Default 1.0 is identical.
    with torch.inference_mode():
        vertices, faces = reconstruct(
            vertices,
            faces,
            resolution,
            band=float(remesh_band),
            verbose=True,
            remove_inner_faces=bool(remove_inner_faces),
        )
    if remove_floaters:
        with torch.inference_mode():
            vertices, faces = _remove_floaters_fast(cumesh_module, vertices, faces)
    vertices = vertices.detach().to(device="cpu", dtype=torch.float32).contiguous()
    faces = faces.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("Quad Reconstruction produced an empty mesh.")
    print(
        f"[Mesh Quad] Complete: {len(vertices):,} vertices / {len(faces):,} faces.",
        flush=True,
    )
    return vertices, faces


class MeshReconstructWithQuad:
    """VisualBruno Quad Reconstruction adapted from MESHWITHVOXEL to MESH."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "remesh_band": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "resolution": (RESOLUTIONS, {"default": 512}),
                "remove_floaters": ("BOOLEAN", {"default": True}),
                "remove_inner_faces": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Mesh/Quad Reconstruction"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "VisualBruno/CuMesh quad dual-contouring reconstruction adapted to native "
        "ComfyUI MESH input and output. Rebuilds topology and removes UVs, textures, "
        "materials, colors, and custom normals."
    )

    def process(
        self,
        mesh,
        remesh_band,
        resolution,
        remove_floaters,
        remove_inner_faces,
    ):
        _require_cuda()
        output_items = []
        for vertices, faces in _mesh_batch_items(mesh):
            output_items.append(
                _reconstruct_item(
                    vertices,
                    faces,
                    remesh_band=remesh_band,
                    resolution=resolution,
                    remove_floaters=remove_floaters,
                    remove_inner_faces=remove_inner_faces,
                )
            )
        return (_pack_mesh_batch(output_items),)


NODE_CLASS_MAPPINGS = {
    "MeshReconstructWithQuad": MeshReconstructWithQuad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshReconstructWithQuad": "Mesh - Reconstruct With Quad (512-8K)",
}