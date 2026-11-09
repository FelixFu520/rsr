"""Create a separate V7 asset with conforming local contact-mesh subdivision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.package_refinement import map_material_probes, refine_surface
from scripts.prepare_package_assets import prepare


def refine(source: Path, target: Path, half_size_xy: tuple[float, float]) -> dict:
    """Copy dependencies and refine surfaces while preserving original loft anchors."""
    from pxr import Sdf, Usd, UsdGeom, Vt

    if source.resolve() == target.resolve():
        raise ValueError("Refinement must write a separate asset directory")
    source_file = source / "parcel_two_mesh.usd"
    before = hashlib.sha256(source_file.read_bytes()).hexdigest()
    prepare(source, target)
    output = target / "parcel_two_mesh.usd"
    stage = Usd.Stage.Open(str(output))
    root = stage.GetPrimAtPath("/World/Parcel")
    anchors = np.asarray(root.GetAttribute("parcel:loftSpringIndices").Get()).copy()
    records = {}
    previous_reference = root.GetAttribute("parcel:qualityReference")
    retain_reference = previous_reference and previous_reference.HasAuthoredValueOpinion()
    quality_reference = json.loads(previous_reference.Get()) if retain_reference else {}
    for name in ("Garment", "Bag"):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/Parcel/{name}Body/{name}"))
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int).reshape(-1, 3)
        mass_attr = mesh.GetPrim().GetAttribute("physics:masses")
        masses = np.asarray(mass_attr.Get(), dtype=float)
        if name == "Garment" and not retain_reference:
            quality_reference["garment_masses"] = masses.tolist()
        elif name == "Bag" and not retain_reference:
            quality_reference.update(bag_vertex_count=len(points), bag_triangles=triangles.tolist())
        refined = refine_surface(points, triangles, masses, half_size_xy=half_size_xy)
        if name == "Bag":
            support = np.asarray(quality_reference.get("bag_face_probe_indices", triangles))
            weights = np.asarray(
                quality_reference.get("bag_face_probe_weights", np.full((len(support), 3), 1 / 3))
            )
            probe_indices, probe_weights = map_material_probes(refined, triangles, support, weights)
            quality_reference.update(
                bag_face_probe_indices=probe_indices.tolist(),
                bag_face_probe_weights=probe_weights.tolist(),
            )
        for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
            if not primvar.GetAttr().HasAuthoredValueOpinion():
                continue
            interpolation = primvar.GetInterpolation()
            if interpolation == UsdGeom.Tokens.constant:
                continue
            original = primvar.ComputeFlattened()
            values = np.asarray(original)
            if interpolation == UsdGeom.Tokens.uniform:
                mapped = values[refined.parents]
            else:
                if interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
                    corners = values[triangles]
                elif interpolation == UsdGeom.Tokens.faceVarying:
                    corners = values.reshape(len(triangles), 3, *values.shape[1:])
                else:
                    raise ValueError(f"Unsupported primvar interpolation: {primvar.GetName()}")
                if values.dtype.kind != "f":
                    raise ValueError(f"Cannot interpolate categorical primvar: {primvar.GetName()}")
                mapped = np.einsum(
                    "fci,fi...->fc...", refined.barycentric, corners[refined.parents]
                )
                mapped = mapped.reshape(-1, *values.shape[1:])
                primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
            primvar.BlockIndices()
            primvar.GetAttr().Set(type(original).FromNumpy(mapped.astype(values.dtype)))
        for child in mesh.GetPrim().GetChildren():
            if child.IsA(UsdGeom.Subset):
                subset = UsdGeom.Subset(child)
                if subset.GetElementTypeAttr().Get() == UsdGeom.Tokens.face:
                    previous = np.asarray(subset.GetIndicesAttr().Get(), dtype=int)
                    subset.GetIndicesAttr().Set(
                        Vt.IntArray.FromNumpy(
                            np.flatnonzero(np.isin(refined.parents, previous)).astype(np.int32)
                        )
                    )
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(refined.vertices.astype(np.float32)))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(refined.triangles)))
        mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray.FromNumpy(refined.triangles.astype(np.int32).ravel())
        )
        mass_attr.Set(refined.masses.tolist())
        face = refined.vertices[refined.triangles]
        cross = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
        normals = np.zeros_like(refined.vertices)
        for corner in range(3):
            np.add.at(normals, refined.triangles[:, corner], cross)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        mesh.GetExtentAttr().Set(
            Vt.Vec3fArray.FromNumpy(
                np.array([refined.vertices.min(0), refined.vertices.max(0)], dtype=np.float32)
            )
        )
        edges = np.sort(
            np.concatenate(
                [refined.triangles[:, :2], refined.triangles[:, 1:], refined.triangles[:, [2, 0]]]
            ),
            axis=1,
        )
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        if name == "Bag" and not np.all(edge_counts == 2):
            raise ValueError("Refinement opened the closed bag surface")
        np.testing.assert_array_equal(refined.vertices[: len(points)], points)
        records[name] = dict(
            original_vertices=len(points),
            refined_vertices=len(refined.vertices),
            original_triangles=len(triangles),
            refined_triangles=len(refined.triangles),
            mass_kg=float(refined.masses.sum()),
            all_vertices_dynamic=bool(np.all(refined.masses > 0)),
            original_anchor_positions_preserved=True,
            bag_closed=bool(np.all(edge_counts == 2)) if name == "Bag" else None,
        )
    np.testing.assert_array_equal(
        anchors, np.asarray(root.GetAttribute("parcel:loftSpringIndices").Get())
    )
    root.CreateAttribute("parcel:qualityReference", Sdf.ValueTypeNames.String).Set(
        json.dumps(quality_reference)
    )
    stage.GetRootLayer().Save()
    reloaded = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(str(output)))
    roundtrip_reference = json.loads(
        reloaded.GetPrimAtPath("/World/Parcel").GetAttribute("parcel:qualityReference").Get()
    )
    if roundtrip_reference != quality_reference:
        raise ValueError("USD roundtrip changed the original quality probe population")
    for name, record in records.items():
        mesh = UsdGeom.Mesh(reloaded.GetPrimAtPath(f"/World/Parcel/{name}Body/{name}"))
        assert len(mesh.GetPointsAttr().Get()) == record["refined_vertices"]
        uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        if name == "Bag" and not uv:
            raise ValueError("Refined bag lost its photograph UV coordinates")
        if uv and len(uv.ComputeFlattened()) != 3 * record["refined_triangles"]:
            raise ValueError("USD roundtrip has incompatible UV coordinates")
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != before:
        raise RuntimeError("Refinement modified its source")
    provenance_file = target / "provenance.json"
    provenance = json.loads(provenance_file.read_text())
    provenance.update(
        topology_modified=True,
        geometry_modified=True,
        surface_preserved_by_midpoint_subdivision=True,
        materials_modified=False,
        loft_spring_anchors_modified=False,
        source_asset_sha256=before,
        half_size_xy_m=half_size_xy,
        refinement=records,
        source_validation_applies_to_original_asset_only=True,
        original_sampling_retained=True,
        usd_roundtrip_verified=True,
        reference_probe_counts=dict(
            garment_vertices=len(roundtrip_reference["garment_masses"]),
            bag_vertices=roundtrip_reference["bag_vertex_count"],
            bag_triangles=len(roundtrip_reference["bag_triangles"]),
        ),
    )
    for original, renamed in [
        ("README.md", "SOURCE_README.md"),
        ("usd_rerun_validation.json", "SOURCE_usd_rerun_validation.json"),
    ]:
        if (target / original).exists():
            (target / original).rename(target / renamed)
            provenance["sha256"][renamed] = provenance["sha256"].pop(original)
        elif (source / renamed).exists():
            shutil.copy2(source / renamed, target / renamed)
            provenance["sha256"][renamed] = hashlib.sha256(
                (target / renamed).read_bytes()
            ).hexdigest()
    provenance["sha256"]["parcel_two_mesh.usd"] = hashlib.sha256(output.read_bytes()).hexdigest()
    provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")
    get_logger(__name__).info(
        "Locally refined V7 asset prepared", target=str(target), parts=records
    )
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("assets/PhysicSolver/Package/v7_sorting")
    )
    parser.add_argument(
        "--target", type=Path, default=Path("assets/PhysicSolver/Package/v7_sorting_refined")
    )
    parser.add_argument("--half-width", type=float, default=0.085)
    parser.add_argument("--half-length", type=float, default=0.055)
    parser.add_argument("--usd-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.usd_worker:
        # Isaac's standalone Python needs the installed USD extension's Python
        # and shared-library paths; use the same interpreter, without starting Kit.
        root = Path(sys.executable).resolve().parents[3]
        libraries = sorted(root.glob("extscache/omni.usd.libs-*/pxr/Gf"))
        if not libraries:
            raise RuntimeError("Run this tool with the installed Isaac Sim python.sh")
        library = libraries[-1].parent.parent
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(library) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["LD_LIBRARY_PATH"] = (
            str(library / "bin") + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
        raise SystemExit(
            subprocess.call(
                [sys.executable, __file__, *sys.argv[1:], "--usd-worker"], env=environment
            )
        )
    refine(args.source, args.target, (args.half_width, args.half_length))
