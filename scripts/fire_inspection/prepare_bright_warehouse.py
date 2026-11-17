"""Copy the demo warehouse and brighten only its ordinary overhead lighting.

Run with Isaac python.sh. All other lights retain the effective intensity of the
2K demo (authored intensity times 1.5). The new configuration uses light_scale=1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from real2sim2real.core.logger import get_logger
from scripts.fire_inspection.prepare_fire_inspection_assets import _import_pxr

log = get_logger(__name__)
ROOT = Path("assets/FireInspection")
SOURCE = ROOT / "warehouse/warehouse_cardbox_fire.usda"
DESTINATION = ROOT / "warehouse/warehouse_cardbox_fire_bright.usda"
REPORT = ROOT / "bright_warehouse_provenance.json"


def prepare(
    multiplier: float = 3.0,
    destination: Path = DESTINATION,
    report_path: Path = REPORT,
    extend_fire_animation_s: float = 0.0,
) -> None:
    """Create an independent root layer, audit light changes and preserve source assets."""
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Light multiplier must be finite and positive")
    if not math.isfinite(extend_fire_animation_s) or extend_fire_animation_s < 0:
        raise ValueError("Fire animation duration must be finite and nonnegative")
    if destination.parent.resolve() != SOURCE.parent.resolve():
        raise ValueError("Keep the scene copy beside the source to preserve relative dependencies")
    if not report_path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Keep provenance inside assets/FireInspection")
    pxr, app = _import_pxr()
    exit_code = 0
    try:
        from pxr import UsdGeom, UsdLux

        if destination.exists() or report_path.exists():
            raise FileExistsError("Bright warehouse already exists; refusing to overwrite")
        source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        layer = pxr.Sdf.Layer.FindOrOpen(str(SOURCE.resolve()))
        if not layer.Export(str(destination.resolve())):
            raise RuntimeError(f"Cannot copy {SOURCE}")
        stage = pxr.Usd.Stage.Open(str(destination.resolve()))
        changes = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdLux.LightAPI):
                continue
            light = UsdLux.LightAPI(prim)
            path = str(prim.GetPath())
            color = tuple(light.GetColorAttr().Get())
            position = (
                UsdGeom.Xformable(prim)
                .ComputeLocalToWorldTransform(pxr.Usd.TimeCode.Default())
                .ExtractTranslation()
            )
            # Restrict to known ceiling-lamp groups and high-mounted white lamps.
            ordinary = (
                (
                    path.startswith("/Root/Lighting/Lamp")
                    or path.startswith("/Root/Room_03/Lamp_Lighting_Room_03/SpotLight")
                    or (prim.GetParent().GetPath() == pxr.Sdf.Path("/Root"))
                )
                and position[2] >= 3.0
                and min(color) / max(color) >= 0.4
            )
            authored = float(light.GetIntensityAttr().Get())
            reference = authored * 1.5
            intensity = reference * (multiplier if ordinary else 1.0)
            light.GetIntensityAttr().Set(intensity)
            changes.append(
                {
                    "path": path,
                    "ordinary_overhead": ordinary,
                    "source_intensity": authored,
                    "reference_demo_intensity": reference,
                    "intensity": intensity,
                    "color": color,
                }
            )
        animations = []
        if extend_fire_animation_s:
            end_code = extend_fire_animation_s * stage.GetTimeCodesPerSecond()
            for prim in stage.Traverse():
                if not str(prim.GetPath()).startswith("/FireFX/") or not str(
                    prim.GetTypeName()
                ).startswith("FlowEmitter"):
                    continue
                for attr in prim.GetAttributes():
                    times = attr.GetTimeSamples()
                    if len(times) < 2:
                        continue
                    period = times[-1] - times[0]
                    values = [attr.Get(t) for t in times]
                    # Preserve the first authored cycle, then repeat its forcing.
                    # Keep each boundary's original final value; USD interpolates
                    # to the next cycle's first nonzero sample without resetting Flow.
                    for cycle in range(1, math.ceil(end_code / period)):
                        for code, value in zip(times[1:], values[1:], strict=True):
                            attr.Set(value, code + cycle * period)
                    animations.append(
                        {
                            "attribute": str(attr.GetPath()),
                            "period_s": period / stage.GetTimeCodesPerSecond(),
                            "original_last_time_code": times[-1],
                            "extended_last_time_code": attr.GetTimeSamples()[-1],
                        }
                    )
            stage.SetEndTimeCode(max(stage.GetEndTimeCode(), end_code))
        stage.GetRootLayer().Save()
        used = [
            str(Path(layer.realPath).resolve()) for layer in stage.GetUsedLayers() if layer.realPath
        ]
        if any(not Path(path).is_relative_to(ROOT.resolve()) for path in used):
            raise RuntimeError("Scene contains a layer outside assets/FireInspection")
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
        report = {
            "source": str(SOURCE),
            "source_sha256": source_hash,
            "destination": str(destination),
            "destination_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "reference_config": "configs/fire_inspection/x30_physx_warehouse_demo_2k.toml",
            "ordinary_multiplier_vs_demo": multiplier,
            "other_lights_multiplier_vs_demo": 1.0,
            "used_layers": used,
            "lights": changes,
            "source_and_license": "assets/FireInspection/provenance.json",
            "extended_fire_animation_s": extend_fire_animation_s,
            "fire_emitter_animations": animations,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        log.info(
            "Bright warehouse copied",
            lights=len(changes),
            brightened=sum(row["ordinary_overhead"] for row in changes),
        )
    except Exception:
        exit_code = 1
        log.exception("Bright warehouse preparation failed")
        raise
    finally:
        if app is not None:
            app.close(exit_code=exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--extend-fire-animation-s", type=float, default=0.0)
    args = parser.parse_args()
    prepare(args.multiplier, args.destination, args.report, args.extend_fire_animation_s)
