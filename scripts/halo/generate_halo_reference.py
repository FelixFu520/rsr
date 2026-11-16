"""Generate new G1 reference candidates with local Kimodo checkpoints; no Kit or physics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import numpy as np

from real2sim2real.core.logger import get_logger

LOG = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Generate candidates and preserve inputs; physical acceptance is a separate step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--robot", choices=["g1"], default="g1")
    parser.add_argument("--model", default="Kimodo-G1-RP-v1")
    parser.add_argument("--models", type=Path, default=ROOT / "assets/Halo/models/kimodo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--constraints", type=Path, help="Kimodo constraints JSON in its Y-up frame"
    )
    parser.add_argument(
        "--origin", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X_M", "Y_M", "YAW_DEG")
    )
    args = parser.parse_args()
    if not 0 < args.duration <= 10 or args.samples < 1 or args.steps < 1:
        parser.error("duration must be in (0, 10] seconds; samples and steps must be positive")
    if not args.prompt.strip():
        parser.error("prompt must not be empty")
    output, models = args.output.resolve(), args.models.resolve()
    formal = (ROOT / "outputs/halo").resolve()
    if output == formal or formal in output.parents:
        parser.error("Unvalidated references must not be written to outputs/halo")
    scene = args.scene_config.resolve()
    scene_bytes = scene.read_bytes()
    model_manifest = models / "provenance.json"
    for name in (
        args.model,
        "Meta-Llama-3-8B-Instruct",
        "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    ):
        if not (models / name).is_dir():
            raise FileNotFoundError(models / name)
    if args.constraints:
        args.constraints.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CHECKPOINT_DIR=str(models),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TEXT_ENCODER_MODE="local",
    )
    try:
        import torch
        from kimodo import load_model
        from kimodo.constraints import load_constraints_lst
        from kimodo.exports.motion_io import save_kimodo_npz
        from kimodo.exports.mujoco import MujocoQposConverter
        from kimodo.model import LLM2VecEncoder
        from kimodo.skeleton import G1Skeleton34
        from kimodo.tools import seed_everything
        from peft import PeftModel
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError(
            "Install the Kimodo wheel and .[halo] dependencies; see docs/halo.md §6"
        ) from exc

    torch.set_num_threads(8)
    seed_everything(args.seed)
    LOG.info("kimodo_text_encoder_loading", device=args.device)
    # Load the base and adapters explicitly: historical adapter metadata points to the old
    # HALO project. No checkpoint config is edited and no old project path is followed.
    encoder = LLM2VecEncoder(
        str(models / "Meta-Llama-3-8B-Instruct"),
        None,
        dtype="bfloat16",
        llm_dim=4096,
        device=args.device,
    )
    encoder.model.model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    encoder.model.model = PeftModel.from_pretrained(
        encoder.model.model, str(models / "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
    ).merge_and_unload()
    encoder.model.model = PeftModel.from_pretrained(
        encoder.model.model, str(models / "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")
    )
    encoder.model.model.eval()
    features, lengths = encoder([args.prompt])
    features = features.detach().clone()
    torch.save(
        {"prompt": args.prompt, "features": features.cpu(), "lengths": lengths},
        output / "text_embedding.pt",
    )
    del encoder
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    class CachedEncoder:
        def __call__(self, texts):
            if any(text != args.prompt for text in texts):
                raise ValueError("Text embedding does not match the requested prompt")
            return features.repeat(len(texts), 1, 1), lengths * len(texts)

    LOG.info("kimodo_motion_model_loading", model=args.model)
    model = load_model(args.model, device=args.device, text_encoder=CachedEncoder())
    if not isinstance(model.skeleton, G1Skeleton34):
        raise ValueError("This exporter supports the G1 skeleton; other robots need retargeting")
    constraints = (
        load_constraints_lst(str(args.constraints), model.skeleton, device=args.device)
        if args.constraints
        else []
    )
    frames = int(round(args.duration * model.fps))
    LOG.info("kimodo_generating", frames=frames, samples=args.samples, steps=args.steps)
    generated = model(
        prompts=[args.prompt] * args.samples,
        num_frames=[frames] * args.samples,
        constraint_lst=constraints,
        num_denoising_steps=args.steps,
        return_numpy=True,
        post_processing=False,
    )
    converter = MujocoQposConverter(model.skeleton)
    qpos = np.asarray(
        converter.dict_to_qpos(generated, device="cpu", root_quat_w_first=True, numpy=True)
    )
    if qpos.ndim == 2:
        qpos = qpos[None]
    if qpos.shape != (args.samples, frames, 36) or not np.isfinite(qpos).all():
        raise RuntimeError(f"Invalid generated qpos: {qpos.shape}")
    rotation = Rotation.from_euler("z", args.origin[2], degrees=True)
    for index, candidate in enumerate(qpos):
        candidate[:, :3] = rotation.apply(candidate[:, :3])
        candidate[:, :2] += args.origin[:2]
        candidate[:, 3:7] = (rotation * Rotation.from_quat(candidate[:, [4, 5, 6, 3]])).as_quat()[
            :, [3, 0, 1, 2]
        ]
        target = output / f"sample_{index:03d}"
        target.mkdir()
        np.save(target / "qpos.npy", candidate)
        spec = {
            "fps": float(model.fps),
            "source": "Kimodo fresh generation",
            "phases": [{"name": "generated", "start": 0, "end": frames - 1}],
            "physics_validated": False,
        }
        (target / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    save_kimodo_npz(str(output / "motion.npz"), generated)
    shutil.copy2(scene, output / "scene_config.toml")
    if args.constraints:
        shutil.copy2(args.constraints, output / "constraints.json")
    manifest = {
        "prompt": args.prompt,
        "model": args.model,
        "robot": args.robot,
        "models": str(models),
        "model_manifest_sha256": hashlib.sha256(model_manifest.read_bytes()).hexdigest()
        if model_manifest.exists()
        else None,
        "scene_config": str(scene),
        "scene_config_sha256": hashlib.sha256(scene_bytes).hexdigest(),
        "origin_xy_yaw_deg": args.origin,
        "duration_s": frames / model.fps,
        "seed": args.seed,
        "samples": args.samples,
        "steps": args.steps,
        "device": args.device,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("kimodo", "torch", "transformers", "peft", "huggingface-hub")
        },
        "physics_validated": False,
        "notes": "Scene config is provenance only. Geometry must be encoded in constraints; "
        "motion.npz stays in Kimodo coordinates, qpos uses Z-up/metres/wxyz plus origin.",
    }
    (output / "generation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOG.info("kimodo_candidates_ready", output=str(output), shape=qpos.shape)


if __name__ == "__main__":
    main()
