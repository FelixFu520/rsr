#!/usr/bin/env bash
# 完整执行双包裹物理、逐相机渲染和独立验收；可复用已通过的物理缓存。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if (( $# > 1 )); then
    echo "用法: $0 [输出目录]" >&2
    exit 2
fi
STAMP="$(TZ=America/New_York date +%Y%m%d_%H%M%S)"
OUTPUT="${1:-outputs/sorting_stream_showcase_1k_$STAMP}"
CONFIG="configs/PhysicSolver_mujoco_vbd_sorting_stream_showcase.toml"
mkdir -p "$OUTPUT/logs"
LOG="$OUTPUT/logs/pipeline_$STAMP.log"

run_logged() {
    "$@" 2>&1 | tee -a "$LOG"
}

ARGS=(collect render-sorting -c "$CONFIG" --output "$OUTPUT")
if [[ -d "$OUTPUT/physics" ]]; then
    if ! ./app/python.sh - "$OUTPUT/physics/result.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text()) if path.is_file() else {}
accepted = (
    result.get("success")
    and result.get("strict_quality_success")
    and result.get("completed")
    and result.get("continuous_world")
    and result.get("accepted_demonstration", True)
    and not result.get("diagnostic_restart", False)
    and result.get("gripper_geometry")
    == "authored_ProM6_Robotiq_with_visual_matched_27x7.5x65mm_pads"
)
raise SystemExit(0 if accepted else 1)
PY
    then
        echo "已有 physics/ 未通过胶垫对齐与严格间隙验收，请选择新的输出目录重算。旧数据已保留。" >&2
        exit 1
    fi
    ARGS+=(--source "$OUTPUT/physics")
fi

if [[ ! -s "$OUTPUT/result.json" ]]; then
    run_logged ./scripts/rsr "${ARGS[@]}"
fi
run_logged ./app/python.sh scripts/validate_sorting_stream.py "$OUTPUT"
printf '\n执行与独立验收通过。输出: %s\n日志: %s\n' "$OUTPUT" "$LOG"
