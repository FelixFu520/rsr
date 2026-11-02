# 在 Isaac Sim 环境中安装 cuRobo

抓放规划器依赖 **cuRobo V2**（`curobo.motion_planner`）。必须用本仓库的
`./app/python.sh` 安装，与仿真共用同一套 Python / torch。

默认 extra `.[cu12]` 走 **cuda.core 运行时编译，不需要 `nvcc` / CUDA toolkit**。
不要 `apt install nvidia-cuda-toolkit`，也不要另开 conda / uv / 系统 Python。

Isaac standalone 已按 [`README.md`](../README.md) 装好，仓库根有 `app -> isaacsim`。

```bash
cd /path/to/Real2Sim2Real
# 必须是绝对路径：后面会 cd 到 ~/src/curobo，相对路径 ./app/python.sh 会失效
export ISAAC_PY="$(readlink -f ./app/python.sh)"

# 1. torch（Isaac 6.0.1 官方匹配：2.11.0 + CUDA 12.8）
$ISAAC_PY -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
$ISAAC_PY -c "import torch; print(torch.__version__, torch.version.cuda)"

# 2. 源码放仓库外（默认分支即 V2）
sudo apt install -y git git-lfs
git lfs install
git clone https://github.com/NVlabs/curobo.git ~/src/curobo
cd ~/src/curobo && git lfs pull

# 3. 打 wheel 再装进 Isaac（不要 .[cu12-torch]，会再下一套 torch）
#    --no-isolation 让构建看到上一步的 torch
$ISAAC_PY -m pip install -U setuptools setuptools_scm build wheel
$ISAAC_PY -m build --wheel --no-isolation
$ISAAC_PY -m pip install "$(ls dist/nvidia_curobo-*.whl)[cu12]"

# 4. 验证
cd /path/to/Real2Sim2Real
$ISAAC_PY -c "from curobo.motion_planner import MotionPlanner; print('curobo ok')"

# 5. 跑抓放（本包若未装： $ISAAC_PY -m pip install -e '.[dev]'）
./scripts/rsr collect run -c configs/arm_franka_curobo_pickplace/home000.toml
```

已有与当前 Isaac / Python 3.12 / torch 匹配的 `nvidia_curobo-*.whl` 时，跳过 2–3，改为：

```bash
$ISAAC_PY -m pip install /path/to/nvidia_curobo-*.whl
```

| 不要 | 原因 |
|------|------|
| `.[cu12-torch]` / `.[cu13-torch]` | 覆盖刚装的 Isaac torch |
| 官方 V1 的 `pip install -e ".[isaacsim]"` | V2 没有这个 extra |

上游：<https://github.com/NVlabs/curobo> · V2 文档：<https://nvlabs.github.io/curobo/latest/getting-started/installation.html>
