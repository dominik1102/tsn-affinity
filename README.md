# clbench-dt

**clbench-dt** is a modular research codebase for **continual learning (CL)** benchmarks with **Decision Transformer** policies. It supports three benchmark families:

- **CartPole** (discrete control)
- **Atari** (discrete control, offline trajectories)
- **Panda** (continuous control, offline trajectories)

The repository is designed for controlled CL experiments, including:

- benchmark specification generation,
- continual training with multiple strategies,
- single-task reference baselines,
- expert dataset generation,
- run logging and post-hoc analysis.

## Features

- Unified continual-learning benchmark framework
- Offline **Decision Transformer** training for discrete and continuous control
- Built-in CL strategies, including:
  - **Naive**
  - **Cumulative Replay**
  - **EWC**
  - **SI**
  - **TSN**
  - **TSN Original Reuse**
  - **TSN Improved Reuse**
- Support for **single-task** baselines as upper/reference bounds
- Tools for generating and inspecting offline expert datasets
- Standard CL metrics: **ACC**, **BWT**, **Forgetting**, and optionally **FWT**

---

## Installation

Install the core dependencies:

```bash
pip install torch gymnasium gymnasium[atari] numpy
```

Additional benchmark-specific dependencies may be required:

- **Atari**: ALE ROMs
- **Panda**: Panda environment dependencies and offline Panda datasets

---

## Repository workflow

A typical workflow consists of four stages:

1. **Build a benchmark specification** (`.json` task list)
2. **Run a baseline** (random, single-task DT, or continual DT)
3. **Train/evaluate a continual method**
4. **Analyze the saved run directory**

---

## 1. Build benchmark specifications

### CartPole CL-7

```bash
python bin/clb-build.py \
  --benchmark cartpole \
  --kind cartpole-cl-7 \
  --seed 0 \
  --out specs_cp.json
```

### Atari CL benchmark

```bash
python bin/clb-build.py \
  --benchmark atari \
  --kind atari-cl-3 \
  --seed 0 \
  --out specs_atari.json
```

The resulting JSON file defines the ordered list of tasks used in continual training and evaluation.

---

## 2. Random baseline / sanity check

```bash
python bin/clb-run.py \
  --spec specs_cp.json \
  --episodes-eval 3 \
  --steps-per-task 1000
```

This produces a performance matrix `P[i,j]` and standard CL metrics.

---

## 3. Continual Decision Transformer training

### Available strategies

Depending on the runner and benchmark, the codebase supports:

- `naive`
- `cumulative`
- `ewc`
- `si`
- `tsn`
- `tsn_origin_reuse`
- `tsn_improved_reuse`

### CartPole / discrete control

```bash
python bin/clb-run-dt.py \
  --spec specs_cp.json \
  --strategy cumulative \
  --dataset-root <path-to-cartpole-expert-datasets> \
  --seq-len 20 \
  --steps-per-task 50000 \
  --episodes-eval 15 \
  --device cuda
```

### Atari / discrete offline DT

```bash
python bin/clb-run-dt.py \
  --spec configs/specs_atari_cl_5_minari_like.json \
  --dataset-root <path-to-atari-expert-datasets> \
  --strategy cumulative \
  --steps-per-task 20000 \
  --seq-len 20 \
  --episodes-eval 30 \
  --max-steps 27000 \
  --atari-env minari_like \
  --replay-check \
  --device cuda
```

### Panda / continuous offline DT

#### Naive

```bash
python bin/clb-run-dt-panda.py \
  --strategy naive \
  --steps-per-task 50000 \
  --episodes-eval 5
```

#### Cumulative replay

```bash
python bin/clb-run-dt-panda.py \
  --strategy cumulative \
  --steps-per-task 50000 \
  --episodes-eval 5
```

---

## 4. Single-task Decision Transformer baselines

Single-task scripts train a **separate DT per task**, without replay or continual updates. These runs are useful as reference/upper-bound baselines.

### CartPole / Atari single-task

```bash
python bin/clb-run-dt-cartpole-single.py \
  --spec configs/specs_cp.json \
  --dataset-root <path-to-cartpole-expert-datasets> \
  --seq-len 20 \
  --steps-per-task 50000 \
  --episodes-eval 5 \
  --device cuda
```

For Atari:

```bash
python bin/clb-run-dt-atari-single.py \
  --spec configs/specs_atari_cl_5_minari_like.json \
  --dataset-root <path-to-atari-expert-datasets> \
  --steps 20000 \
  --seq-len 20 \
  --batch-size 64 \
  --episodes-eval 30 \
  --max-ep-len 27000 \
  --device cuda \
  --debug-replay
```

### Panda single-task

```bash
python bin/clb-run-dt-panda-single.py \
  --datasets-root <path-to-panda-expert-datasets> \
  --seq-len 20 \
  --steps-per-task 50000 \
  --batch-size 64 \
  --episodes-eval 5 \
  --device cuda
```

---

## 5. Expert dataset generation

### CartPole expert datasets

This script trains a separate DQN expert per CartPole task and exports trajectories as `.npz` files.

```bash
python bin/train_cartpole_expert.py \
  --spec specs_cp.json \
  --episodes-per-task 200 \
  --max-len 500 \
  --total-steps-expert 50000 \
  --out-dir <path-to-output-cartpole-dataset> \
  --device cuda \
  --expert-action-prob 0.7
```

Each task directory contains `expert_trajs.npz` with:

- `observations`
- `actions`
- `rewards`
- `dones`
- `episode_lengths`

### Atari expert datasets

```bash
python bin/train_atari_expert.py \
  --spec specs_atari.json \
  --episodes-per-task 200 \
  --max-len 20000 \
  --total-steps-expert 5000000 \
  --out-dir <path-to-output-atari-dataset> \
  --device cuda \
  --expert-action-prob 1
```

### Export Atari trajectories from Minari to DT `.npz`

```bash
python bin/generate_atari_traj_from_mintari.py \
  --config configs/specs_atari_cl_5_minari_like.json \
  --out-root <path-to-atari-expert-datasets> \
  --max-len 50000 \
  --obs-dtype float32
```

---

## 6. Result analysis

Runs are saved under `runs/<timestamp>/<benchmark>/<strategy>/...`.

To analyze a saved run directory:

```bash
python analyze_runs.py --run-dir runs/<timestamp>/<benchmark>/<strategy>/<run-name>
```

The analysis script can be used to export summary CSV files and visualizations such as performance heatmaps.

---

## 7. Inspect offline Atari trajectories

```bash
python tools/inspect_atari_trajs.py \
  --root <path-to-atari-expert-datasets> \
  --tasks A_Pong B_Breakout C_Seaquest
```

---

## Important evaluation notes

### Standard CL metrics

The repository reports the following continual-learning metrics:

- **ACC** — final average performance across tasks
- **BWT** — backward transfer
- **Forgetting** — average performance drop on previously learned tasks
- **FWT** — forward transfer (when a suitable zero-shot baseline is provided)

### TSN-style methods

For TSN-based methods, `task_id` is part of evaluation.

That means:

- the **lower triangle** of the CL matrix is the primary object of interest,
- evaluating **future tasks before their masks exist** is usually not informative,
- task-aware evaluation should explicitly activate the correct task mask during testing.

### Atari dataset consistency

For offline Atari experiments, dataset-to-environment consistency matters. The following checks are recommended:

- use the same preprocessing as in dataset generation,
- verify frame stacking and reward clipping settings,
- run the built-in replay check when available.

---

## TSN Atari example

```bash
python bin/clb-run-dt.py \
  --strategy tsn \
  --spec configs/specs_atari_cl_5_minari_like.json \
  --dataset-root <path-to-atari-expert-datasets> \
  --atari-env minari_like \
  --seq-len 20 \
  --steps-per-task 2000 \
  --batch-size 64 \
  --target-mode max \
  --tsn-keep-ratio 0.5 \
  --tsn-quant-clusters 16 \
  --tsn-skip-module dt.te \
  --tsn-keep-schedule equal_remaining \
  --tsn-min-keep-ratio 1e-3 \
  --tsn-grad-clip 1.0
```

### TSN defaults that are typically stable for Atari

- `keep_ratio=0.5`
- `allow_weight_reuse=False`
- `include_embeddings=True`
- `skip_module_names=("dt.te",)`
- `freeze_non_mask_params_after_first=True`

Skipping `dt.te` is often a useful default because the time embedding is large and shared across tasks.

---

## Extending the framework

### Add a new benchmark

1. Implement an adapter in `clbench/adapters/`
2. Register it in `TaskRegistry`
3. Add benchmark presets in `clbench/benchmark/builder.py`

### Add a new strategy

1. Add a class in `strategies/`
2. Inherit from `BaseStrategy`
3. Implement `train_task()` and, if needed, `after_task()`

---

## Citation

If you use this repository in academic work, please cite the accompanying paper.

---

## License

Add the project license information here before release.
