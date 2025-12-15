# clbench-dt — Continual Learning Benchmarks + Decision Transformer (Full Project)

Modular framework to build/run continual learning (CL) benchmarks (CartPole, Atari, Panda)
and train a Decision Transformer with strategies: **Naive**, **Cumulative (replay)**, and **EWC**.

---

## 📦 Installation

Required packages:

```bash
pip install torch gymnasium gymnasium[atari] numpy
```

(For Atari you also need ALE ROMs.)

---

## 🚀 How to run

### 1. Building benchmarks (JSON specs)

CartPole CL-7:
```bash
python bin/clb-build.py --benchmark cartpole --kind cartpole-cl-7 --seed 0 --out specs_cp.json
```

Atari CL-3 (Pong → Breakout → Seaquest):
```bash
python bin/clb-build.py --benchmark atari --kind atari-cl-3 --seed 0 --out specs_atari.json
```

---

### 2. Random baseline (sanity check)

```bash
python bin/clb-run.py --spec specs_cp.json --episodes-eval 3 --steps-per-task 1000
```

Output: performance matrix **P[i,j]** + CL metrics **ACC, BWT, Forgetting** (and optionally **FWT**).

---

### 3. Decision Transformer with a chosen CL strategy

Available strategies: `naive`, `cumulative`, `ewc`.

CartPole:
```bash
python bin/clb-run-dt.py --spec specs_cp.json --strategy cumulative --dataset-root data/cartpole_expert  --seq-len 20 --steps-per-task 50000 --episodes-eval 15 --device cuda
```

Atari:
```bash
python bin/clb-run-dt.py   --spec specs_atari.json   --strategy cumulative   --steps-per-task 20000   --episodes-eval 10   --device cuda
```

Panda (offline, PandaReach → PandaPush → PandaPickAndPlace):  
(requires datasets in `resources/datasets/*panda*_1m_expert.pkl`)

```bash
# naive
python bin/clb-run-dt-panda.py   --strategy naive   --steps-per-task 50000   --episodes-eval 5

# cumulative replay
python bin/clb-run-dt-panda.py   --strategy cumulative   --steps-per-task 50000   --episodes-eval 5
```

---

### 4. Single-task Decision Transformer (no continual baseline)

These scripts train a **separate DT for each task independently** (no replay, no CL),
and report per-task performance. Useful as an “upper bound” / reference for CL runs.

#### CartPole / Atari (discrete)

```bash
# Single-task DT for each task in the spec (works for CartPole or Atari)
python bin/clb-run-dt-cartpole-single.py --spec specs_cp.json --dataset-root data/cartpole_expert --seq-len 20 --steps-per-task 50000 --episodes-eval 5 --device cuda
```

For Atari just change the spec:

```bash
python bin/clb-run-dt-atari-single.py   --spec specs_atari.json   --steps 20000   --max-ep-len 1000   --seq-len 20   --collect-episodes 5   --episodes-eval 10   --device cuda
```

#### Panda (continuous, offline datasets)

```bash
python bin/clb-run-dt-panda-single.py   --datasets-root resources/datasets   --seq-len 20   --steps-per-task 50000   --batch-size 64   --episodes-eval 5   --device cuda
```

`clb-run-dt-panda-single.py` trains one PandaDecisionTransformer per task
(PandaReach, PandaPush, PandaPickAndPlace) only on its own offline dataset
and then evaluates it on the corresponding `panda_gym` environment.

---

### 5. CartPole DQN expert (offline dataset)

This script trains a **separate DQN expert** for each CartPole task specified in a
JSON spec file (e.g. CL-7 CartPole), then collects expert trajectories and saves
them to disk as `.npz` files. These datasets can later be used as offline data
for Decision Transformer training (e.g., instead of random / on‑policy data).

Train experts and generate datasets:

```bash
python bin/train_cartpole_expert.py   --spec specs_cp.json   --episodes-per-task 200   --max-len 500   --total-steps-expert 50000   --out-dir data/cartpole_expert --device cuda   --expert-action-prob 0.7
```

This will create a directory structure like:

```text
data/cartpole_expert/
  A_default/
    expert_trajs.npz
  B_heavier_pole/
    expert_trajs.npz
  C_stronger_gravity/
    expert_trajs.npz
  D_longer_pole/
    expert_trajs.npz
  E_weaker_force/
    expert_trajs.npz
  F_faster_dynamics/
    expert_trajs.npz
  G_combo/
    expert_trajs.npz
```

Each `expert_trajs.npz` contains:

- `observations`:    `[N, obs_dim]`
- `actions`:         `[N]`
- `rewards`:         `[N]`
- `dones`:           `[N]`
- `episode_lengths`: `[n_episodes]`

You can reconstruct episode boundaries using `episode_lengths` and feed these
trajectories into your DT training pipeline (similar to existing robot datasets).

---


```bash
python bin/train_atari_expert.py   --spec specs_atari.json   --episodes-per-task 200   --max-len 1000   --total-steps-expert 5000000  --out-dir data/atari_expert   --device cuda   --expert-action-prob 1
```


### 7. Additional parameters

Common flags used in DT scripts:

- `--seq-len` — sequence length (default: 20),
- `--warm-episodes` — number of episodes for initial bootstrap of DT (random trajectories),
- `--collect-episodes` — number of on‑policy episodes collected per task,
- `--device` — `cpu` / `cuda` (defaults to `cuda` if available).

---

## 📊 Continual Learning metrics

- **ACC** – final average performance over all tasks,
- **BWT** – backward transfer,
- **Forgetting** – average performance drop on past tasks,
- **FWT** – forward transfer (if a zero‑shot baseline is provided).

---

## 🔧 Extensibility

- **New benchmarks**:  
  Add an adapter in `clbench/adapters/`, register it in `TaskRegistry`, and define presets in `clbench/benchmark/builder.py`.

- **New strategies**:  
  Add a class in `strategies/`, inherit from `BaseStrategy`, and implement `train_task()` and optionally `after_task()`.

---

## 🔍 Result analysis

Analysis (heatmaps + CSV) with `analyze_runs.py` on a specific run directory, e.g.:

CartPole / Atari:
```bash
python analyze_runs.py --run-dir runs/20251128-152557/atari/cumulative/specs_atari
```

Panda (example; path depends on timestamp and strategy):
```bash
python analyze_runs.py --run-dir runs/20251122-151458/panda/cumulative/panda3
```

---
