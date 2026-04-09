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
python bin/clb-run-dt.py   --spec configs/specs_atari_cl_5_minari_like.json --dataset-root resources/atari_expert --steps 20000 --seq-len 20  --episodes-eval 30 --device cuda   --strategy cumulative 
```

## cyfronet
Atari:
```bash
python bin/clb-run-dt.py --spec configs/specs_atari_cl_5_minari_like.json --dataset-root /net/tscratch/people/plgdomin088/datasets/atari_expert --strategy cumulative --steps-per-task 20000   --seq-len 20 --episodes-eval 30 --max-steps 27000 --atari-env minari_like  --replay-check --device cuda
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
python bin/clb-run-dt-cartpole-single.py --spec configs/specs_cp.json --dataset-root data/cartpole_expert --seq-len 20 --steps-per-task 50000 --episodes-eval 5 --device cuda
```

For Atari just change the spec:

```bash
python bin/clb-run-dt-atari-single.py --spec configs/specs_atari_cl_5_minari_like.json --dataset-root resources/atari_expert --steps 20000 --seq-len 20 --batch-size 64 --episodes-eval 30 --max-ep-len 27000 --device cuda --debug-replay
```

for atari on cyfronet
```bash
python bin/clb-run-dt-atari-single.py --spec configs/specs_atari_cl_5_minari_like.json --dataset-root /net/tscratch/people/plgdomin088/datasets/atari_expert --steps 20000 --seq-len 20 --batch-size 64 --episodes-eval 30 --max-ep-len 27000 --device cuda --debug-replay --rtg-scale 0
 ```

debug dataset and env
```bash
python bin/clb-run-dt-atari-single.py  --spec configs/specs_atari.json --dataset-root resources/atari_expert --steps 1 --debug-replay
 ```
#### Panda (continuous, offline datasets)

```bash
python bin/clb-run-dt-panda-single.py   --datasets-root resources/datasets   --seq-len 20   --steps-per-task 50000   --batch-size 64   --episodes-eval 5   --device cuda
```
### helios
```bash
python bin/clb-run-dt-panda-single.py   --datasets-root /net/scratch/hscra/plgrid/plgdomin088/datasets/panda_expert   --seq-len 20   --steps-per-task 50000   --batch-size 64   --episodes-eval 5   --device cuda
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
python bin/train_atari_expert.py   --spec specs_atari.json   --episodes-per-task 200   --max-len 20000   --total-steps-expert 5000000  --out-dir data/atari_expert   --device cuda   --expert-action-prob 1
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
python analyze_runs.py --run-dir runs/20260406-211003/atari/tsn_improved_reuse/specs_atari_cl_5_minari_like_breakout_first__dm128_L3_H4_K20_drop0.10__rsm-hybrid_hthr0.5_ha0.70
```

---
Inspect atari expert trajectories (visualize frames + stats):
```bash
python tools/inspect_atari_trajs.py --root resources/atari_expert --tasks A_Pong B_Breakout C_Seaquest
```

---
Generate Atari expert dataset in DT `.npz` format from Minari:
```bash
python bin/generate_atari_traj_from_mintari.py --config configs/specs_atari_cl_5_minari_like.json --out-root /net/tscratch/people/plgdomin088/datasets/atari_expert --max-len 50000 --obs-dtype float32
```
---



# TSN -> Atari DT integration notes

## What this version does

This is a pragmatic `TSNStrategy` for your offline Atari Decision Transformer flow.
It implements:

- trainable score masks on `nn.Conv2d`, `nn.Linear`, and optionally `nn.Embedding`
- one binary mask per task
- freezing of already-occupied parameters from previous tasks
- optional reuse of occupied weights (`allow_weight_reuse`)
- post-task KMeans quantization of newly claimed weights only

## What is intentionally missing for now

- KL-based task similarity check / model duplication
- replay-memory based sharing decisions
- greedy post-training pruning search
- paper-style exact capacity accounting

## Important defaults

Recommended starting defaults for Atari:

- `keep_ratio=0.5`
- `allow_weight_reuse=False`
- `include_embeddings=True`
- `skip_module_names=("dt.te",)`
- `freeze_non_mask_params_after_first=True`

Why skip `dt.te` first?
Because the time embedding is large and shared across tasks; leaving it dense/frozen makes the first Atari port more stable.
The action embedding `dt.ae` is still converted to TSN when embeddings are enabled.

## Very important detail

This port reinitializes score tensors before each new task.
That is necessary because task-specific masks are stored explicitly after each task, so score tensors must be free to learn a fresh subnetwork for the next task.

## Evaluation caveat

For TSN-like methods, `task_id` matters.
So in your CL matrix the lower triangle is the meaningful one.
Evaluating unseen future tasks before their mask exists is not very informative.

```bash
python bin/clb-run-dt.py  --strategy tsn --spec configs/specs_atari_cl_5_minari_like.json --dataset-root /net/tscratch/people/plgdomin088/datasets/atari_expert --atari-env minari_like  --seq-len 20 --steps-per-task 2000 --batch-size 64 --target-mode max  --tsn-keep-ratio 0.5 --tsn-quant-clusters 16 --tsn-skip-module dt.te --tsn-keep-schedule equal_remaining --tsn-min-keep-ratio 1e-3 --tsn-grad-clip 1.0
```


```bash
python run_panda_cl.py --strategy tsn --tag panda3_tsn --seq-len 20 --steps-per-task 1000000 --episodes-eval 20 --max-steps 50 --device cuda -batch-size 128 -d-model 128 -n-layers 3 -n-heads 1 --p-drop 0.1 --lr 1e-4 --weight-decay 1e-4 --grad-clip 0.25 -max-ep-len 50 --rtg-scale 1000.0 \-tsn-keep-ratio 0.5 --tsn-quant-clusters 16
```
