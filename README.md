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
python bin/clb-run-dt.py --spec specs_cp.json --strategy cumulative --steps-per-task 5000 --episodes-eval 5 --device cuda
```

Atari:
```bash
python bin/clb-run-dt.py --spec specs_atari.json --strategy cumulative --steps-per-task 20000 --episodes-eval 10 --device cuda
```

Panda (offline, PandaReach → PandaPush → PandaPickAndPlace):  
(requires datasets in `resources/datasets/*panda*_1m_expert.pkl`)
```bash
# naive
python bin/clb-run-dt-panda.py --strategy naive --steps-per-task 50000 --episodes-eval 5

# or with cumulative replay:
python bin/clb-run-dt-panda.py --strategy cumulative --steps-per-task 50000 --episodes-eval 5
```

---

### 4. Additional parameters

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
python analyze_runs.py --run-dir runs/20251122-151458/cartpole/cumulative/specs_cp
```

Panda (example; path depends on timestamp and strategy):
```bash
python analyze_runs.py --run-dir runs/20251122-151458/panda/cumulative/panda3
```

---
