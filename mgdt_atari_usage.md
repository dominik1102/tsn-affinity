# MGDT Atari joint training (5-game setup)

Files:
- `mgdt_pytorch.py` -> put into your project (e.g. `dt/mgdt_pytorch.py` or similar)
- `clb_run_mgdt_atari.py` -> runner script

Suggested placement in your repo:
- `dt/mgdt_pytorch.py`
- `bin/clb-run-mgdt-atari.py`

If you move `mgdt_pytorch.py` under `dt/`, change the import in the runner from:

```python
from mgdt_pytorch import ...
```

to:

```python
from dt.mgdt_pytorch import ...
```

Example command for your five-game Atari spec:

```bash
python bin/clb-run-mgdt-atari.py --spec configs/specs_atari_cl_5_minari_like_breakout_first.json --dataset-root /path/to/atari_expert --dataset-file expert_minari_dqn.npz --seq-len 20  --steps 20000 --batch-size 64 --device cuda --d-model 512 --n-layers 8 --n-heads 8 --p-drop 0.1 --patch-size 14 --lr 3e-4 --weight-decay 1e-4 --eval-every 2000 --offline-eval-batches 20 --env-eval-every 10000 episodes-eval 10 max-ep-len 27000 greedy-actions tag mgdt5_joint
```

What the script gives you:
- joint multi-game MGDT training on the same 5 Atari games as in your paper,
- the same `--spec` / `--dataset-root` interface style as your current Atari scripts,
- periodic offline per-game teacher-forced losses,
- optional environment evaluation with MGDT-style return-guided action inference.

Important note:
The environment evaluation here is a practical PyTorch autoregressive implementation for sanity-checking learning.
It should be treated as an MGDT-style evaluation loop, not yet as a bit-exact reproduction of the original JAX/Colab code.
