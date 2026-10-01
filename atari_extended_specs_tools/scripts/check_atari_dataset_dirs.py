#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser(description="Check Atari expert NPZ folders for candidate games.")
    ap.add_argument("--dataset-root", default="/net/tscratch/people/plgdomin088/datasets/atari_expert")
    ap.add_argument("games", nargs="*", default=["Assault", "Phoenix", "Qbert", "Seaquest", "SpaceInvaders", "Pong", "Asterix", "BeamRider"])
    args = ap.parse_args()
    root = Path(args.dataset_root)
    print(f"dataset_root={root}")
    if not root.exists():
        print("[ERROR] dataset root does not exist")
        return
    for g in args.games:
        hits = []
        gl = g.lower().replace("_", "")
        for p in root.iterdir():
            if not p.is_dir():
                continue
            tail = p.name.split("_", 1)[-1].lower().replace("_", "")
            name = p.name.lower().replace("_", "")
            if tail == gl or gl in name:
                npz = p / "expert_minari_dqn.npz"
                hits.append((p.name, npz.exists()))
        if hits:
            for name, ok in hits:
                print(f"{g:16s} -> {name:24s} npz={'OK' if ok else 'MISSING'}")
        else:
            print(f"{g:16s} -> NOT FOUND")
if __name__ == "__main__":
    main()
