#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tensorflow.compat.v2 as tf

def main():




    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="gs://rl-infra-public/multi_game_dt/checkpoint_38274228.pkl")
    ap.add_argument("--dst", default="checkpoint_38274228.pkl")
    args = ap.parse_args()

    tf.enable_v2_behavior()
    print(f"Copying {args.src} -> {args.dst}")
    with tf.io.gfile.GFile(args.src, "rb") as fin, tf.io.gfile.GFile(args.dst, "wb") as fout:
        fout.write(fin.read())
    print("Done.")

if __name__ == "__main__":
    main()
