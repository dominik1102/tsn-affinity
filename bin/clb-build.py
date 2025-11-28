
#!/usr/bin/env python
from __future__ import annotations
import argparse
from clbench.benchmark.builder import build_cartpole_benchmark, build_atari_benchmark
from clbench.io.serialize import save_task_specs

def main():
    p = argparse.ArgumentParser(description="Build and save CL benchmark TaskSpecs")
    p.add_argument("--benchmark", choices=["cartpole","atari"], required=True)
    p.add_argument("--kind", default=None, help="atari: atari-cl-3 or atari-pong-variants; cartpole: cartpole-cl-7")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output JSON with TaskSpecs")
    args = p.parse_args()

    specs = build_cartpole_benchmark(args.kind or "cartpole-cl-7", seed=args.seed) if args.benchmark=="cartpole"             else build_atari_benchmark(args.kind or "atari-cl-3", seed=args.seed)
    save_task_specs(args.out, specs)
    print(f"Saved {len(specs)} TaskSpecs to {args.out}")

if __name__ == "__main__":
    main()
