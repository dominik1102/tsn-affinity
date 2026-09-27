#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean
from typing import Any, Dict, Iterable, List


def _pct(x):
    if x is None:
        return "--"
    return f"{100.0 * float(x):.1f}"


def _method_name(data: Dict[str, Any], path: str) -> str:
    sig = data.get("tsn_reuse_signature") or data.get("strategy") or data.get("name") or os.path.basename(os.path.dirname(path))
    tag = data.get("tag")
    if tag:
        return str(tag)
    return str(sig)


def _iter_jsons(patterns: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        matches = glob.glob(p)
        if os.path.isdir(p):
            matches += glob.glob(os.path.join(p, "results.json"))
        out.extend(matches)
    return sorted(set(x for x in out if os.path.isfile(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="results.json files, run dirs, or glob patterns")
    ap.add_argument("--latex", action="store_true", help="print LaTeX rows")
    args = ap.parse_args()

    rows = []
    for fn in _iter_jsons(args.paths):
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
        stats = data.get("tsn_task_weight_reuse_stats") or {}
        if not stats or "_error" in stats:
            continue

        method = _method_name(data, fn)
        task_names = data.get("task_names") or []
        perf = data.get("perf_matrix") or []
        final_perf = perf[-1] if perf else []

        vals = [v for k, v in stats.items() if str(k) != "0"]
        mean_reuse = mean([float(v.get("reuse_ratio_any", 0.0)) for v in vals]) if vals else 0.0
        mean_new = mean([float(v.get("new_ratio", 0.0)) for v in vals]) if vals else 0.0
        src_cov_vals = [float(v["source_coverage"]) for v in vals if v.get("source_coverage") is not None]
        mean_src_cov = mean(src_cov_vals) if src_cov_vals else None

        rows.append({
            "method": method,
            "run_dir": os.path.dirname(fn),
            "mean_reuse": mean_reuse,
            "mean_new": mean_new,
            "mean_src_cov": mean_src_cov,
            "final_perf": final_perf,
            "task_names": task_names,
            "stats": stats,
        })

    if args.latex:
        for r in rows:
            perf_str = " / ".join(f"{float(x):.1f}" for x in r["final_perf"])
            print(
                f"{r['method']} & {perf_str} & {_pct(r['mean_reuse'])} & "
                f"{_pct(r['mean_src_cov'])} & {_pct(r['mean_new'])} \\\\" 
            )
    else:
        header = ["method", "final_perf", "mean_reuse_%", "mean_source_coverage_%", "mean_new_%", "run_dir"]
        print("\t".join(header))
        for r in rows:
            perf_str = ",".join(f"{float(x):.1f}" for x in r["final_perf"])
            print("\t".join([
                r["method"],
                perf_str,
                _pct(r["mean_reuse"]),
                _pct(r["mean_src_cov"]),
                _pct(r["mean_new"]),
                r["run_dir"],
            ]))


if __name__ == "__main__":
    main()
