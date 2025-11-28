
#!/usr/bin/env python3
"""
analyze_runs.py — viewer for CLBench-DT per-step metrics + heatmaps (with % annotations)
Outputs:
  - forgetting_matrix.csv, delta_vs_diag_matrix.csv (absolute values)
  - forgetting_pct_matrix.csv, delta_vs_diag_pct_matrix.csv (percentages)
  - forgetting_heatmap.png (abs), forgetting_heatmap_pct.png (% of best_so_far)
  - delta_vs_diag_heatmap.png (abs), delta_vs_diag_heatmap_pct.png (% of diag_baseline)
"""
import argparse, json, os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def load_results(results_path: str):
    with open(results_path, "r", encoding="utf-8") as f:
        R = json.load(f)
    P = np.array(R["perf_matrix"], dtype=np.float32)
    names = R["task_names"]
    return names, P, R

def compute_matrices(names: List[str], P: np.ndarray):
    n = P.shape[0]
    F = np.full((n, n), np.nan, dtype=np.float32)
    D = np.full((n, n), np.nan, dtype=np.float32)
    Best = np.full((n, n), np.nan, dtype=np.float32)
    Diag = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        for j in range(i+1):
            current = P[i, j]
            diag_base = P[j, j]
            best_so_far = np.max(P[:i+1, j])
            D[i, j] = current - diag_base
            F[i, j] = best_so_far - current
            Best[i, j] = best_so_far
            Diag[i, j] = diag_base
    eps = 1e-8
    F_pct = np.full_like(F, np.nan)
    D_pct = np.full_like(D, np.nan)
    maskF = ~np.isnan(F) & (Best > eps)
    maskD = ~np.isnan(D) & (Diag > eps)
    F_pct[maskF] = 100.0 * (F[maskF] / Best[maskF])
    D_pct[maskD] = 100.0 * (D[maskD] / Diag[maskD])
    return F, D, F_pct, D_pct

def plot_heatmap(mat: np.ndarray, title: str, xticks: List[str], out_path: str, vmin=None, vmax=None, annotate=False, fmt="abs"):
    M = np.ma.masked_invalid(mat)
    plt.figure()
    im = plt.imshow(M, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Task j (source)")
    plt.ylabel("Step i (after training on task i)")
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=45, ha="right")
    plt.yticks(ticks=range(len(xticks)), labels=[str(i+1) for i in range(len(xticks))])
    plt.colorbar(im)
    if annotate:
        nrows, ncols = mat.shape
        for i in range(nrows):
            for j in range(ncols):
                val = mat[i, j]
                if np.isfinite(val):
                    s = f"{val:.0f}%" if fmt == "pct" else f"{val:.0f}"
                    plt.text(j, i, s, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default="", help="Folder with results.json")
    ap.add_argument("--results", type=str, default="", help="Path to results.json directly")
    ap.add_argument("--out-dir", type=str, default="", help="Where to save charts (defaults to RUN_DIR)")
    args = ap.parse_args()

    if args.results:
        results_path = args.results
        run_dir = args.run_dir or os.path.dirname(results_path)
    elif args.run_dir:
        results_path = os.path.join(args.run_dir, "results.json")
        run_dir = args.run_dir
    else:
        raise SystemExit("Provide --run-dir or --results")

    if not os.path.exists(results_path):
        raise SystemExit(f"results.json not found: {results_path}")

    out_dir = args.out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)

    names, P, _ = load_results(results_path)
    F, D, F_pct, D_pct = compute_matrices(names, P)

    np.savetxt(os.path.join(out_dir, "forgetting_matrix.csv"), F, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(out_dir, "delta_vs_diag_matrix.csv"), D, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(out_dir, "forgetting_pct_matrix.csv"), F_pct, delimiter=",", fmt="%.2f")
    np.savetxt(os.path.join(out_dir, "delta_vs_diag_pct_matrix.csv"), D_pct, delimiter=",", fmt="%.2f")

    fmax = np.nanmax(F) if np.isfinite(F).any() else None
    plot_heatmap(F, "Forgetting (abs): best_so_far - current", names,
                 os.path.join(out_dir, "forgetting_heatmap.png"),
                 vmin=0.0 if fmax is not None else None, vmax=fmax, annotate=True, fmt="abs")
    plot_heatmap(F_pct, "Forgetting (% of best_so_far)", names,
                 os.path.join(out_dir, "forgetting_heatmap_pct.png"),
                 vmin=0.0, vmax=np.nanmax(F_pct) if np.isfinite(F_pct).any() else None, annotate=True, fmt="pct")

    plot_heatmap(D, "Delta vs diag (abs): P[i,j] - P[j,j]", names,
                 os.path.join(out_dir, "delta_vs_diag_heatmap.png"),
                 annotate=True, fmt="abs")
    plot_heatmap(D_pct, "Delta vs diag (% of diag_baseline)", names,
                 os.path.join(out_dir, "delta_vs_diag_heatmap_pct.png"),
                 annotate=True, fmt="pct")

    print(f"Saved heatmaps (with % labels) and CSVs to: {out_dir}")

if __name__ == "__main__":
    main()
