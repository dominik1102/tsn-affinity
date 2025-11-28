
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from .runner import BenchmarkResults

class Metric:
    def compute(self, results: BenchmarkResults) -> float: ...
    name: str = "Metric"

class ACCMetric(Metric):
    name = "ACC"
    def compute(self, results: BenchmarkResults) -> float:
        return float(np.mean(results.perf_matrix[-1]))

class BWTMetric(Metric):
    name = "BWT"
    def compute(self, results: BenchmarkResults) -> float:
        P = results.perf_matrix; n = P.shape[0]
        diffs = [P[-1, j] - P[j, j] for j in range(n-1)]
        return float(np.mean(diffs)) if diffs else 0.0

class ForgettingMetric(Metric):
    name = "Forgetting"
    def compute(self, results: BenchmarkResults) -> float:
        P = results.perf_matrix; n = P.shape[0]
        vals = []
        for j in range(n-1):
            best = np.max(P[:, j]); vals.append(best - P[-1, j])
        return float(np.mean(vals)) if vals else 0.0

class FWTMetric(Metric):
    name = "FWT"
    def compute(self, results: BenchmarkResults, zero_shot: Optional[np.ndarray]=None) -> Optional[float]:
        if zero_shot is None: return None
        P = results.perf_matrix; n = P.shape[0]
        diffs = [P[j-1, j] - zero_shot[j] for j in range(1, n)]
        return float(np.mean(diffs)) if diffs else 0.0

class StandardCLMetrics:
    @staticmethod
    def compute(results: BenchmarkResults, zero_shot: Optional[np.ndarray]=None) -> Dict[str, Any]:
        acc = ACCMetric().compute(results)
        bwt = BWTMetric().compute(results)
        fgt = ForgettingMetric().compute(results)
        fwt = FWTMetric().compute(results, zero_shot)
        out = {"ACC": acc, "BWT": bwt, "Forgetting": fgt}
        if fwt is not None: out["FWT"] = fwt
        return out
