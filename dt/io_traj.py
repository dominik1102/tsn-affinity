
from __future__ import annotations
from typing import List
import pickle
from .dataset import Trajectory

def save_trajs_pickle(path: str, trajs: List[Trajectory]) -> None:
    with open(path, "wb") as f:
        pickle.dump(trajs, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_trajs_pickle(path: str) -> List[Trajectory]:
    with open(path, "rb") as f:
        return pickle.load(f)
