#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_CURRENT_5 = ["Breakout", "Alien", "Atlantis", "Boxing", "Centipede"]
DEFAULT_EXTRA_2 = ["Assault", "Phoenix"]
DEFAULT_EXTRA_STRESS = ["Assault", "Phoenix", "Qbert", "Seaquest", "SpaceInvaders"]

# Prefixes are only used when no matching dataset directory exists.
# Existing dataset directory names are always preferred, because the runner usually
# loads data from <dataset_root>/<task_name>/expert_minari_dqn.npz.
DEFAULT_PREFIXES = {
    "Breakout": "D",
    "Alien": "A",
    "Atlantis": "B",
    "Boxing": "C",
    "Centipede": "E",
    "Assault": "F",
    "Phoenix": "G",
    "Qbert": "H",
    "Seaquest": "I",
    "SpaceInvaders": "J",
    "Pong": "K",
    "Asterix": "L",
    "BeamRider": "M",
}


def canonical_game(s: str) -> str:
    """Normalize common ALE game spellings to CamelCase-like names."""
    x = str(s).strip()
    aliases = {
        "space_invaders": "SpaceInvaders",
        "spaceinvaders": "SpaceInvaders",
        "space invaders": "SpaceInvaders",
        "q_bert": "Qbert",
        "qbert": "Qbert",
        "q*bert": "Qbert",
        "beam_rider": "BeamRider",
        "beamrider": "BeamRider",
        "seaquest": "Seaquest",
    }
    low = re.sub(r"[-]+", "_", x.lower())
    if low in aliases:
        return aliases[low]
    # Keep already CamelCase names. Otherwise title-case chunks.
    if any(ch.isupper() for ch in x[1:]):
        return re.sub(r"[^A-Za-z0-9]", "", x)
    return "".join(part[:1].upper() + part[1:].lower() for part in re.split(r"[^A-Za-z0-9]+", x) if part)


def env_id_for(game: str) -> str:
    return f"ALE/{canonical_game(game)}-v5"


def game_lower(game: str) -> str:
    # ALE kwargs use lowercase game name, e.g. "space_invaders" for SpaceInvaders.
    g = canonical_game(game)
    # CamelCase -> snake lower
    return re.sub(r"(?<!^)(?=[A-Z])", "_", g).lower()


def flatten_strings(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(flatten_strings(v))
    elif isinstance(obj, str):
        out.append(obj)
    return out


def spec_contains_game(spec: Dict[str, Any], game: str) -> bool:
    g = canonical_game(game)
    gl = game_lower(game)
    strings = " ".join(flatten_strings(spec)).lower()
    return g.lower() in strings or gl in strings


def extract_name(spec: Dict[str, Any]) -> str:
    for key in ("name", "task_name", "id"):
        if isinstance(spec.get(key), str):
            return spec[key]
    params = spec.get("params")
    if isinstance(params, dict):
        for key in ("name", "task_name", "id"):
            if isinstance(params.get(key), str):
                return params[key]
    return ""


def infer_game_from_name(name: str) -> Optional[str]:
    if not name:
        return None
    # D_Breakout -> Breakout
    tail = name.split("_", 1)[-1]
    tail = tail.replace("-", "_")
    if not tail:
        return None
    return canonical_game(tail)


def find_dataset_dir(dataset_root: Path, game: str) -> Optional[Path]:
    g = canonical_game(game).lower()
    candidates = []
    if not dataset_root.exists():
        return None
    for p in dataset_root.iterdir():
        if not p.is_dir():
            continue
        name_low = p.name.lower()
        tail = p.name.split("_", 1)[-1].lower()
        if name_low == g or tail == g or g in name_low:
            npz = p / "expert_minari_dqn.npz"
            if npz.exists():
                candidates.append(p)
    if candidates:
        # Prefer exact tail match over broad contains.
        candidates.sort(key=lambda p: (0 if p.name.split("_", 1)[-1].lower() == g else 1, p.name))
        return candidates[0]
    return None


def default_task_name(game: str) -> str:
    g = canonical_game(game)
    pref = DEFAULT_PREFIXES.get(g, g[0].upper())
    return f"{pref}_{g}"


def replace_strings(obj: Any, old_name: str, old_game: str, new_name: str, new_game: str, dataset_dir: Optional[Path]) -> Any:
    """Recursively update obvious game/task/env/path strings while preserving unknown schema."""
    if isinstance(obj, dict):
        new: Dict[str, Any] = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if isinstance(v, str):
                if kl in {"name", "task_name", "id"}:
                    new[k] = new_name
                    continue
                if kl in {"env_id", "env"} and ("ALE/" in v or "-v" in v):
                    new[k] = env_id_for(new_game)
                    continue
                if kl in {"game", "game_name"}:
                    # Use lowercase ALE game only when original was lower; otherwise canonical.
                    new[k] = game_lower(new_game) if v.islower() or "_" in v else canonical_game(new_game)
                    continue
                if kl in {"dataset_path", "data_path", "npz_path", "file", "path"} and dataset_dir is not None:
                    if v.endswith(".npz") or "expert_minari_dqn.npz" in v:
                        new[k] = str(dataset_dir / "expert_minari_dqn.npz")
                        continue
                new[k] = replace_string_value(v, old_name, old_game, new_name, new_game, dataset_dir)
            else:
                new[k] = replace_strings(v, old_name, old_game, new_name, new_game, dataset_dir)
        return new
    if isinstance(obj, list):
        return [replace_strings(v, old_name, old_game, new_name, new_game, dataset_dir) for v in obj]
    if isinstance(obj, str):
        return replace_string_value(obj, old_name, old_game, new_name, new_game, dataset_dir)
    return obj


def replace_string_value(s: str, old_name: str, old_game: str, new_name: str, new_game: str, dataset_dir: Optional[Path]) -> str:
    out = s
    old_game_c = canonical_game(old_game)
    new_game_c = canonical_game(new_game)

    replacements = [
        (old_name, new_name),
        (old_game_c, new_game_c),
        (old_game_c.lower(), new_game_c.lower()),
        (game_lower(old_game_c), game_lower(new_game_c)),
        (f"ALE/{old_game_c}-v5", env_id_for(new_game_c)),
    ]
    for a, b in replacements:
        out = out.replace(a, b)

    # If this is an explicit NPZ path, make sure it points to the detected folder.
    if dataset_dir is not None and ("expert_minari_dqn.npz" in out or out.endswith(".npz")):
        out = str(dataset_dir / "expert_minari_dqn.npz")
    return out


def build_specs(base_specs: List[Dict[str, Any]], games: Sequence[str], dataset_root: Path, allow_missing: bool) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    specs_by_game: Dict[str, Dict[str, Any]] = {}
    for spec in base_specs:
        name = extract_name(spec)
        inferred = infer_game_from_name(name)
        if inferred:
            specs_by_game[canonical_game(inferred)] = spec
        else:
            for g in DEFAULT_CURRENT_5 + DEFAULT_EXTRA_STRESS:
                if spec_contains_game(spec, g):
                    specs_by_game[canonical_game(g)] = spec

    # Use Breakout or first entry as template for games absent from base.
    template = specs_by_game.get("Breakout", base_specs[0])
    template_name = extract_name(template) or "D_Breakout"
    template_game = infer_game_from_name(template_name) or "Breakout"

    out_specs: List[Dict[str, Any]] = []
    for raw_game in games:
        game = canonical_game(raw_game)
        dataset_dir = find_dataset_dir(dataset_root, game)
        if dataset_dir is None:
            msg = f"[WARN] dataset not found for {game}: expected some */{game}/expert_minari_dqn.npz under {dataset_root}"
            warnings.append(msg)
            if not allow_missing:
                raise FileNotFoundError(msg + " (use --allow-missing to still write JSON)")
            new_name = default_task_name(game)
        else:
            new_name = dataset_dir.name

        if game in specs_by_game:
            # Existing game: keep base spec but adjust name to matching dataset dir if needed.
            old_spec = specs_by_game[game]
            old_name = extract_name(old_spec) or default_task_name(game)
            old_game = infer_game_from_name(old_name) or game
            new_spec = replace_strings(copy.deepcopy(old_spec), old_name, old_game, new_name, game, dataset_dir)
        else:
            new_spec = replace_strings(copy.deepcopy(template), template_name, template_game, new_name, game, dataset_dir)

        # Make extra sure common top-level fields are correct.
        if isinstance(new_spec, dict):
            if "name" in new_spec:
                new_spec["name"] = new_name
            if "env_id" in new_spec:
                new_spec["env_id"] = env_id_for(game)
            params = new_spec.get("params")
            if isinstance(params, dict):
                for key in ("env_id", "env"):
                    if key in params:
                        params[key] = env_id_for(game)
                for key in ("game", "game_name"):
                    if key in params:
                        params[key] = game_lower(game)
        out_specs.append(new_spec)

    return out_specs, warnings


def main() -> None:
    ap = argparse.ArgumentParser(description="Create extended Atari task spec JSON from an existing spec.")
    ap.add_argument("--base", default="configs/specs_atari_cl_5_minari_like_breakout_first.json")
    ap.add_argument("--dataset-root", default="/net/tscratch/people/plgdomin088/datasets/atari_expert")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", nargs="+", default=DEFAULT_CURRENT_5 + DEFAULT_EXTRA_2,
                    help="Games in desired order, e.g. Breakout Alien Atlantis Boxing Centipede Assault Phoenix")
    ap.add_argument("--allow-missing", action="store_true", help="Write JSON even if some dataset folders are missing")
    ap.add_argument("--print-schema", action="store_true", help="Print first base spec and exit")
    args = ap.parse_args()

    base_path = Path(args.base)
    dataset_root = Path(args.dataset_root)

    with base_path.open("r", encoding="utf-8") as f:
        base_specs = json.load(f)
    if not isinstance(base_specs, list) or not base_specs:
        raise ValueError(f"Expected {base_path} to contain a non-empty JSON list")

    if args.print_schema:
        print(json.dumps(base_specs[0], indent=2))
        return

    games = [canonical_game(g) for g in args.games]
    out_specs, warnings = build_specs(base_specs, games, dataset_root, bool(args.allow_missing))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_specs, f, indent=2)

    print(f"[OK] wrote {out_path}")
    print("[tasks]")
    for spec in out_specs:
        name = extract_name(spec)
        strings = " ".join(flatten_strings(spec))
        env_match = re.search(r"ALE/[A-Za-z0-9_]+-v\d+", strings)
        print(f"  - {name:24s} env={env_match.group(0) if env_match else '?'}")

    if warnings:
        print("\n".join(warnings))


if __name__ == "__main__":
    main()
