from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dt.dataset import Trajectory, make_minibatches
from dt.model import DecisionTransformer
from .base import BaseStrategy
from .tsn_common import (
    TSNConversionConfig,
    _convert_module_inplace,
    _iter_tsn_modules,
    _kmeans_quantize_selected_,
    _rebuild_optimizer_like,
)
from .utils import _unpack_batch


@dataclass
class _ReuseCopyState:
    """Persistent state of one model copy in old-style TSN reuse."""

    model: torch.nn.Module
    opt: torch.optim.Optimizer
    per_task_masks: Dict[int, Dict[str, Optional[torch.Tensor]]]
    consolidated_masks: Dict[str, Optional[torch.Tensor]]
    task_codebooks: Dict[int, Dict[str, np.ndarray]]
    task_keep_ratios: Dict[int, float]


class TSNOriginalReuseStrategy(BaseStrategy):
    """
    Old-style TSN reuse adapted to Atari DT.

    What this implements from the original repo:
      1) keep a replay memory per task,
      2) before training a new task, compare its replay memory to previous tasks
         using a KL-style similarity score,
      3) if the best previous task is close enough -> route the task to that
         task's model copy,
      4) otherwise create a fresh model copy,
      5) inside a selected copy, allow mask overlap with already occupied weights,
         i.e. old weights can be reused but stay frozen.

    Important notes:
      - This is a clean RL adaptation, not a byte-for-byte port of the repo's
        script logic.
      - Similarity is computed on stored raw observation memories using KL on
        flattened observation tensors. This is the closest analogue of the old
        repo's replay-memory KL routing without introducing a new policy-aware
        reuse mechanism yet.
      - Reuse is ALWAYS enabled inside an existing copy (old TSN behavior).
      - Quantization remains optional and can be switched off as in your runs.
    """

    def __init__(
        self,
        *args,
        keep_ratio: float = 0.5,
        include_embeddings: bool = True,
        quantize_after_task: bool = True,
        quant_clusters: int = 16,
        freeze_non_mask_params_after_first: bool = True,
        skip_module_names: Tuple[str, ...] = ("dt.te",),
        expected_num_tasks: Optional[int] = None,
        keep_ratio_schedule: str = "constant",
        min_keep_ratio: float = 1e-3,
        grad_clip: float = 1.0,
        reuse_memory_size: int = 256,
        reuse_kl_threshold: float = 0.25,
        max_model_copies: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.base_keep_ratio = float(keep_ratio)
        self.current_keep_ratio = float(keep_ratio)
        self.include_embeddings = bool(include_embeddings)
        self.quantize_after_task = bool(quantize_after_task)
        self.quant_clusters = int(quant_clusters)
        # old reuse should allow overlap inside a selected copy
        self.allow_weight_reuse = True
        self.freeze_non_mask_params_after_first = bool(freeze_non_mask_params_after_first)
        self.skip_module_names = tuple(skip_module_names)
        self.expected_num_tasks = None if expected_num_tasks is None else int(expected_num_tasks)
        self.keep_ratio_schedule = str(keep_ratio_schedule)
        self.min_keep_ratio = float(min_keep_ratio)
        self.grad_clip = float(grad_clip)

        self.reuse_memory_size = int(reuse_memory_size)
        self.reuse_kl_threshold = float(reuse_kl_threshold)
        self.max_model_copies = None if max_model_copies is None else int(max_model_copies)

        self.current_task_id: int = 0
        self.current_copy_id: int = 0
        self.active_eval_task: Optional[int] = None

        self.task_to_copy: Dict[int, int] = {}
        self.task_memories: Dict[int, torch.Tensor] = {}
        self.task_keep_ratios: Dict[int, float] = {}
        self.task_similarity: Dict[int, Dict[str, Optional[float]]] = {}

        # create first copy from the BaseStrategy model
        first_copy = self._convert_existing_model_to_copy(self.model, self.opt)
        self.copy_states: List[_ReuseCopyState] = [first_copy]
        self._activate_copy(0)

    def _sync_public_state_to_active_copy(self) -> None:
        """
        Synchronize public strategy refs (self.model / self.opt) back into the
        currently active copy state.

        This is crucial because the runner may patch the public model/optimizer
        after strategy construction (e.g. expand dt.te and rebuild optimizer).
        Without syncing, _activate_copy(0) could restore a stale optimizer from
        copy_states[0].
        """
        if not hasattr(self, "copy_states"):
            return
        if not self.copy_states:
            return
        if self.current_copy_id < 0 or self.current_copy_id >= len(self.copy_states):
            return

        st = self.copy_states[self.current_copy_id]
        st.model = self.model
        st.opt = self.opt

    # ------------------------------------------------------------------
    # copy / model management
    # ------------------------------------------------------------------
    def _effective_model_kwargs(self) -> Dict[str, object]:
        """
        Build kwargs for creating a fresh model copy.

        We use the current effective max_ep_len / rtg_scale so copies created
        after the runner patches remain compatible with evaluation.
        """
        hp = dict(self.model_hparams)
        hp["max_ep_len"] = int(getattr(self.model, "max_ep_len", hp.get("max_ep_len", 10000)))
        hp["rtg_scale"] = float(getattr(self.model, "rtg_scale", hp.get("rtg_scale", 1000.0)))
        return hp

    def _convert_existing_model_to_copy(
        self,
        model: torch.nn.Module,
        opt: torch.optim.Optimizer,
    ) -> _ReuseCopyState:
        cfg = TSNConversionConfig(
            keep_ratio=float(self.base_keep_ratio),
            include_embeddings=bool(self.include_embeddings),
            allow_weight_reuse=True,
            skip_module_names=tuple(self.skip_module_names),
        )
        _convert_module_inplace(model, cfg)
        model.to(self.device)
        opt = _rebuild_optimizer_like(opt, model.parameters())
        return _ReuseCopyState(
            model=model,
            opt=opt,
            per_task_masks={},
            consolidated_masks={},
            task_codebooks={},
            task_keep_ratios={},
        )

    def _make_fresh_copy(self) -> _ReuseCopyState:
        hp = self._effective_model_kwargs()
        model = DecisionTransformer(
            obs_shape=hp["obs_shape"],
            n_actions=int(hp["n_actions"]),
            d_model=int(hp["d_model"]),
            n_layers=int(hp["n_layers"]),
            n_heads=int(hp["n_heads"]),
            seq_len=int(hp["seq_len"]),
            p_drop=float(hp["p_drop"]),
            max_ep_len=int(hp["max_ep_len"]),
            rtg_scale=float(hp["rtg_scale"]),
        ).to(self.device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(hp["lr"]),
            weight_decay=float(hp["weight_decay"]),
        )
        return self._convert_existing_model_to_copy(model, opt)

    def _activate_copy(self, copy_id: int) -> None:
        self._sync_public_state_to_active_copy()

        self.current_copy_id = int(copy_id)
        st = self.copy_states[self.current_copy_id]
        self.model = st.model
        self.opt = st.opt
        self._refresh_name_sets()

    # ------------------------------------------------------------------
    # old-reuse task memory and KL routing
    # ------------------------------------------------------------------
    def _build_task_memory(self, task_trajs: List[Trajectory]) -> torch.Tensor:
        """
        Build per-task replay memory used by old-style routing.

        Important fix:
        use a LOCAL RNG so we do not perturb the global numpy RNG that is also
        used later by make_minibatches(...). This keeps task-0 behavior much
        closer to plain TSN.
        """
        obs_chunks: List[np.ndarray] = []
        total = 0
        for tr in task_trajs:
            obs = np.asarray(tr.obs, dtype=np.float32)
            if obs.ndim < 2:
                continue
            obs_chunks.append(obs)
            total += int(obs.shape[0])
            if total >= self.reuse_memory_size * 4:
                break

        if not obs_chunks:
            raise ValueError("TSNOriginalReuseStrategy: cannot build task memory from empty observations")

        obs_all = np.concatenate(obs_chunks, axis=0)
        n = min(int(obs_all.shape[0]), int(self.reuse_memory_size))

        # local RNG -> does not consume global np.random state used by training samplers
        rng = np.random.default_rng(10_000 + int(self.current_task_id))
        idx = rng.choice(obs_all.shape[0], size=n, replace=False)

        mem = torch.as_tensor(obs_all[idx], dtype=torch.float32)
        return mem

    def _memory_kl(self, mem_a: torch.Tensor, mem_b: torch.Tensor) -> float:
        """
        KL between two replay memories in the spirit of the old repo.

        The original repo applies KLDivLoss to replay-memory arrays directly.
        Here we make it numerically valid by flattening each observation sample
        and turning it into a probability distribution with softmax.
        """
        a = mem_a.reshape(mem_a.shape[0], -1)
        b = mem_b.reshape(mem_b.shape[0], -1)
        m = min(int(a.shape[0]), int(b.shape[0]))
        a = a[:m]
        b = b[:m]
        a_logp = F.log_softmax(a, dim=-1)
        b_p = F.softmax(b, dim=-1)
        kl = F.kl_div(a_logp, b_p, reduction="batchmean")
        return float(kl.detach().cpu().item())

    def _select_copy_for_new_task(self, task_memory: torch.Tensor) -> Tuple[int, Optional[int], Optional[float], bool]:
        """
        Old-style reuse decision:
          - compare new task memory with all previous task memories,
          - find the previous task with minimum KL,
          - if min KL <= threshold -> reuse that task's model copy,
          - else create a new model copy.

        Returns:
            (copy_id, source_task_id, best_kl, created_new_copy)
        """
        if self.current_task_id == 0 or not self.task_memories:
            return 0, None, None, False

        best_task: Optional[int] = None
        best_kl: Optional[float] = None
        for t, mem in self.task_memories.items():
            kl = self._memory_kl(task_memory, mem)
            if (best_kl is None) or (kl < best_kl):
                best_kl = kl
                best_task = int(t)

        create_new = False
        if best_kl is None:
            create_new = True
        elif best_kl > self.reuse_kl_threshold:
            create_new = True

        if create_new:
            if self.max_model_copies is not None and len(self.copy_states) >= self.max_model_copies:
                # fall back to best existing copy if max number of copies is reached
                copy_id = self.task_to_copy[best_task] if best_task is not None else 0
                return int(copy_id), best_task, best_kl, False
            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            return len(self.copy_states) - 1, best_task, best_kl, True

        assert best_task is not None
        return int(self.task_to_copy[best_task]), best_task, best_kl, False

    # ------------------------------------------------------------------
    # TSN bookkeeping inside the active copy
    # ------------------------------------------------------------------
    def _refresh_name_sets(self) -> None:
        self._maskable_param_names = set()
        self._score_param_names = set()
        for mod_name, mod in _iter_tsn_modules(self.model):
            self._maskable_param_names.add(f"{mod_name}.weight")
            self._score_param_names.add(f"{mod_name}.score")
            if getattr(mod, "bias", None) is not None:
                self._maskable_param_names.add(f"{mod_name}.bias")
            if getattr(mod, "bias_score", None) is not None:
                self._score_param_names.add(f"{mod_name}.bias_score")

    def _active_state(self) -> _ReuseCopyState:
        return self.copy_states[self.current_copy_id]

    def _reset_all_scores(self) -> None:
        for _, mod in _iter_tsn_modules(self.model):
            mod.reset_scores()
            mod.clear_active_masks()

    def _sync_occupied_masks_into_modules(self) -> None:
        st = self._active_state()
        for name, mod in _iter_tsn_modules(self.model):
            mod.clear_occupied_masks()
            w_key = f"{name}.weight"
            b_key = f"{name}.bias"
            if w_key in st.consolidated_masks and st.consolidated_masks[w_key] is not None:
                mod.occupied_weight_mask = st.consolidated_masks[w_key].detach().clone()
            if b_key in st.consolidated_masks and st.consolidated_masks[b_key] is not None:
                mod.occupied_bias_mask = st.consolidated_masks[b_key].detach().clone()

    def _collect_current_task_masks(self) -> Dict[str, Optional[torch.Tensor]]:
        out: Dict[str, Optional[torch.Tensor]] = {}
        for name, mod in _iter_tsn_modules(self.model):
            if mod.weight_mask is None:
                with torch.no_grad():
                    _ = mod._current_weight_mask()
                    if getattr(mod, "bias", None) is not None:
                        _ = mod._current_bias_mask()
            out[f"{name}.weight"] = None if mod.weight_mask is None else mod.weight_mask.detach().cpu().clone()
            out[f"{name}.bias"] = None if mod.bias_mask is None else mod.bias_mask.detach().cpu().clone()
        return out

    def _update_consolidated_masks(self, task_masks: Dict[str, Optional[torch.Tensor]]) -> None:
        st = self._active_state()
        if not st.consolidated_masks:
            st.consolidated_masks = {k: (None if v is None else v.clone()) for k, v in task_masks.items()}
            return
        for key, mask in task_masks.items():
            if mask is None:
                st.consolidated_masks.setdefault(key, None)
                continue
            if st.consolidated_masks.get(key) is None:
                st.consolidated_masks[key] = mask.clone()
            else:
                st.consolidated_masks[key] = torch.logical_or(
                    st.consolidated_masks[key].bool(),
                    mask.bool(),
                ).to(torch.uint8)

    def _apply_eval_masks(self, task_id: Optional[int]) -> None:
        if task_id is None:
            for _, mod in _iter_tsn_modules(self.model):
                mod.clear_active_masks()
            return

        copy_id = self.task_to_copy.get(int(task_id), None)
        if copy_id is None:
            for _, mod in _iter_tsn_modules(self.model):
                mod.clear_active_masks()
            return

        self._activate_copy(copy_id)
        st = self._active_state()
        task_masks = st.per_task_masks.get(int(task_id), None)
        for name, mod in _iter_tsn_modules(self.model):
            if task_masks is None:
                mod.clear_active_masks()
                continue
            mod.active_weight_mask = task_masks.get(f"{name}.weight", None)
            mod.active_bias_mask = task_masks.get(f"{name}.bias", None)

    def has_task_mask(self, task_id: int) -> bool:
        copy_id = self.task_to_copy.get(int(task_id), None)
        if copy_id is None:
            return False
        return int(task_id) in self.copy_states[copy_id].per_task_masks

    def set_eval_task(self, task_id: int) -> None:
        task_id = int(task_id)
        self.active_eval_task = task_id if task_id in self.task_to_copy else None
        self._apply_eval_masks(self.active_eval_task)

    def clear_eval_task(self) -> None:
        self.active_eval_task = None
        self._apply_eval_masks(None)

    def _compute_task_keep_ratio(self) -> float:
        if self.keep_ratio_schedule == "constant" or self.expected_num_tasks is None:
            return float(max(self.min_keep_ratio, min(1.0, self.base_keep_ratio)))

        if self.keep_ratio_schedule == "equal_remaining":
            remaining = max(1, int(self.expected_num_tasks) - int(self.current_task_id))
            value = 1.0 / float(remaining)
            return float(max(self.min_keep_ratio, min(1.0, value)))

        raise ValueError(f"Unsupported keep_ratio_schedule: {self.keep_ratio_schedule}")

    def _apply_keep_ratio_to_modules(self, keep_ratio: float) -> None:
        self.current_keep_ratio = float(keep_ratio)
        for _, mod in _iter_tsn_modules(self.model):
            mod.keep_ratio = float(keep_ratio)

    def _zero_prev_task_param_grads(self) -> None:
        st = self._active_state()
        if not st.consolidated_masks:
            return
        name_to_module = dict(self.model.named_modules())
        for key, old_mask in st.consolidated_masks.items():
            if old_mask is None:
                continue
            module_name, attr = key.rsplit(".", 1)
            module = name_to_module.get(module_name, None)
            if module is None:
                continue
            param = getattr(module, attr, None)
            if param is None or param.grad is None:
                continue
            mask = old_mask.to(device=param.grad.device, dtype=torch.bool)
            param.grad.masked_fill_(mask, 0.0)

    def _zero_non_maskable_grads(self) -> None:
        if not self.freeze_non_mask_params_after_first:
            return
        st = self._active_state()
        if not st.consolidated_masks:
            return
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name in self._maskable_param_names or name in self._score_param_names:
                continue
            param.grad.zero_()

    def _snapshot_frozen_params(self) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        snap: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        st = self._active_state()
        if not st.consolidated_masks:
            return snap

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            old_mask = st.consolidated_masks.get(name, None)
            if old_mask is not None:
                snap[name] = (
                    param.detach().clone(),
                    old_mask.to(device=param.device, dtype=torch.bool),
                )
                continue

            if self.freeze_non_mask_params_after_first:
                if name not in self._maskable_param_names and name not in self._score_param_names:
                    snap[name] = (param.detach().clone(), None)

        return snap

    def _restore_frozen_params_(self, snapshot: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> None:
        if not snapshot:
            return
        name_to_param = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, (old_value, restore_mask) in snapshot.items():
                param = name_to_param.get(name, None)
                if param is None:
                    continue
                if restore_mask is None:
                    param.copy_(old_value)
                else:
                    param.copy_(torch.where(restore_mask, old_value, param))

    def _prepare_current_task(self) -> None:
        self.clear_eval_task()
        self._sync_occupied_masks_into_modules()
        keep_ratio = self._compute_task_keep_ratio()
        self._apply_keep_ratio_to_modules(keep_ratio)
        self.task_keep_ratios[self.current_task_id] = float(keep_ratio)
        st = self._active_state()
        st.task_keep_ratios[self.current_task_id] = float(keep_ratio)

        # fresh score search for every task inside the selected copy
        # reset only if this copy already trained at least one task
        if len(st.per_task_masks) > 0:
            self._reset_all_scores()
            st.opt = _rebuild_optimizer_like(st.opt, self.model.parameters())
            self.opt = st.opt
        else:
            self.opt = st.opt

    def _quantize_new_weights_for_current_task(self, task_masks: Dict[str, Optional[torch.Tensor]]) -> Dict[str, np.ndarray]:
        codebooks: Dict[str, np.ndarray] = {}
        if not self.quantize_after_task:
            return codebooks

        st = self._active_state()
        name_to_module = dict(self.model.named_modules())
        for key, mask in task_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            module_name, _ = key.rsplit(".", 1)
            module = name_to_module.get(module_name, None)
            if module is None:
                continue
            prev_mask = st.consolidated_masks.get(key, None)
            if prev_mask is None:
                new_mask = mask.bool()
            else:
                new_mask = torch.logical_and(mask.bool(), ~prev_mask.bool())
            if int(new_mask.sum().item()) == 0:
                continue
            centers = _kmeans_quantize_selected_(module.weight, new_mask, self.quant_clusters)
            if centers is not None:
                codebooks[key] = centers
        return codebooks

    # ------------------------------------------------------------------
    # BaseStrategy-like API
    # ------------------------------------------------------------------
    def train_task(self, task_trajs: List[Trajectory], steps: int = 2000, batch_size: int = 64):
        task_memory = self._build_task_memory(task_trajs)
        copy_id, src_task, best_kl, created_new = self._select_copy_for_new_task(task_memory)
        self._activate_copy(copy_id)
        self.task_similarity[self.current_task_id] = {
            "source_task": None if src_task is None else int(src_task),
            "copy_id": int(copy_id),
            "best_kl": None if best_kl is None else float(best_kl),
            "created_new_copy": bool(created_new),
        }

        self._prepare_current_task()
        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)

        self.model.train()
        last_loss = None
        for it in range(int(steps)):
            obs, actions, rtg, ts, mask = _unpack_batch(next(loader))
            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                actions.reshape(-1),
                ignore_index=-1,
            )

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self._zero_prev_task_param_grads()
            self._zero_non_maskable_grads()
            frozen_snapshot = self._snapshot_frozen_params()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self._restore_frozen_params_(frozen_snapshot)

            last_loss = float(loss.detach().item())
            if it % 20000 == 0 or it == int(steps) - 1:
                meta = self.task_similarity[self.current_task_id]
                print(
                    f"[tsn-old-reuse] task={self.current_task_id} copy={meta['copy_id']} "
                    f"src={meta['source_task']} kl={meta['best_kl']} new_copy={int(meta['created_new_copy'])} "
                    f"it={it} ce={last_loss:.6e} keep_ratio={self.current_keep_ratio:.4f}"
                )

        # store task memory after successful training selection
        self.task_memories[self.current_task_id] = task_memory.detach().cpu()
        # make sure the active copy stores the final up-to-date model/optimizer
        self._sync_public_state_to_active_copy()
        return {"loss": last_loss, "keep_ratio": float(self.current_keep_ratio)}

    def after_task(self, task_trajs: List[Trajectory]):
        self._sync_public_state_to_active_copy()
        task_id = int(self.current_task_id)
        st = self._active_state()

        task_masks = self._collect_current_task_masks()
        st.per_task_masks[task_id] = task_masks
        st.task_codebooks[task_id] = self._quantize_new_weights_for_current_task(task_masks)
        self._update_consolidated_masks(task_masks)

        self.task_to_copy[task_id] = int(self.current_copy_id)

        used = 0
        total = 0
        for key, mask in st.consolidated_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            used += int(mask.sum().item())
            total += int(mask.numel())
        ratio = float(used / max(1, total))
        meta = self.task_similarity.get(task_id, {})
        print(
            f"[tsn-old-reuse] after task {task_id}: copy={self.current_copy_id} occupied_ratio={ratio:.4f} "
            f"source_task={meta.get('source_task', None)} best_kl={meta.get('best_kl', None)}"
        )

        self.set_eval_task(task_id)
        self.current_task_id += 1
