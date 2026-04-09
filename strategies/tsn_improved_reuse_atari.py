from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dt.dataset import Trajectory, make_minibatches

from .tsn_common import _iter_tsn_modules
from .tsn_original_reuse_atari import TSNOriginalReuseStrategy
from .utils import _unpack_batch


class TSNImprovedReuseAtariStrategy(TSNOriginalReuseStrategy):
    """
    RL-oriented improved reuse for Atari DT.

    Compared to TSNOriginalReuseStrategy (origin reuse), this version adds:
      1) routing by ACTION COMPATIBILITY on offline expert demonstrations,
      2) optional routing by LATENT similarity (symmetric KL between diagonal
         Gaussians fitted to encoded observation latents),
      3) optional HYBRID routing = alpha * normalized(action_score)
                                  + (1-alpha) * normalized(latent_score),
      4) warm-start of mask scores from the selected source-task mask.

    Important:
      - task 0 does NOT call parent train_task/after_task;
      - task 0 is handled in the same improved pipeline,
        only with source_task=None and copy_id=0.
    """

    def __init__(
        self,
        *args,
        reuse_score_mode: str = "action",   # action | latent | hybrid
        routing_n_batches: int = 4,
        routing_batch_size: int = 64,
        action_reuse_threshold: float = 12.0,
        latent_reuse_threshold: float = 25.0,
        hybrid_reuse_threshold: float = 0.50,
        hybrid_alpha: float = 0.70,
        normalize_similarity_scores: bool = True,
        warmstart_source_scores: bool = True,
        warmstart_strength: float = 2.0,
        warmstart_noise_std: float = 0.02,
        warmstart_on_new_copy: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.reuse_score_mode = str(reuse_score_mode)
        self.routing_n_batches = int(routing_n_batches)
        self.routing_batch_size = int(routing_batch_size)

        # lower is better
        self.action_reuse_threshold = float(action_reuse_threshold)
        self.latent_reuse_threshold = float(latent_reuse_threshold)
        self.hybrid_reuse_threshold = float(hybrid_reuse_threshold)
        self.hybrid_alpha = float(hybrid_alpha)
        self.normalize_similarity_scores = bool(normalize_similarity_scores)

        self.warmstart_source_scores = bool(warmstart_source_scores)
        self.warmstart_strength = float(warmstart_strength)
        self.warmstart_noise_std = float(warmstart_noise_std)
        self.warmstart_on_new_copy = bool(warmstart_on_new_copy)

        # per-task latent stats under task's own copy/mask
        self.task_latent_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Latent helpers
    # ------------------------------------------------------------------
    def _extract_obs_latents(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract observation latents from DecisionTransformer.obs_enc.

        obs: [B,L,C,H,W] or [B,L,D]
        returns: [B,L,d_model]
        """
        x = obs.to(self.device)
        orig_dtype = x.dtype
        x = x.to(dtype=torch.float32)

        if orig_dtype == torch.uint8:
            x = x / 255.0
        else:
            if x.numel() > 0 and float(x.max().item()) > 1.5:
                x = x / 255.0

        if x.dim() == 5:
            B, L, C, H, W = x.shape
            z = self.model.obs_enc(x.reshape(B * L, C, H, W)).reshape(B, L, -1)
        elif x.dim() == 3:
            B, L, D = x.shape
            z = self.model.obs_enc(x.reshape(B * L, D)).reshape(B, L, -1)
        else:
            raise ValueError(f"Unexpected obs shape for latent extraction: {tuple(x.shape)}")

        return z

    @staticmethod
    def _diag_stats(z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit diagonal Gaussian to latent vectors.
        """
        if z.dim() == 3:
            if mask is None:
                x = z.reshape(-1, z.shape[-1])
            else:
                keep = mask.reshape(-1).bool()
                x = z.reshape(-1, z.shape[-1])[keep]
        else:
            x = z

        if x.numel() == 0:
            H = z.shape[-1]
            return torch.zeros(H, device=z.device), torch.ones(H, device=z.device)

        mu = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False).clamp(min=1e-6)
        return mu, var

    @staticmethod
    def _sym_kl_diag_gaussians(
        mu_a: torch.Tensor,
        var_a: torch.Tensor,
        mu_b: torch.Tensor,
        var_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Symmetric KL between diagonal Gaussians.
        """
        kl_ab = 0.5 * torch.sum(torch.log(var_b / var_a) + (var_a + (mu_a - mu_b).pow(2)) / var_b - 1.0)
        kl_ba = 0.5 * torch.sum(torch.log(var_a / var_b) + (var_b + (mu_b - mu_a).pow(2)) / var_a - 1.0)
        return 0.5 * (kl_ab + kl_ba)

    def _compute_current_memory_latent_stats(self, task_memory_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs = task_memory_obs.to(self.device, dtype=torch.float32).unsqueeze(0)  # [1,N,...]
            z = self._extract_obs_latents(obs)  # [1,N,H]
            mu, var = self._diag_stats(z)
            return mu.detach().cpu(), var.detach().cpu()

    def _store_task_latent_stats(self, task_id: int) -> None:
        """
        Store latent stats of one finished task under its own copy/mask.
        """
        if int(task_id) not in self.task_memories:
            return

        mem = self.task_memories[int(task_id)]
        self.set_eval_task(int(task_id))
        mu, var = self._compute_current_memory_latent_stats(mem)
        self.task_latent_stats[int(task_id)] = (mu, var)
        self.clear_eval_task()

    # ------------------------------------------------------------------
    # Improved routing scores
    # ------------------------------------------------------------------
    def _make_routing_loader(self, task_trajs: List[Trajectory]):
        return make_minibatches(task_trajs, self.seq_len, self.routing_batch_size, self.device)

    def _estimate_action_compatibility(self, source_task_id: int, task_trajs: List[Trajectory]) -> float:
        """
        How well does source-task policy explain expert actions of the NEW task?
        Lower is better.
        """
        self.set_eval_task(int(source_task_id))
        self.model.eval()

        loader = self._make_routing_loader(task_trajs)
        vals: List[float] = []

        with torch.no_grad():
            for _ in range(max(1, self.routing_n_batches)):
                obs, actions, rtg, ts, mask = _unpack_batch(next(loader))
                logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    actions.reshape(-1),
                    ignore_index=-1,
                )
                vals.append(float(ce.detach().cpu().item()))

        self.clear_eval_task()
        return float(np.mean(vals)) if vals else float("inf")

    def _estimate_latent_similarity(self, source_task_id: int, task_memory_obs: torch.Tensor) -> float:
        """
        Compare NEW task latents (under source copy/mask) with stored source-task
        latent distribution. Lower is better.
        """
        if int(source_task_id) not in self.task_latent_stats:
            return float("inf")

        self.set_eval_task(int(source_task_id))
        mu_ref, var_ref = self.task_latent_stats[int(source_task_id)]

        with torch.no_grad():
            mu_new, var_new = self._compute_current_memory_latent_stats(task_memory_obs)
            skl = self._sym_kl_diag_gaussians(
                mu_new.to(self.device),
                var_new.to(self.device),
                mu_ref.to(self.device),
                var_ref.to(self.device),
            )

        self.clear_eval_task()
        return float(skl.detach().cpu().item())

    @staticmethod
    def _normalize_score_dict(values: Dict[int, float]) -> Dict[int, float]:
        if not values:
            return {}
        arr = np.array(list(values.values()), dtype=np.float32)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if abs(vmax - vmin) < 1e-12:
            return {k: 0.0 for k in values}
        return {k: float((v - vmin) / (vmax - vmin)) for k, v in values.items()}

    def _select_copy_for_new_task_improved(
        self,
        task_memory_obs: torch.Tensor,
        task_trajs: List[Trajectory],
    ) -> Tuple[int, Optional[int], Dict[str, Optional[float]], bool]:
        """
        RL-aware routing.

        Modes:
          - action: choose task with minimum action CE on new-task expert data
          - latent: choose task with minimum latent symmetric-KL
          - hybrid: alpha * normalized(action) + (1-alpha) * normalized(latent)

        Returns:
            (copy_id, source_task_id, score_details, created_new_copy)
        """
        if self.current_task_id == 0 or not self.task_to_copy:
            return 0, None, {"best_action": None, "best_latent": None, "best_score": None, "best_kl": None}, False

        action_scores: Dict[int, float] = {}
        latent_scores: Dict[int, float] = {}
        prev_tasks = sorted(self.task_to_copy.keys())

        for t in prev_tasks:
            action_scores[int(t)] = self._estimate_action_compatibility(int(t), task_trajs)
            if self.reuse_score_mode in ("latent", "hybrid"):
                latent_scores[int(t)] = self._estimate_latent_similarity(int(t), task_memory_obs)

        if self.reuse_score_mode == "action":
            final_scores = dict(action_scores)
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            create_new = best_score > self.action_reuse_threshold
            details = {
                "best_action": float(action_scores[best_task]),
                "best_latent": None,
                "best_score": float(best_score),
                "best_kl": None,
            }

        elif self.reuse_score_mode == "latent":
            final_scores = dict(latent_scores)
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            create_new = best_score > self.latent_reuse_threshold
            details = {
                "best_action": float(action_scores.get(best_task, float("nan"))),
                "best_latent": float(latent_scores[best_task]),
                "best_score": float(best_score),
                "best_kl": None,
            }

        elif self.reuse_score_mode == "hybrid":
            act_n = self._normalize_score_dict(action_scores) if self.normalize_similarity_scores else action_scores
            lat_n = self._normalize_score_dict(latent_scores) if self.normalize_similarity_scores else latent_scores
            final_scores = {
                t: float(self.hybrid_alpha * act_n[t] + (1.0 - self.hybrid_alpha) * lat_n[t])
                for t in prev_tasks
            }
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            create_new = best_score > self.hybrid_reuse_threshold
            details = {
                "best_action": float(action_scores[best_task]),
                "best_latent": float(latent_scores[best_task]),
                "best_score": float(best_score),
                "best_kl": None,
            }

        else:
            raise ValueError(f"Unsupported reuse_score_mode: {self.reuse_score_mode}")

        if create_new:
            if self.max_model_copies is not None and len(self.copy_states) >= self.max_model_copies:
                copy_id = self.task_to_copy[int(best_task)]
                return int(copy_id), int(best_task), details, False

            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            return len(self.copy_states) - 1, int(best_task), details, True

        return int(self.task_to_copy[int(best_task)]), int(best_task), details, False

    # ------------------------------------------------------------------
    # Warm-start from selected source mask
    # ------------------------------------------------------------------
    def _warmstart_scores_from_source_mask(self, source_task: Optional[int]) -> None:
        if (not self.warmstart_source_scores) or (source_task is None):
            return

        st = self._active_state()
        if int(source_task) not in st.per_task_masks:
            return

        src_masks = st.per_task_masks[int(source_task)]

        with torch.no_grad():
            for name, mod in _iter_tsn_modules(self.model):
                w_key = f"{name}.weight"
                src_w = src_masks.get(w_key, None)
                if src_w is not None:
                    src_w = src_w.to(device=mod.score.device, dtype=mod.score.dtype)
                    mod.score.normal_(mean=0.0, std=self.warmstart_noise_std)
                    mod.score.add_(self.warmstart_strength * src_w)

                if getattr(mod, "bias_score", None) is not None:
                    b_key = f"{name}.bias"
                    src_b = src_masks.get(b_key, None)
                    if src_b is not None:
                        src_b = src_b.to(device=mod.bias_score.device, dtype=mod.bias_score.dtype)
                        mod.bias_score.normal_(mean=0.0, std=self.warmstart_noise_std)
                        mod.bias_score.add_(self.warmstart_strength * src_b)

    # ------------------------------------------------------------------
    # training API override
    # ------------------------------------------------------------------
    def train_task(self, task_trajs: List[Trajectory], steps: int = 2000, batch_size: int = 64):
        task_memory = self._build_task_memory(task_trajs)

        # task 0 also goes through THIS class, not parent train_task
        if self.current_task_id == 0:
            copy_id = 0
            src_task = None
            score_details = {
                "best_action": None,
                "best_latent": None,
                "best_score": None,
                "best_kl": None,
            }
            created_new = False
        else:
            copy_id, src_task, score_details, created_new = self._select_copy_for_new_task_improved(
                task_memory,
                task_trajs,
            )

        self._activate_copy(copy_id)

        self.task_similarity[self.current_task_id] = {
            "source_task": None if src_task is None else int(src_task),
            "copy_id": int(copy_id),
            "best_action": score_details.get("best_action", None),
            "best_latent": score_details.get("best_latent", None),
            "best_score": score_details.get("best_score", None),
            "best_kl": score_details.get("best_kl", None),
            "score_mode": self.reuse_score_mode,
            "created_new_copy": bool(created_new),
        }

        self._prepare_current_task()

        if src_task is not None:
            if (not created_new) or self.warmstart_on_new_copy:
                self._warmstart_scores_from_source_mask(src_task)

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
                    f"[tsn-improved-reuse-atari] task={self.current_task_id} copy={meta['copy_id']} "
                    f"src={meta['source_task']} score_mode={meta['score_mode']} "
                    f"best_action={meta['best_action']} best_latent={meta['best_latent']} "
                    f"best_score={meta['best_score']} new_copy={int(meta['created_new_copy'])} "
                    f"it={it} ce={last_loss:.6e} keep_ratio={self.current_keep_ratio:.4f}"
                )

        self.task_memories[self.current_task_id] = task_memory.detach().cpu()
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

        if self.reuse_score_mode in ("latent", "hybrid"):
            self._store_task_latent_stats(task_id)

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
            f"[tsn-improved-reuse-atari] after task {task_id}: copy={self.current_copy_id} "
            f"occupied_ratio={ratio:.4f} source_task={meta.get('source_task', None)} "
            f"best_action={meta.get('best_action', None)} "
            f"best_latent={meta.get('best_latent', None)} "
            f"best_score={meta.get('best_score', None)} "
            f"created_new_copy={meta.get('created_new_copy', None)}"
        )

        self.set_eval_task(task_id)
        self.current_task_id += 1