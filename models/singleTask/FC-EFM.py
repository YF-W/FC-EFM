# -*- coding: utf-8 -*-
"""
EmotionFlow_FF
FF is used as a full tri-modal representation encoder.
EmotionFlow keeps the downstream innovation chain unchanged.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..subNets import BertTextEncoder
except ImportError:  # pragma: no cover
    from models.subNets import BertTextEncoder

from .EmotionFlow import EmotionFlow, _get

__all__ = ["EmotionFlowFF"]


class FFSequenceAligner(nn.Module):
    def __init__(self, target_len: int = 0):
        super().__init__()
        self.target_len = int(target_len)

    def forward(self, x: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"FFSequenceAligner expects [B,T,D], got {tuple(x.shape)}")
        tgt = int(target_len) if target_len is not None else self.target_len
        if tgt <= 0 or x.size(1) == tgt:
            return x
        x_t = x.transpose(1, 2)
        x_t = F.interpolate(x_t, size=tgt, mode="linear", align_corners=False)
        return x_t.transpose(1, 2)


class FFCrossAttentionAlign(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        num_heads = int(max(1, num_heads))
        if dim % num_heads != 0:
            num_heads = 1
        self.t2a = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.a2t = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.t2v = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.v2t = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.a2v = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.v2a = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_t = nn.LayerNorm(dim)
        self.ln_a = nn.LayerNorm(dim)
        self.ln_v = nn.LayerNorm(dim)

    def forward(self, t_proj: torch.Tensor, a_proj: torch.Tensor, v_proj: torch.Tensor):
        t2a_out, _ = self.t2a(t_proj, a_proj, a_proj)
        a2t_out, _ = self.a2t(a_proj, t_proj, t_proj)
        t2v_out, _ = self.t2v(t_proj, v_proj, v_proj)
        v2t_out, _ = self.v2t(v_proj, t_proj, t_proj)
        a2v_out, _ = self.a2v(a_proj, v_proj, v_proj)
        v2a_out, _ = self.v2a(v_proj, a_proj, a_proj)
        return {
            "T2A": self.ln_t(t_proj + t2a_out),
            "A2T": self.ln_a(a_proj + a2t_out),
            "T2V": self.ln_t(t_proj + t2v_out),
            "V2T": self.ln_v(v_proj + v2t_out),
            "A2V": self.ln_a(a_proj + a2v_out),
            "V2A": self.ln_v(v_proj + v2a_out),
        }


class FFConflictExtractor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_t = nn.Linear(dim, dim)
        self.proj_a = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def forward(
        self,
        t_proj: torch.Tensor,
        a_proj: torch.Tensor,
        v_proj: torch.Tensor,
        aligned: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conf_t = (aligned["T2A"] - t_proj) + (aligned["T2V"] - t_proj)
        conf_a = (aligned["A2T"] - a_proj) + (aligned["A2V"] - a_proj)
        conf_v = (aligned["V2T"] - v_proj) + (aligned["V2A"] - v_proj)
        return self.proj_t(conf_t), self.proj_a(conf_a), self.proj_v(conf_v)


class FFConflictTripletExtractor(nn.Module):
    def __init__(self, window_size: int = 2, pool: str = "mean"):
        super().__init__()
        self.window_size = int(max(1, window_size))
        self.pool = str(pool).lower()

    def _temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return torch.zeros(x.size(1), device=x.device, dtype=x.dtype)
        if self.pool == "max":
            return x.max(dim=0).values
        return x.mean(dim=0)

    def forward(self, conflict_feat: torch.Tensor, conflict_mask: torch.Tensor):
        if conflict_feat.dim() != 3:
            raise ValueError(f"FFConflictTripletExtractor expects [B,T,D], got {tuple(conflict_feat.shape)}")
        if conflict_mask.dim() != 2:
            raise ValueError(f"FFConflictTripletExtractor expects [B,T] mask, got {tuple(conflict_mask.shape)}")
        if conflict_mask.size(1) != conflict_feat.size(1):
            raise ValueError(
                f"FFConflictTripletExtractor mask length mismatch: feat={conflict_feat.size(1)} mask={conflict_mask.size(1)}"
            )

        B, L, _ = conflict_feat.shape
        pre_list, con_list, post_list = [], [], []
        for b in range(B):
            idx = torch.nonzero(conflict_mask[b] > 0, as_tuple=False).flatten()
            c = idx[0].item() if idx.numel() > 0 else min(L - 1, L // 2)
            pre_start = max(0, c - self.window_size)
            post_end = min(L, c + 1 + self.window_size)
            pre = self._temporal_pool(conflict_feat[b, pre_start:c, :])
            con = conflict_feat[b, c, :]
            post = self._temporal_pool(conflict_feat[b, c + 1:post_end, :])
            pre_list.append(pre)
            con_list.append(con)
            post_list.append(post)
        return torch.stack(pre_list, dim=0), torch.stack(con_list, dim=0), torch.stack(post_list, dim=0)


class FFStructuralOperators(nn.Module):
    def __init__(self, dim: int, alpha: float = 1.0, beta: float = 1.0, use_norm: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.use_norm = bool(use_norm)
        self.pre_proj = nn.Linear(dim, dim)
        self.con_proj = nn.Linear(dim, dim)
        self.post_proj = nn.Linear(dim, dim)
        if self.use_norm:
            self.ln_pre = nn.LayerNorm(dim)
            self.ln_con = nn.LayerNorm(dim)
            self.ln_post = nn.LayerNorm(dim)

    def _norm(self, x: torch.Tensor, layer: Optional[nn.LayerNorm]) -> torch.Tensor:
        return layer(x) if layer is not None else x

    def forward(self, pre: torch.Tensor, con: torch.Tensor, post: torch.Tensor):
        pre_ref = self.pre_proj(pre)
        pre_ref = self._norm(pre_ref, self.ln_pre if self.use_norm else None)
        rel = self.alpha * self.con_proj(con - pre_ref)
        rel = self._norm(rel, self.ln_con if self.use_norm else None)
        resp = self.post_proj(post - self.beta * rel)
        resp = self._norm(resp, self.ln_post if self.use_norm else None)
        return pre_ref, rel, resp


def _non_traditional_similarity(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return float(alpha) * (x_norm * y_norm).sum(dim=-1)


class FFConflictLocator(nn.Module):
    def __init__(self, spectral_metric: str = "l2", top_k: int = 1):
        super().__init__()
        self.spectral_metric = str(spectral_metric).lower()
        self.top_k = int(max(1, top_k))

    def spectral_shift(self, matrix: torch.Tensor, stable_matrix: torch.Tensor) -> torch.Tensor:
        ref = stable_matrix.repeat(1, 3, 1)
        if self.spectral_metric == "cosine":
            x_norm = F.normalize(matrix, dim=-1)
            y_norm = F.normalize(ref, dim=-1)
            return 1.0 - (x_norm * y_norm).sum(dim=-1)
        return torch.norm(matrix - ref, dim=-1)

    def forward(self, structure_matrix: torch.Tensor, stable_matrix: torch.Tensor):
        if structure_matrix.dim() != 3 or structure_matrix.size(1) != 9:
            raise ValueError(f"FFConflictLocator expects [B,9,D], got {tuple(structure_matrix.shape)}")
        scores = self.spectral_shift(structure_matrix, stable_matrix)
        threshold = scores.mean(dim=1, keepdim=True)
        conflict_time_mask = scores > threshold
        max_time_step = structure_matrix.size(1) // 3

        out_t, out_a, out_v = [], [], []
        for b in range(structure_matrix.size(0)):
            idx = torch.nonzero(conflict_time_mask[b], as_tuple=False).flatten().tolist()
            idx = [i for i in idx if i < max_time_step]
            if len(idx) == 0:
                idx = [max_time_step // 2]
            if len(idx) >= self.top_k:
                idx = idx[: self.top_k]
            else:
                idx = idx + [idx[-1]] * (self.top_k - len(idx))
            t_rows = [i * 3 + 0 for i in idx]
            a_rows = [i * 3 + 1 for i in idx]
            v_rows = [i * 3 + 2 for i in idx]
            out_t.append(structure_matrix[b, t_rows, :])
            out_a.append(structure_matrix[b, a_rows, :])
            out_v.append(structure_matrix[b, v_rows, :])
        return torch.stack(out_t, dim=0), torch.stack(out_a, dim=0), torch.stack(out_v, dim=0)


class FFLocalFeatureExtractor(nn.Module):
    def __init__(self, window_size: int = 3, pool: str = "mean"):
        super().__init__()
        self.window_size = int(max(1, window_size))
        self.pool = str(pool).lower()

    def _temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return torch.zeros(x.size(1), device=x.device, dtype=x.dtype)
        if self.pool == "max":
            return x.max(dim=0).values
        return x.mean(dim=0)

    def forward(self, conflict_feat: torch.Tensor) -> torch.Tensor:
        if conflict_feat.dim() != 3:
            raise ValueError(f"FFLocalFeatureExtractor expects [B,T,D], got {tuple(conflict_feat.shape)}")
        B, L, _ = conflict_feat.shape
        local_feats = []
        half = self.window_size // 2
        for b in range(B):
            feat_windows = []
            for t in range(L):
                start = max(0, t - half)
                end = min(L, t + half + 1)
                feat_windows.append(self._temporal_pool(conflict_feat[b, start:end, :]))
            local_feats.append(torch.stack(feat_windows, dim=0))
        return torch.stack(local_feats, dim=0)


class FFLocalFeature3DSpaceBuilder(nn.Module):
    def __init__(self, eps: float = 1e-8, normalize: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.normalize = bool(normalize)

    def forward(self, local_feat: torch.Tensor):
        if local_feat.dim() != 3:
            raise ValueError(f"FFLocalFeature3DSpaceBuilder expects [B,T,D], got {tuple(local_feat.shape)}")
        B, L, _ = local_feat.shape
        device = local_feat.device
        t = torch.linspace(0.0, 1.0, L, device=device).unsqueeze(0).expand(B, -1)
        e = torch.norm(local_feat, dim=-1)
        delta = torch.zeros_like(e)
        delta[:, 1:] = torch.norm(local_feat[:, 1:] - local_feat[:, :-1], dim=-1)
        if self.normalize:
            e = (e - e.mean(dim=1, keepdim=True)) / e.std(dim=1, keepdim=True).clamp_min(self.eps)
            delta = (delta - delta.mean(dim=1, keepdim=True)) / delta.std(dim=1, keepdim=True).clamp_min(self.eps)
        P = torch.stack([t, e, delta], dim=-1)
        return P, {"t": t, "e": e, "delta": delta}


class FFTriModalConflictSpace(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, conflict_t: torch.Tensor, conflict_a: torch.Tensor, conflict_v: torch.Tensor):
        c_ta = F.cosine_similarity(conflict_t, conflict_a, dim=-1)
        c_tv = F.cosine_similarity(conflict_t, conflict_v, dim=-1)
        c_av = F.cosine_similarity(conflict_a, conflict_v, dim=-1)
        c = (c_ta + c_tv + c_av) / 3.0

        k_ta = torch.norm(conflict_t - conflict_a, dim=-1)
        k_tv = torch.norm(conflict_t - conflict_v, dim=-1)
        k_av = torch.norm(conflict_a - conflict_v, dim=-1)
        k = (k_ta + k_tv + k_av) / 3.0

        n_t = torch.norm(conflict_t, dim=-1)
        n_a = torch.norm(conflict_a, dim=-1)
        n_v = torch.norm(conflict_v, dim=-1)
        denom = (n_t + n_a + n_v).clamp_min(self.eps)
        w_t = n_t / denom
        w_a = n_a / denom
        w_v = n_v / denom

        q = torch.stack([c, k, w_t, w_a, w_v], dim=-1)
        return q, {"c": c, "k": k, "wT": w_t, "wA": w_a, "wV": w_v}


class FFRepresentationEncoder(nn.Module):
    """
    Full FF front-end without FF's final FusionModule/FusionPredictor.
    Returns corrected tri-modal sequences after FF's conflict reasoning and feedback.
    """

    def __init__(self, args, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.use_bert = bool(_get(args, "use_bert", False))
        self.use_finetune = bool(_get(args, "use_finetune", False))
        self.transformers = _get(args, "transformers", "bert")
        self.pretrained = _get(args, "pretrained", "bert-base-uncased")

        feat_dims = _get(args, "feature_dims", [768, 5, 20])
        text_in = int(_get(args, "post_text_inputs", feat_dims[0]))
        audio_in = int(_get(args, "audio_inputs", feat_dims[1]))
        video_in = int(_get(args, "video_inputs", feat_dims[2]))
        self.audio_in = int(audio_in)
        self.video_in = int(video_in)
        self.align_dim = int(_get(args, "align_dim", self.out_dim))

        audio_hidden = int(_get(args, "audio_hidden", max(32, self.out_dim // 2)))
        video_hidden = int(_get(args, "video_hidden", max(32, self.out_dim // 2)))
        lstm_layers = int(_get(args, "lstm_num_layers", 1))
        lstm_dropout = float(_get(args, "dropouts", 0.0))
        lstm_dropout = lstm_dropout if lstm_layers > 1 else 0.0
        attn_heads = int(_get(args, "ff_align_heads", 4))
        target_len = int(_get(args, "text_len", 0))

        if self.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=self.use_finetune,
                transformers=self.transformers,
                pretrained=self.pretrained,
            )

        self.audio_extractor = nn.LSTM(
            input_size=audio_in,
            hidden_size=audio_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.video_extractor = nn.LSTM(
            input_size=video_in,
            hidden_size=video_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.map_text = nn.Identity() if text_in == self.align_dim else nn.Linear(text_in, self.align_dim)
        self.map_audio = nn.Linear(2 * audio_hidden, self.align_dim)
        self.map_video = nn.Linear(2 * video_hidden, self.align_dim)

        self.cross_align = FFCrossAttentionAlign(self.align_dim, attn_heads)
        self.conflict_extractor = FFConflictExtractor(self.align_dim)
        self.conflict_triplet = FFConflictTripletExtractor(
            window_size=int(_get(args, "conflict_window", 2)),
            pool=str(_get(args, "conflict_pool", "mean")),
        )
        self.structural_ops = FFStructuralOperators(
            dim=self.align_dim,
            alpha=float(_get(args, "alpha_con", 1.0)),
            beta=float(_get(args, "beta_post", 1.0)),
            use_norm=bool(_get(args, "use_struct_norm", True)),
        )
        self.conflict_locator = FFConflictLocator(
            spectral_metric=str(_get(args, "spectral_metric", "l2")),
            top_k=int(_get(args, "ff_conflict_top_k", 5)),
        )
        local_pool = str(_get(args, "ff_local_pool", _get(args, "conflict_pool", "mean")))
        self.local_extractor = FFLocalFeatureExtractor(
            window_size=int(_get(args, "ff_local_window", 3)),
            pool=local_pool,
        )
        self.local_space = FFLocalFeature3DSpaceBuilder(normalize=bool(_get(args, "ff_local_space_norm", True)))
        self.conflict_space = FFTriModalConflictSpace()

        self.seq_align_text = FFSequenceAligner(target_len=target_len)
        self.seq_align_audio = FFSequenceAligner(target_len=target_len)
        self.seq_align_video = FFSequenceAligner(target_len=target_len)

        self.out_proj_t = nn.Identity() if self.align_dim == self.out_dim else nn.Linear(self.align_dim, self.out_dim)
        self.out_proj_a = nn.Identity() if self.align_dim == self.out_dim else nn.Linear(self.align_dim, self.out_dim)
        self.out_proj_v = nn.Identity() if self.align_dim == self.out_dim else nn.Linear(self.align_dim, self.out_dim)

        self.matrix_alpha = float(_get(args, "matrix_alpha", 1.0))
        self.matrix_noise = float(_get(args, "matrix_noise", 0.1))
        self.feedback_translation = float(_get(args, "alpha_G_translation", 0.1))
        self.feedback_gamma = float(_get(args, "gamma_feedback", 0.1))
        self._dim_warned = {"audio": False, "video": False}

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x

    @staticmethod
    def _align_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(1) == target_len:
            return x
        x_t = x.transpose(1, 2)
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_t.transpose(1, 2)

    def _match_last_dim(self, x: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
        cur_dim = int(x.size(-1))
        if cur_dim == int(expected_dim):
            return x
        if not self._dim_warned.get(name, False):
            print(
                f"[FFDimAlign] {name} feature dim mismatch: expected={expected_dim}, got={cur_dim}; "
                "apply minimal pad/truncate for compatibility."
            )
            self._dim_warned[name] = True
        if cur_dim > int(expected_dim):
            return x[..., :expected_dim]
        pad = x.new_zeros(*x.shape[:-1], int(expected_dim) - cur_dim)
        return torch.cat([x, pad], dim=-1)

    def _build_fake_stable_matrix(self, pre_states):
        rows = []
        for mod_feat in pre_states:
            cand_1 = mod_feat + self.matrix_noise * torch.randn_like(mod_feat)
            cand_2 = mod_feat + 2.0 * self.matrix_noise * torch.randn_like(mod_feat)
            sims = torch.stack(
                [
                    _non_traditional_similarity(mod_feat, cand_1, self.matrix_alpha),
                    _non_traditional_similarity(mod_feat, cand_2, self.matrix_alpha),
                ],
                dim=1,
            )
            weights = torch.softmax(sims, dim=1).unsqueeze(-1)
            candidates = torch.stack([cand_1, cand_2], dim=1)
            rows.append((weights * candidates).sum(dim=1))
        return torch.stack(rows, dim=1)

    def _full_ff_features(self, t_proj: torch.Tensor, a_proj: torch.Tensor, v_proj: torch.Tensor, return_aux: bool = False):
        aligned = self.cross_align(t_proj, a_proj, v_proj)
        conflict_t_raw, conflict_a_raw, conflict_v_raw = self.conflict_extractor(t_proj, a_proj, v_proj, aligned)

        base_len = self.seq_align_text.target_len if self.seq_align_text.target_len > 0 else max(
            conflict_t_raw.size(1), conflict_a_raw.size(1), conflict_v_raw.size(1)
        )
        conflict_t = self._align_seq(conflict_t_raw, base_len)
        conflict_a = self._align_seq(conflict_a_raw, base_len)
        conflict_v = self._align_seq(conflict_v_raw, base_len)

        conflict_score = torch.norm(conflict_t - conflict_a, dim=-1)
        threshold = conflict_score.mean(dim=1, keepdim=True)
        conflict_mask = (conflict_score > threshold).float()

        pre_t, con_t, post_t = self.conflict_triplet(conflict_t, conflict_mask)
        pre_a, con_a, post_a = self.conflict_triplet(conflict_a, conflict_mask)
        pre_v, con_v, post_v = self.conflict_triplet(conflict_v, conflict_mask)

        spre_t, scon_t, spost_t = self.structural_ops(pre_t, con_t, post_t)
        spre_a, scon_a, spost_a = self.structural_ops(pre_a, con_a, post_a)
        spre_v, scon_v, spost_v = self.structural_ops(pre_v, con_v, post_v)

        fake_stable = self._build_fake_stable_matrix([spre_t, spre_a, spre_v])
        raw_structure = torch.stack(
            [pre_t, pre_a, pre_v, con_t, con_a, con_v, post_t, post_a, post_v],
            dim=1,
        )
        true_conflict_t, true_conflict_a, true_conflict_v = self.conflict_locator(raw_structure, fake_stable)

        spre_t, scon_t, spost_t = self.structural_ops(pre_t, true_conflict_t.mean(dim=1), post_t)
        spre_a, scon_a, spost_a = self.structural_ops(pre_a, true_conflict_a.mean(dim=1), post_a)
        spre_v, scon_v, spost_v = self.structural_ops(pre_v, true_conflict_v.mean(dim=1), post_v)
        conflict_modality = (scon_t + scon_a + scon_v) / 3.0

        local_conflict_t = self.local_extractor(conflict_t)
        local_conflict_a = self.local_extractor(conflict_a)
        local_conflict_v = self.local_extractor(conflict_v)

        P_t, info_t = self.local_space(local_conflict_t)
        P_a, info_a = self.local_space(local_conflict_a)
        P_v, info_v = self.local_space(local_conflict_v)
        q_struct, _ = self.conflict_space(conflict_t, conflict_a, conflict_v)

        P_micro = torch.cat([P_t, P_a, P_v], dim=1)
        G_used = torch.cat([q_struct, q_struct, q_struct], dim=1)
        P_micro_adjusted = P_micro.clone()
        P_micro_adjusted[:, :, 0] = P_micro_adjusted[:, :, 0] + self.feedback_translation * G_used[:, :, 0]
        P_micro_adjusted[:, :, 1] = P_micro_adjusted[:, :, 1] + self.feedback_translation * G_used[:, :, 1]
        P_micro_adjusted[:, :, 2] = P_micro_adjusted[:, :, 2] + self.feedback_translation * G_used[:, :, 2]

        feedback_delta = P_micro_adjusted[:, :, 2]
        delta_t, delta_a, delta_v = feedback_delta.split(base_len, dim=1)
        conflict_t = conflict_t + self.feedback_gamma * delta_t.unsqueeze(-1)
        conflict_a = conflict_a + self.feedback_gamma * delta_a.unsqueeze(-1)
        conflict_v = conflict_v + self.feedback_gamma * delta_v.unsqueeze(-1)

        if return_aux:
            q_struct_refined, struct_info_refined = self.conflict_space(conflict_t, conflict_a, conflict_v)
        else:
            q_struct_refined = None
            struct_info_refined = None
        local_conflict = torch.cat([local_conflict_t, local_conflict_a, local_conflict_v], dim=1)
        scale = math.sqrt(float(local_conflict.size(-1)))
        query = conflict_modality.unsqueeze(1)
        attn_scores = torch.bmm(query, local_conflict.transpose(1, 2)) / scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        amplified_conflict_seq = local_conflict * attn_weights.transpose(1, 2)
        amplified_conflict_global = torch.bmm(attn_weights, local_conflict).squeeze(1) if return_aux else None

        conflict_t_fb, conflict_a_fb, conflict_v_fb = amplified_conflict_seq.split(base_len, dim=1)
        conflict_t_fb = self._align_seq(conflict_t_fb, t_proj.size(1))
        conflict_a_fb = self._align_seq(conflict_a_fb, a_proj.size(1))
        conflict_v_fb = self._align_seq(conflict_v_fb, v_proj.size(1))

        t_proj = t_proj - self.feedback_gamma * conflict_t_fb
        a_proj = a_proj - self.feedback_gamma * conflict_a_fb
        v_proj = v_proj - self.feedback_gamma * conflict_v_fb

        if not return_aux:
            return t_proj, a_proj, v_proj, None

        aux = {
            "ff_conflict_mask": conflict_mask,
            "ff_conflict_score": conflict_score,
            "ff_true_conflict_T": true_conflict_t,
            "ff_true_conflict_A": true_conflict_a,
            "ff_true_conflict_V": true_conflict_v,
            "ff_fake_stable": fake_stable,
            "ff_q_struct": q_struct_refined,
            "ff_struct_info_wT": struct_info_refined["wT"],
            "ff_struct_info_wA": struct_info_refined["wA"],
            "ff_struct_info_wV": struct_info_refined["wV"],
            "ff_attn_weights": attn_weights,
            "ff_global_conflict": amplified_conflict_global,
            "ff_local_conflict": local_conflict,
            "ff_p_micro": P_micro,
            "ff_p_micro_adjusted": P_micro_adjusted,
            "ff_major_conflict_local": P_micro_adjusted[
                torch.arange(P_micro_adjusted.size(0), device=P_micro_adjusted.device),
                P_micro_adjusted[:, :, 2].argmax(dim=1),
                :,
            ],
            "ff_local_space_t": info_t,
            "ff_local_space_a": info_a,
            "ff_local_space_v": info_v,
            "ff_spre_t": spre_t,
            "ff_spre_a": spre_a,
            "ff_spre_v": spre_v,
            "ff_scon_t": scon_t - self.feedback_gamma * amplified_conflict_global,
            "ff_scon_a": scon_a - self.feedback_gamma * amplified_conflict_global,
            "ff_scon_v": scon_v - self.feedback_gamma * amplified_conflict_global,
            "ff_spost_t": spost_t,
            "ff_spost_a": spost_a,
            "ff_spost_v": spost_v,
        }
        return t_proj, a_proj, v_proj, aux

    def forward(self, text_x, audio_x, video_x, return_aux: bool = False):
        text = text_x[0] if isinstance(text_x, (list, tuple)) else text_x
        audio = audio_x[0] if isinstance(audio_x, (list, tuple)) else audio_x
        video = video_x[0] if isinstance(video_x, (list, tuple)) else video_x

        if self.use_bert and text.dim() == 3 and text.size(1) >= 3:
            text_feat = self.text_model(text)
        else:
            text_feat = text

        text_feat = self._ensure_3d(text_feat)
        audio = self._ensure_3d(audio)
        video = self._ensure_3d(video)
        audio = self._match_last_dim(audio, self.audio_in, "audio")
        video = self._match_last_dim(video, self.video_in, "video")

        audio_out, _ = self.audio_extractor(audio)
        video_out, _ = self.video_extractor(video)

        t_proj = self.map_text(text_feat)
        a_proj = self.map_audio(audio_out)
        v_proj = self.map_video(video_out)

        z_t, z_a, z_v, aux = self._full_ff_features(t_proj, a_proj, v_proj, return_aux=return_aux)
        z_t = self.seq_align_text(self.out_proj_t(z_t))
        z_a = self.seq_align_audio(self.out_proj_a(z_a))
        z_v = self.seq_align_video(self.out_proj_v(z_v))

        if return_aux:
            aux = dict(aux)
            aux["ff_t_proj_out"] = z_t
            aux["ff_a_proj_out"] = z_a
            aux["ff_v_proj_out"] = z_v
            return z_t, z_a, z_v, aux
        return z_t, z_a, z_v


class EmotionFlowFF(EmotionFlow):
    """
    Full-FF -> EmotionFlow fusion.
    FF provides feedback-corrected tri-modal features; EmotionFlow keeps downstream flow intact.
    """

    def __init__(self, args):
        super().__init__(args)

        # Replace EmotionFlow's native modality encoding with FF front-end features.
        self.use_bert = False
        self.proj_t = nn.Identity()
        self.proj_a = nn.Identity()
        self.proj_v = nn.Identity()
        self.ff_encoder = FFRepresentationEncoder(args, out_dim=self.hidden_dim)

    @staticmethod
    def _extract_text_mask(text_x, masks) -> Optional[torch.Tensor]:
        text_mask = None
        if isinstance(masks, dict):
            text_mask = masks.get("text", masks.get("text_mask", None))
        if text_mask is None and torch.is_tensor(text_x) and text_x.dim() == 3 and text_x.size(1) == 3:
            text_mask = text_x[:, 1, :]
        if text_mask is None:
            return None
        if text_mask.dim() == 3:
            text_mask = text_mask.squeeze(-1)
        return text_mask.float()

    @staticmethod
    def _align_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        if mask.size(1) == target_len:
            return mask
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=target_len, mode="nearest")
        return mask.squeeze(1)

    @staticmethod
    def _merge_masks(masks, text_mask: Optional[torch.Tensor]):
        merged = dict(masks) if isinstance(masks, dict) else {}
        if text_mask is not None:
            merged["text"] = text_mask
        return merged if len(merged) > 0 else None

    @staticmethod
    def _merge_valid_lengths(valid_lengths, text_mask: Optional[torch.Tensor]):
        if isinstance(valid_lengths, dict):
            merged = dict(valid_lengths)
        elif valid_lengths is None:
            merged = {}
        else:
            return valid_lengths
        if text_mask is not None:
            merged["text"] = text_mask.sum(dim=1).long()
        return merged if len(merged) > 0 else None

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        labels = kwargs.get("labels", None)
        masks = kwargs.get("masks", None)
        valid_lengths = kwargs.get("valid_lengths", None)
        dbg_ctx = kwargs.get("dbg_ctx", None)

        ff_out = self.ff_encoder(
            text_x,
            audio_x,
            video_x,
            return_aux=self.return_debug_tensors,
        )
        if self.return_debug_tensors:
            z_t, z_a, z_v, ff_aux = ff_out
        else:
            z_t, z_a, z_v = ff_out
            ff_aux = None

        text_mask = self._extract_text_mask(text_x, masks)
        if text_mask is not None:
            text_mask = text_mask.to(z_t.device)
            text_mask = self._align_mask(text_mask, z_t.size(1))

        merged_masks = self._merge_masks(masks, text_mask)
        merged_lengths = self._merge_valid_lengths(valid_lengths, text_mask)

        out = super().forward(
            z_t,
            z_a,
            z_v,
            labels=labels,
            masks=merged_masks,
            valid_lengths=merged_lengths,
            dbg_ctx=dbg_ctx,
        )

        if self.return_debug_tensors:
            out["z_T_ff"] = z_t
            out["z_A_ff"] = z_a
            out["z_V_ff"] = z_v
            if ff_aux is not None:
                out.update(ff_aux)
        return out
