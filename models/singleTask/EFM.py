# -*- coding: utf-8 -*-
"""
EmotionFlow: 情感流 + 证据曲线 + 时间权重 + 先验/后验权重 + 融合输出（回归）
"""
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= NaN/Inf 守护工具 =========
_DBG_CTX = None

def _set_dbg_ctx(ctx):
    global _DBG_CTX
    _DBG_CTX = ctx

def _assert_finite(name: str, x: torch.Tensor):
    """检查张量是否为有限值，若出现 NaN/Inf 立即抛出，打印诊断信息。"""
    if torch.isfinite(x).all():
        return
    nan_cnt = torch.isnan(x).sum().item()
    inf_cnt = torch.isinf(x).sum().item()
    nan_min = torch.nanmin(x).item()
    nan_max = torch.nanmax(x).item()

    ctx = _DBG_CTX or {}
    epoch = ctx.get("epoch")
    batch_idx = ctx.get("batch_idx")
    mode = ctx.get("mode")
    ids = ctx.get("ids")
    indices = ctx.get("indices")
    text_mask_sum = ctx.get("text_mask_sum")
    audio_lengths = ctx.get("audio_lengths")
    vision_lengths = ctx.get("vision_lengths")
    bad_idx = None
    try:
        bad_mask = ~torch.isfinite(x)
        if x.dim() >= 1:
            bad_idx = bad_mask.view(bad_mask.size(0), -1).any(dim=1).nonzero(as_tuple=False).view(-1).tolist()
    except Exception:
        bad_idx = None

    id_list = None
    if ids is not None and bad_idx is not None:
        try:
            id_list = [ids[i] for i in bad_idx]
        except Exception:
            id_list = None

    idx_list = None
    if indices is not None and bad_idx is not None:
        try:
            idx_list = [indices[i] for i in bad_idx]
        except Exception:
            idx_list = None

    msg = (
        f"[NaNGuard] {name} non-finite detected | "
        f"mode={mode} epoch={epoch} batch={batch_idx} "
        f"shape={tuple(x.shape)} "
        f"nan={nan_cnt} inf={inf_cnt} "
        f"nanmin={nan_min} nanmax={nan_max} "
        f"bad_idx={bad_idx} ids={id_list} indices={idx_list} "
        f"text_mask_sum={text_mask_sum} audio_lengths={audio_lengths} vision_lengths={vision_lengths}"
    )
    raise RuntimeError(msg)

# 兼容包内相对导入
try:
    from ..subNets import BertTextEncoder
except ImportError:  # pragma: no cover
    from models.subNets import BertTextEncoder

__all__ = ["EmotionFlow", "_get"]


def _get(args, name, default=None):
    """兼容 edict / dict / Namespace 的安全取值"""
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    if hasattr(args, name):
        return getattr(args, name)
    try:
        return args[name]
    except Exception:
        return default


class EmotionFlow(nn.Module):
    """
    EmotionFlow 回归模型（单任务）
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # ---------- 基本配置 ----------
        feat_dims = _get(args, "feature_dims", [768, 5, 20])
        text_in, audio_in, video_in = feat_dims[:3]

        self.use_bert = bool(_get(args, "use_bert", False))
        self.use_finetune = bool(_get(args, "use_finetune", False))
        self.transformers = _get(args, "transformers", "bert")
        self.pretrained = _get(args, "pretrained", "bert-base-uncased")

        self.hidden_dim = int(_get(args, "ef_hidden", 128))          # D
        self.score_hidden = int(_get(args, "score_hidden", 128))
        self.dropout_p = float(_get(args, "ef_dropout", 0.10))

        # ---------- EmotionFlow 相关超参 ----------
        self.ema_beta = float(_get(args, "ema_beta", _get(args, "ema_alpha", 0.90)))
        self.evidence_k = int(_get(args, "evidence_k", _get(args, "K", 5)))
        self.evidence_k_eval = int(_get(args, "evidence_k_eval", 5))
        self.evidence_eta = float(_get(args, "evidence_eta", _get(args, "eta", 0.10)))
        self.evidence_temp = float(_get(args, "evidence_temp", _get(args, "Te", 1.0)))
        self.omega_kappa = float(_get(args, "omega_kappa", _get(args, "kappa", 5.0)))
        self.dirichlet_c = float(_get(args, "dirichlet_c", _get(args, "c", 1.0)))
        self.mc_dropout_eval = bool(_get(args, "mc_dropout_eval", False))
        self.evidence_use_net = bool(_get(args, "evidence_use_net", True))
        self.evidence_hidden = int(_get(args, "evidence_hidden", 64))
        self.evidence_area_ratio = bool(_get(args, "evidence_area_ratio", True))
        self.area_eps = float(_get(args, "area_eps", 1e-6))
        self.w_smooth_lambda = float(_get(args, "w_smooth_lambda", 0.0))
        self.return_debug_tensors = bool(_get(args, "return_debug_tensors", True))

        self.pooling = _get(args, "pooling", "mean")
        self.label_clip = float(_get(args, "label_clip", 0.0))
        # Method-level switches (keep architecture and interfaces unchanged).
        self.gate_detach_mode = str(_get(args, "gate_detach_mode", "none")).lower()
        self.weight_constraint_mode = str(_get(args, "weight_constraint_mode", "none")).lower()
        self.weight_constraint_min = float(_get(args, "weight_constraint_min", 0.0))
        self.weight_constraint_max = float(_get(args, "weight_constraint_max", 0.0))

        # ---------- 文本编码 ----------
        if self.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=self.use_finetune,
                transformers=self.transformers,
                pretrained=self.pretrained,
            )

        # ---------- 模态投影到统一维度 D ----------
        self.proj_t = nn.Identity() if text_in == self.hidden_dim else nn.Linear(text_in, self.hidden_dim)
        self.proj_a = nn.Identity() if audio_in == self.hidden_dim else nn.Linear(audio_in, self.hidden_dim)
        self.proj_v = nn.Identity() if video_in == self.hidden_dim else nn.Linear(video_in, self.hidden_dim)

        # ---------- 情感强度 / 极性头 ----------
        self.s_head = self._make_mlp(self.hidden_dim, self.score_hidden, 1, self.dropout_p)
        self.p_head = self._make_mlp(self.hidden_dim, self.score_hidden, 1, self.dropout_p)

        # ---------- 基准头 & 条件头 ----------
        # 基准头：e_t -> (mu0, logsig0)
        self.base_head = self._make_mlp(4, self.score_hidden, 2, self.dropout_p)
        # 条件头：u_m -> (mu_m, logsig_m)
        self.cond_heads = nn.ModuleDict({
            "T": self._make_mlp(self.hidden_dim + 4, self.score_hidden, 2, self.dropout_p),
            "A": self._make_mlp(self.hidden_dim + 4, self.score_hidden, 2, self.dropout_p),
            "V": self._make_mlp(self.hidden_dim + 4, self.score_hidden, 2, self.dropout_p),
        })

        # ---------- EvidenceNet（方案A：训练用 y 生成 f_target，融合用 f_pred） ----------
        self.evidence_in_dim = 1 + 1 + 1 + 4  # [mu_mean, mu_std, mu0, e]
        self.evidence_heads = nn.ModuleDict({
            "T": self._make_mlp(self.evidence_in_dim, self.evidence_hidden, 1, self.dropout_p),
            "A": self._make_mlp(self.evidence_in_dim, self.evidence_hidden, 1, self.dropout_p),
            "V": self._make_mlp(self.evidence_in_dim, self.evidence_hidden, 1, self.dropout_p),
        })
        # ---------- 情感先验（Dirichlet prior） ----------
        self.prior_head = self._make_mlp(4, self.score_hidden, 3, self.dropout_p)

        # ---------- 融合回归头 ----------
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, 1),
        )

        # 注意力池化（可选）
        if self.pooling == "attn":
            self.attn_pool = nn.Linear(self.hidden_dim, 1)

    def _make_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _align_seq(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """序列按时间线性插值到指定长度"""
        if x.size(1) == target_len:
            return x
        x_t = x.transpose(1, 2)  # [B, D, T]
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_t.transpose(1, 2)  # [B, T, D]

    def _lengths_to_mask(self, lengths: Union[torch.Tensor, list], max_len: int, device) -> torch.Tensor:
        if lengths is None:
            return None
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=device)
        lengths = lengths.to(device).view(-1)
        steps = torch.arange(max_len, device=device).unsqueeze(0)  # [1, T]
        mask = steps < lengths.unsqueeze(1)  # [B, T]
        return mask

    def _masked_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """对时间维做 mask mean，mask: [B,T] or [B,T,1]"""
        if mask is None:
            return x.mean(dim=1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.float()
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return (x * mask).sum(dim=1) / denom

    def _presence(self, x: torch.Tensor, mask: Optional[torch.Tensor], lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """判断模态是否缺失（返回 [B] 0/1）"""
        if mask is not None:
            mask = mask.to(x.device)
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            present = (mask.sum(dim=1) > 0).float()
            return present
        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, device=x.device)
            lengths = lengths.to(x.device)
            present = (lengths.view(-1) > 0).float()
            return present
        # 全零视作缺失
        present = (x.abs().sum(dim=(1, 2)) > 1e-6).float()
        return present

    def _ema_delta(self, s: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EMA 平滑 + 变化残差
        这里 Δs_t,Δp_t 是“相对历史趋势的增量/残差”，不是再求一次变化率；
        后面 e_t-e_{t-1} 才是“二阶变化/加速度”，用于判断情绪波动是否突然变大。
        """
        B, T, _ = s.shape
        s_bar = torch.zeros_like(s)
        p_bar = torch.zeros_like(p)
        ds = torch.zeros_like(s)
        dp = torch.zeros_like(p)
        s_prev = torch.zeros_like(s[:, 0:1])
        p_prev = torch.zeros_like(p[:, 0:1])
        for t in range(T):
            s_bar[:, t:t+1] = self.ema_beta * s_prev + (1.0 - self.ema_beta) * s[:, t:t+1]
            p_bar[:, t:t+1] = self.ema_beta * p_prev + (1.0 - self.ema_beta) * p[:, t:t+1]
            ds[:, t:t+1] = s[:, t:t+1] - s_prev
            dp[:, t:t+1] = p[:, t:t+1] - p_prev
            s_prev = s_bar[:, t:t+1]
            p_prev = p_bar[:, t:t+1]
        return ds, dp

    def _time_weights(self, e: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """基于 e(t)-e(t-1) 的时间权重 ω_t"""
        B, T, _ = e.shape
        if T == 1:
            return torch.ones(B, 1, 1, device=e.device)
        diff = e[:, 1:] - e[:, :-1]  # [B, T-1, De]
        d_e = torch.norm(diff, p=2, dim=-1, keepdim=True)  # [B, T-1, 1]
        pad = torch.zeros(B, 1, 1, device=e.device)
        d_e = torch.cat([pad, d_e], dim=1)  # [B, T, 1]
        if mask is None:
            omega = torch.softmax(self.omega_kappa * d_e, dim=1)
            return omega
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.float()  # [B, T, 1]

        valid = (mask.sum(dim=1).squeeze(-1) > 0)  # [B]
        zero_len = ~valid  # [B]

        # 对 valid 样本：masked softmax，padding 置 -inf，避免 0/0
        d_e_masked = d_e.masked_fill(mask == 0, -1e9)
        omega = torch.zeros_like(d_e)
        if valid.any():
            omega_valid = torch.softmax(self.omega_kappa * d_e_masked[valid], dim=1)
            denom = omega_valid.sum(dim=1, keepdim=True).clamp_min(1e-6)
            omega_valid = omega_valid / denom
            omega[valid] = omega_valid

        # 对 zero_len 样本：直接设为均匀分布，彻底避免 softmax 0/0
        if zero_len.any():
            omega[zero_len] = 1.0 / T
            idxs = zero_len.nonzero(as_tuple=False).view(-1).tolist()
            ctx = _DBG_CTX or {}
            ids = ctx.get("ids")
            id_list = None
            if ids is not None:
                try:
                    id_list = [ids[i] for i in idxs]
                except Exception:
                    id_list = None
            print(
                f"[NaNGuard] _time_weights zero_len: {len(idxs)}/{B} "
                f"idxs={idxs} ids={id_list} "
                f"text_shape={ctx.get('text_shape')} "
                f"audio_shape={ctx.get('audio_shape')} "
                f"vision_shape={ctx.get('vision_shape')} "
                f"valid_lengths={ctx.get('valid_lengths')}"
            )

        _assert_finite("omega", omega)
        return omega

    def _evidence_target(
        self,
        mu0: torch.Tensor,
        y: torch.Tensor,
        z_m: torch.Tensor,
        e: torch.Tensor,
        head: nn.Module,
        present: torch.Tensor,
    ) -> torch.Tensor:
        """
        训练用：基于标签的目标证据曲线 f_target(t)
        """
        if y.dim() == 2:
            y = y.unsqueeze(1)
        y = y.to(mu0.device)
        err0 = torch.abs(mu0 - y)  # [B, T, 1]

        K_eff = self.evidence_k if self.training else max(1, self.evidence_k_eval)
        deltas = []
        for _ in range(K_eff):
            # 目标证据允许在训练/评估中使用同样的扰动策略
            z_k = F.dropout(z_m, p=self.dropout_p, training=True)
            u_k = torch.cat([z_k, e], dim=-1)  # [B, T, D+De]
            mu_k = head(u_k)[..., 0:1]
            err_k = torch.abs(mu_k - y)
            deltas.append(err0 - err_k)
        deltas = torch.stack(deltas, dim=0)  # [K, B, T, 1]
        mean_delta = deltas.mean(dim=0)
        std_delta = deltas.std(dim=0, unbiased=False)
        g = mean_delta - self.evidence_eta * std_delta
        f = F.softplus(g / self.evidence_temp)
        # 缺失模态直接置零
        f = f * present.view(-1, 1, 1)
        _assert_finite("f_target", f)
        return f

    def _evidence_stats(self, z_m: torch.Tensor, e: torch.Tensor, head: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成 EvidenceNet 的输入统计量：mu_mean, mu_std
        """
        use_dropout = self.training or self.mc_dropout_eval
        K_eff = self.evidence_k if self.training else max(1, self.evidence_k_eval)
        if not use_dropout:
            K_eff = 1
        mu_samples = []
        for _ in range(K_eff):
            if use_dropout:
                z_k = F.dropout(z_m, p=self.dropout_p, training=True)
            else:
                z_k = z_m
            u_k = torch.cat([z_k, e], dim=-1)  # [B, T, D+De]
            mu_k = head(u_k)[..., 0:1]
            mu_samples.append(mu_k)
        mu_stack = torch.stack(mu_samples, dim=0)  # [K, B, T, 1]
        if (not self.training) and self.mc_dropout_eval and use_dropout and K_eff > 1:
            spread = (mu_stack - mu_stack[0:1]).abs().amax()
            if torch.isfinite(spread) and spread.item() <= 1e-12:
                print("[Evidence] mc_dropout_eval=True but all MC samples are identical.")
        mu_mean = mu_stack.mean(dim=0)
        mu_std = mu_stack.std(dim=0, unbiased=False)
        # 数值保护：std 太小会导致后续 1/0，太大易溢出
        mu_std = mu_std.clamp(min=1e-8, max=1e3)
        _assert_finite("mu_mean", mu_mean)
        _assert_finite("mu_std", mu_std)
        return mu_mean, mu_std

    def _evidence_area_ratio(
        self,
        f_tav: torch.Tensor,
        mask: Optional[torch.Tensor],
        pres: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cumulative trapezoid area A_m(t) and area ratio across modalities.
        f_tav: [B, T, 3], non-negative evidence curves.
        """
        eps = self.area_eps
        f = f_tav.clamp_min(0.0)
        prev = torch.cat([f[:, 0:1, :], f[:, :-1, :]], dim=1)
        area_step = 0.5 * (prev + f)  # dt = 1
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            area_step = area_step * mask.float()
        area = torch.cumsum(area_step, dim=1)  # [B, T, 3]
        if pres is not None:
            area = area * pres.view(area.size(0), 1, 3)
        denom = area.sum(dim=-1, keepdim=True)  # [B, T, 1]
        ratio = area / denom.clamp_min(eps)

        if pres is None:
            fallback = torch.full_like(ratio, 1.0 / ratio.size(-1))
        else:
            pres_t = pres.float().view(pres.size(0), 1, 3)
            fallback = pres_t / pres_t.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ratio = torch.where((denom <= eps).expand_as(ratio), fallback, ratio)
        _assert_finite("A_tav", area)
        _assert_finite("A_ratio_tav", ratio)
        return area, ratio

    def _apply_gate_detach(self, s_tav: torch.Tensor, omega: torch.Tensor, base_curve: torch.Tensor) -> torch.Tensor:
        mode = self.gate_detach_mode
        if mode in ("none", ""):
            return s_tav
        if mode in ("partial", "partial_detach"):
            # Keep evidence branch trainable while detaching temporal gate branch.
            return omega.detach() * base_curve
        if mode in ("full", "full_detach"):
            # Fully detach weighting branch to avoid unstable updates.
            return s_tav.detach()
        return s_tav

    def _apply_weight_constraint(self, w: torch.Tensor) -> torch.Tensor:
        mode = self.weight_constraint_mode
        if mode in ("none", ""):
            return w
        if mode == "softplus":
            w = F.softplus(w)
        elif mode == "clamp":
            lo = float(self.weight_constraint_min)
            hi = float(self.weight_constraint_max)
            if hi > lo > 0.0:
                w = torch.clamp(w, min=lo, max=hi)
            elif lo > 0.0:
                w = torch.clamp(w, min=lo)
            elif hi > 0.0:
                w = torch.clamp(w, max=hi)
        return w

    def _pool(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """时间聚合"""
        if self.pooling == "attn":
            scores = self.attn_pool(x).squeeze(-1)  # [B, T]
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            alpha = torch.softmax(scores, dim=1)  # [B, T]
            return (alpha.unsqueeze(-1) * x).sum(dim=1)
        # 默认 mean pooling
        return self._masked_mean(x, mask)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        """
        输入:
          text_x: [B, T, D] 或 BERT 输入 [B, 3, T]
          audio_x: [B, T, D]
          video_x: [B, T, D]
        输出:
          dict, 至少包含 'M' -> [B, 1]
        """
        labels = kwargs.get("labels", None)
        dbg_ctx = kwargs.get("dbg_ctx", None)
        _set_dbg_ctx(dbg_ctx)
        masks = kwargs.get("masks", None)
        valid_lengths = kwargs.get("valid_lengths", None)

        # ---------- 解包模态输入 ----------
        text = text_x
        audio = audio_x[0] if isinstance(audio_x, (tuple, list)) else audio_x
        video = video_x[0] if isinstance(video_x, (tuple, list)) else video_x

        audio_aux = audio_x[1] if isinstance(audio_x, (tuple, list)) and len(audio_x) > 1 else None
        video_aux = video_x[1] if isinstance(video_x, (tuple, list)) and len(video_x) > 1 else None

        # ---------- 文本编码 ----------
        text_mask = None
        if isinstance(masks, dict):
            text_mask = masks.get("text", masks.get("text_mask", None))

        if self.use_bert and text.dim() == 3 and text.size(1) == 3:
            # text: [B, 3, T]
            text_mask = text[:, 1, :].float() if text_mask is None else text_mask
            h_t = self.text_model(text)  # [B, T, D_t]
        else:
            h_t = text  # [B, T, D_t]

        if h_t.dim() == 2:
            h_t = h_t.unsqueeze(1)

        B, T, _ = h_t.shape
        if text_mask is not None:
            text_mask = text_mask.to(h_t.device)

        # 若提供了 text 长度，则生成 mask
        if text_mask is None and isinstance(valid_lengths, dict):
            text_len = valid_lengths.get("text", None)
            if text_len is not None:
                text_mask = self._lengths_to_mask(text_len, T, h_t.device)

        # ---------- 对齐 A/V 到文本时间长度 ----------
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if video.dim() == 2:
            video = video.unsqueeze(1)
        if audio.size(1) != T:
            audio = self._align_seq(audio, T)
        if video.size(1) != T:
            video = self._align_seq(video, T)

        # ---------- 统一维度投影 ----------
        z_T = self.proj_t(h_t)      # [B, T, D]
        z_A = self.proj_a(audio)    # [B, T, D]
        z_V = self.proj_v(video)    # [B, T, D]

        # ---------- 模态缺失检测 ----------
        audio_mask = None
        video_mask = None
        if isinstance(masks, dict):
            audio_mask = masks.get("audio", masks.get("audio_mask", None))
            video_mask = masks.get("vision", masks.get("vision_mask", None))

        if isinstance(valid_lengths, dict):
            audio_aux = valid_lengths.get("audio", audio_aux)
            video_aux = valid_lengths.get("vision", video_aux)

        pres_T = torch.ones(B, device=z_T.device)
        pres_A = self._presence(z_A, audio_mask, audio_aux)
        pres_V = self._presence(z_V, video_mask, video_aux)
        pres = torch.stack([pres_T, pres_A, pres_V], dim=1)  # [B, 3]

        # 缺失模态的表征置零，避免污染融合
        z_A = z_A * pres_A.view(B, 1, 1)
        z_V = z_V * pres_V.view(B, 1, 1)

        # ---------- EmotionFlow 信号 ----------
        # Step B2: 强度/极性
        s = F.softplus(self.s_head(z_T))  # [B, T, 1]
        p = torch.tanh(self.p_head(z_T))  # [B, T, 1]
        _assert_finite("s", s)
        _assert_finite("p", p)
        # Step B3: EMA + 残差
        ds, dp = self._ema_delta(s, p)    # [B, T, 1]
        _assert_finite("ds", ds)
        _assert_finite("dp", dp)
        # Step B4: e(t)
        e = torch.cat([ds, dp, s, p], dim=-1)  # [B, T, 4]
        _assert_finite("e", e)

        # ---------- 基准头 ----------
        base_out = self.base_head(e)      # [B, T, 2]
        mu0 = base_out[..., 0:1]          # [B, T, 1]
        logsig0 = base_out[..., 1:2]      # [B, T, 1]
        _assert_finite("mu0", mu0)

        # ---------- 条件头（非扰动版本） ----------
        u_T = torch.cat([z_T, e], dim=-1)  # [B, T, D+4]
        u_A = torch.cat([z_A, e], dim=-1)
        u_V = torch.cat([z_V, e], dim=-1)

        out_T = self.cond_heads["T"](u_T)  # [B, T, 2]
        out_A = self.cond_heads["A"](u_A)
        out_V = self.cond_heads["V"](u_V)

        mu_T, logsig_T = out_T[..., 0:1], out_T[..., 1:2]
        mu_A, logsig_A = out_A[..., 0:1], out_A[..., 1:2]
        mu_V, logsig_V = out_V[..., 0:1], out_V[..., 1:2]
        _assert_finite("mu_T", mu_T)
        _assert_finite("mu_A", mu_A)
        _assert_finite("mu_V", mu_V)

        # ---------- 证据曲线（融合只用 f_pred，训练可用 f_target 监督） ----------
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.view(-1, 1)
            if self.label_clip > 0:
                labels = torch.clamp(labels, -self.label_clip, self.label_clip)

        if self.evidence_use_net:
            mu_T_mean, mu_T_std = self._evidence_stats(z_T, e, self.cond_heads["T"])
            mu_A_mean, mu_A_std = self._evidence_stats(z_A, e, self.cond_heads["A"])
            mu_V_mean, mu_V_std = self._evidence_stats(z_V, e, self.cond_heads["V"])

            evid_T = torch.cat([mu_T_mean, mu_T_std, mu0, e], dim=-1)  # [B, T, 7]
            evid_A = torch.cat([mu_A_mean, mu_A_std, mu0, e], dim=-1)
            evid_V = torch.cat([mu_V_mean, mu_V_std, mu0, e], dim=-1)

            f_T = F.softplus(self.evidence_heads["T"](evid_T)) * pres_T.view(-1, 1, 1)
            f_A = F.softplus(self.evidence_heads["A"](evid_A)) * pres_A.view(-1, 1, 1)
            f_V = F.softplus(self.evidence_heads["V"](evid_V)) * pres_V.view(-1, 1, 1)
            _assert_finite("f_T", f_T)
            _assert_finite("f_A", f_A)
            _assert_finite("f_V", f_V)

            # 训练时生成目标证据，用于 L_evid
            if labels is not None:
                f_T_target = self._evidence_target(mu0, labels, z_T, e, self.cond_heads["T"], pres_T)
                f_A_target = self._evidence_target(mu0, labels, z_A, e, self.cond_heads["A"], pres_A)
                f_V_target = self._evidence_target(mu0, labels, z_V, e, self.cond_heads["V"], pres_V)
            else:
                f_T_target = f_A_target = f_V_target = None
        else:
            # 兼容旧方案：直接用误差改进或方差证据
            f_T = self._evidence_target(mu0, labels, z_T, e, self.cond_heads["T"], pres_T) if labels is not None else torch.zeros_like(mu0)
            f_A = self._evidence_target(mu0, labels, z_A, e, self.cond_heads["A"], pres_A) if labels is not None else torch.zeros_like(mu0)
            f_V = self._evidence_target(mu0, labels, z_V, e, self.cond_heads["V"], pres_V) if labels is not None else torch.zeros_like(mu0)
            _assert_finite("f_T", f_T)
            _assert_finite("f_A", f_A)
            _assert_finite("f_V", f_V)
            f_T_target = f_A_target = f_V_target = None

        # ---------- 时间权重 ----------
        omega = self._time_weights(e, text_mask)  # [B, T, 1]

        # ---------- 逐时刻证据注入 ----------
        f_tav = torch.cat([f_T, f_A, f_V], dim=-1)  # [B, T, 3]
        A_tav, A_ratio_tav = self._evidence_area_ratio(f_tav, text_mask, pres)
        if self.evidence_area_ratio:
            # Use area ratio for modality competition at each timestep.
            base_curve = A_ratio_tav
            S_tav = omega * base_curve
        else:
            base_curve = f_tav
            S_tav = omega * base_curve
        S_tav = self._apply_gate_detach(S_tav, omega, base_curve)
        _assert_finite("S_tav", S_tav)

        # ---------- 逐时刻先验 + 后验 ----------
        alpha_prior_t = F.softplus(self.prior_head(e)) + 1.0  # [B, T, 3]
        _assert_finite("alpha_prior_t", alpha_prior_t)
        alpha_post_t = alpha_prior_t + self.dirichlet_c * S_tav  # [B, T, 3]
        alpha_post_t = alpha_post_t.clamp(min=1e-8)
        _assert_finite("alpha_post_t_raw", alpha_post_t)

        # 缺失模态权重压低（广播到时间维）
        eps = 1e-6
        pres_t = pres.view(B, 1, 3)
        alpha_post_t = alpha_post_t * pres_t + eps * (1.0 - pres_t)
        _assert_finite("alpha_post_t", alpha_post_t)
        denom_w = alpha_post_t.sum(dim=-1, keepdim=True).clamp_min(eps)
        _assert_finite("denom_w", denom_w)
        w = alpha_post_t / denom_w  # [B, T, 3]
        w = self._apply_weight_constraint(w)
        _assert_finite("w", w)

        # padding 位置权重置为均匀（避免干扰）
        if text_mask is not None:
            mask3 = text_mask.unsqueeze(-1).float()  # [B, T, 1]
            w = w * mask3 + (1.0 / 3.0) * (1.0 - mask3)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(eps)
        _assert_finite("w_renorm", w)
        if w.dim() != 3 or w.size(0) != B or w.size(1) != T or w.size(2) != 3:
            raise RuntimeError(f"[ShapeGuard] w shape mismatch, got={tuple(w.shape)}, expect=({B}, {T}, 3)")
        w_sum = w.sum(dim=-1, keepdim=True)
        _assert_finite("w_sum", w_sum)

        # ---------- 权重平滑正则（可选） ----------
        w_smooth_loss = None
        if self.w_smooth_lambda > 0.0 and w.size(1) > 1:
            dw = w[:, 1:, :] - w[:, :-1, :]  # [B, T-1, 3]
            if text_mask is not None:
                mask = text_mask[:, 1:].unsqueeze(-1).float()  # [B, T-1, 1]
                dw = dw * mask
                denom = mask.sum() * w.size(-1) + 1e-6
                w_smooth_loss = dw.pow(2).sum() / denom
            else:
                w_smooth_loss = dw.pow(2).mean()

        # ---------- 融合（逐时刻动态权重） ----------
        wT = w[:, :, 0:1]
        wA = w[:, :, 1:2]
        wV = w[:, :, 2:3]
        z_fuse = wT * z_T + wA * z_A + wV * z_V  # [B, T, D]
        _assert_finite("z_fuse", z_fuse)

        # ---------- 时间聚合 + 回归 ----------
        h = self._pool(z_fuse, text_mask)        # [B, D]
        y_hat = self.fuse_mlp(h)                 # [B, 1]
        _assert_finite("y_hat", y_hat)

        # ---------- 输出 ----------
        res = {
            "M": y_hat,
            "mu0": mu0,
            "mu_T": mu_T,
            "mu_A": mu_A,
            "mu_V": mu_V,
            "f_T": f_T,
            "f_A": f_A,
            "f_V": f_V,
            "f_T_target": f_T_target,
            "f_A_target": f_A_target,
            "f_V_target": f_V_target,
            "omega": omega,
            "w": w,
            "loss_w_smooth": w_smooth_loss,
            # 兼容输出特征（可选）
            "Feature_t": self._masked_mean(z_T, text_mask),
            "Feature_a": self._masked_mean(z_A, text_mask),
            "Feature_v": self._masked_mean(z_V, text_mask),
            "Feature_f": self._masked_mean(z_fuse, text_mask),
        }
        if self.return_debug_tensors:
            res["e"] = e
            res["S_tav"] = S_tav
            res["A_tav"] = A_tav
            res["A_ratio_tav"] = A_ratio_tav
            res["alpha_post_t"] = alpha_post_t
            res["denom_w"] = denom_w
            res["w_sum"] = w_sum
            res["z_fuse"] = z_fuse
        return res
