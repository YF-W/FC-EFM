import logging
import os
import json
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils import MetricsTop, dict_to_str

logger = logging.getLogger('MSA')


def _get(args, name, default=None):
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


class EmotionFlowTrain:
    def __init__(self, args):
        self.args = args
        self.loss_name = str(_get(args, 'loss_name', 'mse')).lower()
        if args.train_mode == 'regression':
            if self.loss_name in ('huber', 'smoothl1', 'smooth_l1'):
                beta = float(_get(args, 'smooth_l1_beta', 1.0))
                self.criterion = nn.SmoothL1Loss(beta=beta)
            else:
                self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.evid_criterion = nn.L1Loss()
        self.dataset_name = str(_get(args, 'dataset_name', '')).lower()
        self.is_sims = self.dataset_name in ('sims', 'simsv2')

        self.lambda0 = float(_get(args, 'lambda0', 0.2))
        self.lambda1 = float(_get(args, 'lambda1', 0.2))
        self.lambda_e = float(_get(args, 'lambda_e', 0.1))
        self.method_lambda_schedule = str(_get(args, 'method_lambda_schedule', 'legacy')).lower()
        self.method_lambda_warm_ratio = float(_get(args, 'method_lambda_warm_ratio', 0.2))
        self.method_lambda_mid_ratio = float(_get(args, 'method_lambda_mid_ratio', 1.0))
        self.method_lambda_tail_ratio = float(_get(args, 'method_lambda_tail_ratio', 1.5))
        self.method_lambda_warm_portion = float(_get(args, 'method_lambda_warm_portion', 0.3))
        self.method_lambda_mid_portion = float(_get(args, 'method_lambda_mid_portion', 0.7))
        self.w_smooth_lambda = float(_get(args, 'w_smooth_lambda', 0.0))
        self.corr_loss_lambda = float(_get(args, 'corr_loss_lambda', 0.0))
        self.pred_std_lambda = float(_get(args, 'pred_std_lambda', 0.0))
        self.pred_std_target = float(_get(args, 'pred_std_target', 0.0))
        self.update_epochs = int(_get(args, 'update_epochs', 1))
        self.grad_clip = float(_get(args, 'grad_clip', -1.0))
        self.show_progress = bool(_get(args, 'show_progress', True))
        self.optimizer_name = str(_get(args, 'optimizer_name', 'adam')).lower()
        self.adam_betas = _get(args, 'adam_betas', [0.9, 0.999])
        self.adam_eps = float(_get(args, 'adam_eps', 1e-8))
        self.bert_lr_ratio = float(_get(args, 'bert_lr_ratio', 0.05))
        self.head_lr_ratio = float(_get(args, 'head_lr_ratio', 1.0))
        self.scheduler_name = str(_get(args, 'scheduler_name', 'none')).lower()
        self.warmup_ratio = float(_get(args, 'warmup_ratio', 0.0))
        self.min_lr_ratio = float(_get(args, 'min_lr_ratio', 0.1))
        self.use_param_group_decay = bool(_get(args, 'use_param_group_decay', True))
        self.use_amp = bool(_get(args, 'use_amp', False))

        self.nan_guard = bool(_get(args, 'nan_guard', True))
        self.nan_guard_raise = bool(_get(args, 'nan_guard_raise', True))
        self.skip_zero_len_eval = bool(_get(args, 'skip_zero_len_eval', True))
        self.skip_zero_len_train = bool(_get(args, 'skip_zero_len_train', False))
        self.save_last_good_ckpt = bool(_get(args, 'save_last_good_ckpt', False))
        self.last_good_ckpt_name = _get(args, 'last_good_ckpt_name', 'last_good.pth')
        self.check_param_finite = bool(_get(args, 'check_param_finite', True))
        self.autograd_detect_anomaly = bool(_get(args, 'autograd_detect_anomaly', False))
        self.loss_audit_enabled = bool(_get(args, 'loss_audit_enabled', False))
        self.loss_audit_batches = int(_get(args, 'loss_audit_batches', 3))
        self.dump_first_batch_stats = bool(_get(args, 'dump_first_batch_stats', False))
        self.metric_audit_enabled = bool(_get(args, 'metric_audit_enabled', False))
        self.metric_audit_dump_arrays = bool(_get(args, 'metric_audit_dump_arrays', False))
        self.metric_audit_warn_same_acc = bool(_get(args, 'metric_audit_warn_same_acc', True))
        self.metric_audit_raise_on_constant = bool(_get(args, 'metric_audit_raise_on_constant', False))
        self.metric_audit_pred_std_min = float(_get(args, 'metric_audit_pred_std_min', 1e-3))
        self.metric_audit_dir = _get(args, 'metric_audit_dir', '')
        self.metric_stall_window = int(_get(args, 'metric_stall_window', 3))
        self.metric_pred_calibrate = bool(_get(args, 'metric_pred_calibrate', False))
        self.metric_pred_calibrate_min_std = float(_get(args, 'metric_pred_calibrate_min_std', 1e-4))
        self.metric_pred_calibrate_max_scale = float(_get(args, 'metric_pred_calibrate_max_scale', 25.0))
        self.metric_pred_calibrate_mode = str(_get(args, 'metric_pred_calibrate_mode', 'std')).lower()
        self.metric_pred_calibrate_blend = float(_get(args, 'metric_pred_calibrate_blend', 0.0))
        self.metric_pred_calibrate_start_epoch = int(_get(args, 'metric_pred_calibrate_start_epoch', 1))
        self.metric_pred_clip = float(_get(args, 'metric_pred_clip', 0.0))
        self._metric_calib = {'scale': 1.0, 'bias': 0.0, 'pred_std': 1.0, 'label_std': 1.0}
        self._metric_history = {}
        self._eval_hash_history = {}

        # SIMS-only stabilization/audit knobs; no behavior change for other datasets.
        self.sims_label_clip = float(_get(args, 'sims_label_clip', 1.0))
        self.sims_loss_clip_preds = bool(_get(args, 'sims_loss_clip_preds', True))
        self.sims_pred_clip_mode = str(_get(args, 'sims_pred_clip_mode', 'ste')).lower()
        self.sims_loss_warmup_epochs = int(_get(args, 'sims_loss_warmup_epochs', 5))
        self.sims_grad_clip = float(_get(args, 'sims_grad_clip', self.grad_clip))
        self.sims_skip_nonfinite_batch = bool(_get(args, 'sims_skip_nonfinite_batch', True))
        self.sims_skip_zero_len_train = bool(_get(args, 'sims_skip_zero_len_train', True))
        self.sims_skip_zero_len_eval = bool(_get(args, 'sims_skip_zero_len_eval', True))
        self.sims_spike_loss_threshold = float(_get(args, 'sims_spike_loss_threshold', 1e4))
        self._sims_spike_train_logged = False
        self._sims_spike_eval_logged = False
        self._sims_shape_logged = set()
        self._sims_dataset_stats_logged = False

        if self.is_sims:
            self.skip_zero_len_train = self.sims_skip_zero_len_train
            self.skip_zero_len_eval = self.sims_skip_zero_len_eval
            self.grad_clip = self.sims_grad_clip

    def _sims_label_process(self, labels: torch.Tensor) -> torch.Tensor:
        if not self.is_sims or self.args.train_mode != 'regression':
            return labels
        c = max(1e-8, self.sims_label_clip)
        return torch.clamp(labels, -c, c)

    def _sims_pred_process_for_loss(self, preds: torch.Tensor) -> torch.Tensor:
        if not self.is_sims or self.args.train_mode != 'regression':
            return preds
        c = max(1e-8, self.sims_label_clip)
        if self.sims_pred_clip_mode == 'hard':
            return torch.clamp(preds, -c, c)
        if self.sims_pred_clip_mode == 'tanh':
            return c * torch.tanh(preds / c)
        # Default: STE clip keeps clipped forward values while avoiding zero gradients outside clip range.
        clipped = torch.clamp(preds, -c, c)
        return preds + (clipped - preds).detach()

    def _loss_inputs(self, preds: torch.Tensor, labels: torch.Tensor):
        labels_loss = labels
        preds_loss = preds
        if _get(self.args, 'label_clip', 0.0) > 0:
            c = float(self.args.label_clip)
            labels_loss = torch.clamp(labels_loss, -c, c)
        if self.is_sims and self.args.train_mode == 'regression':
            labels_loss = self._sims_label_process(labels_loss)
            if self.sims_loss_clip_preds:
                preds_loss = self._sims_pred_process_for_loss(preds_loss)
        return preds_loss, labels_loss

    def _log_sims_dataset_stats_once(self, dataloader):
        if (not self.is_sims) or self._sims_dataset_stats_logged:
            return
        ds = dataloader.get('train', None)
        if ds is None or not hasattr(ds, 'dataset'):
            return
        if not hasattr(ds.dataset, 'labels') or 'M' not in ds.dataset.labels:
            return
        y = np.asarray(ds.dataset.labels['M']).reshape(-1)
        if y.size == 0:
            return
        q = np.quantile(y, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]).tolist()
        logger.info(
            '[SIMS-AUDIT] label_dist n=%d min=%.6f max=%.6f mean=%.6f std=%.6f q=%s',
            int(y.size),
            float(np.min(y)),
            float(np.max(y)),
            float(np.mean(y)),
            float(np.std(y)),
            [round(float(v), 6) for v in q],
        )
        self._sims_dataset_stats_logged = True

    def _log_sims_shape_audit(self, mode, epoch, batch_idx, labels, outputs, loss_main):
        if not self.is_sims:
            return
        key = f'{mode}_{epoch}_{batch_idx}'
        if key in self._sims_shape_logged:
            return
        if batch_idx != 0 or epoch != 1:
            return
        preds = outputs['M']
        mu0 = outputs.get('mu0', None)
        muT = outputs.get('mu_T', None)
        muA = outputs.get('mu_A', None)
        muV = outputs.get('mu_V', None)
        mu0_mean = mu0.mean(dim=1) if mu0 is not None else None
        muT_mean = muT.mean(dim=1) if muT is not None else None
        muA_mean = muA.mean(dim=1) if muA is not None else None
        muV_mean = muV.mean(dim=1) if muV is not None else None
        reduction = getattr(self.criterion, 'reduction', 'unknown')
        logger.info(
            '[SIMS-AUDIT] mode=%s epoch=%d batch=%d reduction=%s labels_shape=%s y_hat_shape=%s '
            'mu0_shape=%s mu0_mean_shape=%s muT_mean_shape=%s muA_mean_shape=%s muV_mean_shape=%s '
            'broadcast_main=%s broadcast_mu0=%s loss_main=%.6f',
            mode,
            epoch,
            batch_idx,
            reduction,
            tuple(labels.shape),
            tuple(preds.shape),
            tuple(mu0.shape) if mu0 is not None else None,
            tuple(mu0_mean.shape) if mu0_mean is not None else None,
            tuple(muT_mean.shape) if muT_mean is not None else None,
            tuple(muA_mean.shape) if muA_mean is not None else None,
            tuple(muV_mean.shape) if muV_mean is not None else None,
            tuple(preds.shape) != tuple(labels.shape),
            (mu0_mean is not None and tuple(mu0_mean.shape) != tuple(labels.shape)),
            float(loss_main.item()) if torch.is_tensor(loss_main) else float(loss_main),
        )
        self._sims_shape_logged.add(key)

    def _log_sims_spike_once(self, mode, epoch, batch_idx, loss_main, labels_loss, preds_loss, meta):
        if not self.is_sims:
            return
        val = float(loss_main.item())
        if val < self.sims_spike_loss_threshold:
            return
        if mode == 'train' and self._sims_spike_train_logged:
            return
        if mode != 'train' and self._sims_spike_eval_logged:
            return
        logger.error(
            '[SIMS-SPIKE] mode=%s function=%s epoch=%d batch=%d loss_main=%.6f labels_shape=%s preds_shape=%s '
            'labels[min=%.6f max=%.6f mean=%.6f std=%.6f] preds[min=%.6f max=%.6f mean=%.6f std=%.6f] '
            'text_mask_sum=%s audio_lengths=%s vision_lengths=%s',
            mode,
            'EmotionFlowTrain.do_train' if mode == 'train' else 'EmotionFlowTrain.do_test',
            int(epoch),
            int(batch_idx),
            val,
            tuple(labels_loss.shape),
            tuple(preds_loss.shape),
            self._tensor_stats_dict(labels_loss)['min'],
            self._tensor_stats_dict(labels_loss)['max'],
            self._tensor_stats_dict(labels_loss)['mean'],
            self._tensor_stats_dict(labels_loss)['std'],
            self._tensor_stats_dict(preds_loss)['min'],
            self._tensor_stats_dict(preds_loss)['max'],
            self._tensor_stats_dict(preds_loss)['mean'],
            self._tensor_stats_dict(preds_loss)['std'],
            meta.get('text_mask_sum'),
            meta.get('audio_lengths'),
            meta.get('vision_lengths'),
        )
        if mode == 'train':
            self._sims_spike_train_logged = True
        else:
            self._sims_spike_eval_logged = True

    def _sanitize_metrics(self, metrics: dict, mode: str):
        if not isinstance(metrics, dict):
            return metrics
        for k, v in list(metrics.items()):
            if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                if self.is_sims:
                    logger.warning('[SIMS-Guard] metric non-finite -> 0 | mode=%s metric=%s value=%s', mode, k, v)
                    metrics[k] = 0.0
                else:
                    raise RuntimeError(f'[NaNGuard] metric non-finite: {k}={v} mode={mode}')
        return metrics

    @staticmethod
    def _corr_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = preds.view(-1).float()
        y = labels.view(-1).float()
        x = x - x.mean()
        y = y - y.mean()
        x_var = (x.pow(2).mean()).clamp_min(1e-8)
        y_var = (y.pow(2).mean()).clamp_min(1e-8)
        corr = (x * y).mean() / torch.sqrt(x_var * y_var)
        corr = torch.clamp(corr, -1.0, 1.0)
        return 1.0 - corr

    def _method_lambda_scale(self, epoch_idx: int, total_epochs: int) -> float:
        mode = self.method_lambda_schedule
        if mode in ("legacy", "none", ""):
            return 1.0
        total = max(1, int(total_epochs))
        p = float(epoch_idx) / float(total)
        p1 = min(max(float(self.method_lambda_warm_portion), 0.0), 1.0)
        p2 = min(max(float(self.method_lambda_mid_portion), p1), 1.0)
        v0 = float(self.method_lambda_warm_ratio)
        v1 = float(self.method_lambda_mid_ratio)
        v2 = float(self.method_lambda_tail_ratio)

        if p <= p1:
            return v0
        if p <= p2:
            span = max(1e-8, p2 - p1)
            t = (p - p1) / span
            return v0 + t * (v1 - v0)
        span = max(1e-8, 1.0 - p2)
        t = (p - p2) / span
        return v1 + t * (v2 - v1)

    @staticmethod
    def _np_stats(x: np.ndarray):
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return {'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan}
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
        }

    @staticmethod
    def _hist_dict(x: np.ndarray):
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return {}
        values, counts = np.unique(arr, return_counts=True)
        return {str(int(v)): int(c) for v, c in zip(values, counts)}

    @staticmethod
    def _bins10(x: np.ndarray):
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return {'edges': [], 'counts': []}
        counts, edges = np.histogram(arr, bins=10)
        return {
            'edges': [float(v) for v in edges.tolist()],
            'counts': [int(v) for v in counts.tolist()],
        }

    def _discretize_for_audit(self, preds: np.ndarray, labels: np.ndarray):
        p = np.asarray(preds).reshape(-1)
        y = np.asarray(labels).reshape(-1)
        ds = self.dataset_name
        out = {}
        if ds in ('mosi', 'mosei'):
            p7 = np.clip(p, -3.0, 3.0).astype(int)
            y7 = np.clip(y, -3.0, 3.0).astype(int)
            p5 = np.clip(p, -2.0, 2.0).astype(int)
            y5 = np.clip(y, -2.0, 2.0).astype(int)
            p3 = np.clip(p, -1.0, 1.0).astype(int)
            y3 = np.clip(y, -1.0, 1.0).astype(int)
            p2 = np.clip(p, 0.0, 1.0).astype(int)
            y2 = np.clip(y, 0.0, 1.0).astype(int)
        else:
            # SIMS / SIMSV2 thresholds (kept consistent with metricsTop)
            p = np.clip(p, -1.0, 1.0)
            y = np.clip(y, -1.0, 1.0)

            def _bucket(arr, edges):
                out_arr = arr.copy()
                for i in range(len(edges) - 1):
                    lo, hi = edges[i], edges[i + 1]
                    out_arr[np.logical_and(arr > lo, arr <= hi)] = i
                return out_arr.astype(int)

            p2 = _bucket(p, [-1.01, 0.0, 1.01])
            y2 = _bucket(y, [-1.01, 0.0, 1.01])
            p3 = _bucket(p, [-1.01, -0.1, 0.1, 1.01])
            y3 = _bucket(y, [-1.01, -0.1, 0.1, 1.01])
            p5 = _bucket(p, [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01])
            y5 = _bucket(y, [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01])
            p7 = _bucket(p, [-1.01, -0.7, -0.3, -0.1, 0.1, 0.3, 0.7, 1.01])
            y7 = _bucket(y, [-1.01, -0.7, -0.3, -0.1, 0.1, 0.3, 0.7, 1.01])

        out['pred_7'], out['true_7'] = p7, y7
        out['pred_5'], out['true_5'] = p5, y5
        out['pred_3'], out['true_3'] = p3, y3
        out['pred_2'], out['true_2'] = p2, y2
        return out

    def _get_metric_audit_dir(self):
        if self.metric_audit_dir:
            d = Path(self.metric_audit_dir)
        else:
            base = _get(self.args, 'res_save_dir', './results')
            run_id = str(_get(self.args, 'run_id', 'default'))
            d = Path(base) / '_metric_audit' / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _update_metric_calibration(self, pred_train: torch.Tensor, true_train: torch.Tensor, epoch: int):
        if (not self.metric_pred_calibrate) or self.args.train_mode != 'regression':
            return
        if int(epoch) < int(self.metric_pred_calibrate_start_epoch):
            return
        p = pred_train.detach().cpu().view(-1).numpy().astype(np.float64)
        y = true_train.detach().cpu().view(-1).numpy().astype(np.float64)
        p_mean, y_mean = float(np.mean(p)), float(np.mean(y))
        p_std = float(np.std(p))
        y_std = float(np.std(y))
        denom_std = max(self.metric_pred_calibrate_min_std, p_std)
        scale_std = y_std / denom_std
        # Optional linear-regression calibration; still train-only, no val/test labels involved.
        p_center = p - p_mean
        y_center = y - y_mean
        var_p = float(np.mean(p_center * p_center))
        cov_py = float(np.mean(p_center * y_center))
        denom_lr = max(self.metric_pred_calibrate_min_std ** 2, var_p)
        scale_lr = cov_py / denom_lr
        mode = self.metric_pred_calibrate_mode
        if mode in ('linreg', 'linear', 'ols', 'lr'):
            scale = scale_lr
        else:
            scale = scale_std
        blend = float(np.clip(self.metric_pred_calibrate_blend, 0.0, 1.0))
        if blend > 0.0:
            scale = (1.0 - blend) * scale + blend * scale_std
        scale = float(np.clip(scale, 1.0 / self.metric_pred_calibrate_max_scale, self.metric_pred_calibrate_max_scale))
        bias = y_mean - scale * p_mean
        self._metric_calib = {
            'scale': scale,
            'bias': bias,
            'pred_std': p_std,
            'label_std': y_std,
            'mode': mode,
            'scale_std': scale_std,
            'scale_lr': scale_lr,
            'blend': blend,
        }
        logger.info(
            '[MetricCalib] epoch=%s mode=%s scale=%.6f bias=%.6f train_pred_std=%.6f train_label_std=%.6f scale_std=%.6f scale_lr=%.6f blend=%.2f',
            int(epoch),
            mode,
            scale,
            bias,
            p_std,
            y_std,
            float(scale_std),
            float(scale_lr),
            blend,
        )

    def _apply_metric_calibration(self, pred_eval: torch.Tensor, mode: str):
        if (not self.metric_pred_calibrate) or self.args.train_mode != 'regression':
            return pred_eval
        scale = float(self._metric_calib.get('scale', 1.0))
        bias = float(self._metric_calib.get('bias', 0.0))
        out = pred_eval * scale + bias
        if self.metric_pred_clip > 0:
            c = float(self.metric_pred_clip)
            out = torch.clamp(out, -c, c)
        logger.info(
            '[MetricCalib] mode=%s epoch=%s apply scale=%.6f bias=%.6f clip=%.6f',
            mode,
            _get(self.args, 'cur_epoch', None),
            scale,
            bias,
            float(self.metric_pred_clip),
        )
        return out

    @staticmethod
    def _hash_vector(x: np.ndarray, n: int = 32):
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return 'empty'
        sample = arr[: min(n, arr.size)]
        payload = '|'.join([f'{float(v):.8f}' for v in sample.tolist()]).encode('utf-8')
        return hashlib.md5(payload).hexdigest()

    def _audit_eval(self, mode: str, epoch: int, pred: torch.Tensor, true: torch.Tensor, eval_results: dict):
        if not self.metric_audit_enabled:
            return

        p = pred.detach().cpu().view(-1).numpy()
        y = true.detach().cpu().view(-1).numpy()
        disc = self._discretize_for_audit(p, y)

        p_stats = self._np_stats(p)
        y_stats = self._np_stats(y)
        pred_hash = self._hash_vector(p)
        key = f'{mode.lower()}'
        prev_hash = self._eval_hash_history.get(key, None)
        self._eval_hash_history[key] = pred_hash
        same_hash = prev_hash == pred_hash if prev_hash is not None else False

        logger.info(
            '[MetricAudit] mode=%s epoch=%s pred[min=%.6f max=%.6f mean=%.6f std=%.6f] '
            'true[min=%.6f max=%.6f mean=%.6f std=%.6f] pred_hash=%s same_as_prev=%s',
            mode,
            epoch,
            p_stats['min'],
            p_stats['max'],
            p_stats['mean'],
            p_stats['std'],
            y_stats['min'],
            y_stats['max'],
            y_stats['mean'],
            y_stats['std'],
            pred_hash,
            same_hash,
        )
        logger.info(
            '[MetricAudit] mode=%s epoch=%s pred_hist_7=%s true_hist_7=%s pred_hist_5=%s true_hist_5=%s '
            'pred_hist_3=%s true_hist_3=%s pred_hist_2=%s true_hist_2=%s',
            mode,
            epoch,
            self._hist_dict(disc['pred_7']),
            self._hist_dict(disc['true_7']),
            self._hist_dict(disc['pred_5']),
            self._hist_dict(disc['true_5']),
            self._hist_dict(disc['pred_3']),
            self._hist_dict(disc['true_3']),
            self._hist_dict(disc['pred_2']),
            self._hist_dict(disc['true_2']),
        )
        logger.info(
            '[MetricAudit] mode=%s epoch=%s pred_bins10=%s true_bins10=%s',
            mode,
            epoch,
            self._bins10(p),
            self._bins10(y),
        )

        acc7 = eval_results.get('Acc_7', eval_results.get('acc_7', None))
        acc5 = eval_results.get('Acc_5', eval_results.get('acc_5', None))
        acc3 = eval_results.get('Acc_3', eval_results.get('acc_3', None))
        acc2 = eval_results.get('Acc_2', eval_results.get('acc_2', None))
        logger.info(
            '[MetricAudit] mode=%s epoch=%s metric_obj_ids acc7=%s acc5=%s acc3=%s acc2=%s values=(%s,%s,%s,%s)',
            mode,
            epoch,
            id(acc7),
            id(acc5),
            id(acc3),
            id(acc2),
            acc7,
            acc5,
            acc3,
            acc2,
        )

        metric_group = self._metric_history.setdefault(key, {})
        for mk in ('Acc_7', 'acc_7', 'Acc_5', 'acc_5', 'Acc_3', 'acc_3', 'Acc_2', 'acc_2', 'F1_score', 'Corr', 'MAE'):
            if mk in eval_results:
                metric_group.setdefault(mk, []).append(float(eval_results[mk]))

        if self.metric_audit_warn_same_acc and all(v is not None for v in (acc7, acc5, acc3)):
            if np.isclose(float(acc7), float(acc5), atol=1e-12) and np.isclose(float(acc5), float(acc3), atol=1e-12):
                logger.warning(
                    '[MetricAudit][WARN] mode=%s epoch=%s Acc_7/Acc_5/Acc_3 are identical: %.8f',
                    mode,
                    epoch,
                    float(acc7),
                )

        if self.metric_audit_raise_on_constant:
            for mk, values in metric_group.items():
                if len(values) < self.metric_stall_window:
                    continue
                tail = np.asarray(values[-self.metric_stall_window :], dtype=np.float64)
                if np.all(np.isfinite(tail)) and np.max(np.abs(tail - tail[0])) <= 1e-12:
                    raise RuntimeError(
                        f'[MetricAudit] mode={mode} metric={mk} constant for last {self.metric_stall_window} epochs: {tail.tolist()}'
                    )

        if mode.upper() == 'VAL' and p_stats['std'] < self.metric_audit_pred_std_min:
            logger.warning(
                '[MetricAudit][WARN] mode=%s epoch=%s pred std too small (%.8f < %.8f): potential collapse.',
                mode,
                epoch,
                p_stats['std'],
                self.metric_audit_pred_std_min,
            )

        if self.metric_audit_dump_arrays:
            out_dir = self._get_metric_audit_dir()
            run_id = str(_get(self.args, 'run_id', 'default'))
            stem = f'{run_id}_{self.dataset_name}_{mode.lower()}_epoch{int(epoch):03d}'
            np.savez_compressed(
                out_dir / f'{stem}.npz',
                preds=p.astype(np.float32),
                labels=y.astype(np.float32),
                pred_7=disc['pred_7'].astype(np.int32),
                true_7=disc['true_7'].astype(np.int32),
                pred_5=disc['pred_5'].astype(np.int32),
                true_5=disc['true_5'].astype(np.int32),
                pred_3=disc['pred_3'].astype(np.int32),
                true_3=disc['true_3'].astype(np.int32),
                pred_2=disc['pred_2'].astype(np.int32),
                true_2=disc['true_2'].astype(np.int32),
            )
            meta = {
                'mode': mode,
                'epoch': int(epoch),
                'dataset': self.dataset_name,
                'run_id': run_id,
                'pred_stats': p_stats,
                'true_stats': y_stats,
                'metrics': {k: float(v) for k, v in eval_results.items() if isinstance(v, (int, float, np.floating))},
                'pred_hist_7': self._hist_dict(disc['pred_7']),
                'true_hist_7': self._hist_dict(disc['true_7']),
                'pred_hist_5': self._hist_dict(disc['pred_5']),
                'true_hist_5': self._hist_dict(disc['true_5']),
                'pred_hist_3': self._hist_dict(disc['pred_3']),
                'true_hist_3': self._hist_dict(disc['true_3']),
                'pred_hist_2': self._hist_dict(disc['pred_2']),
                'true_hist_2': self._hist_dict(disc['true_2']),
                'pred_bins10': self._bins10(p),
                'true_bins10': self._bins10(y),
                'pred_hash': pred_hash,
                'same_hash_as_prev': same_hash,
            }
            with open(out_dir / f'{stem}.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    def _to_list(self, x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().view(-1).tolist()
        if isinstance(x, np.ndarray):
            return x.reshape(-1).tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def _tensor_minmax(self, x: torch.Tensor):
        finite = x[torch.isfinite(x)]
        if finite.numel() == 0:
            return float('nan'), float('nan')
        return finite.min().item(), finite.max().item()

    def _tensor_stats_dict(self, x: torch.Tensor):
        if x is None:
            return {'shape': None, 'min': float('nan'), 'max': float('nan'), 'mean': float('nan'), 'std': float('nan')}
        xd = x.detach().float()
        finite = xd[torch.isfinite(xd)]
        if finite.numel() == 0:
            return {'shape': tuple(x.shape), 'min': float('nan'), 'max': float('nan'), 'mean': float('nan'), 'std': float('nan')}
        return {
            'shape': tuple(x.shape),
            'min': finite.min().item(),
            'max': finite.max().item(),
            'mean': finite.mean().item(),
            'std': finite.std(unbiased=False).item() if finite.numel() > 1 else 0.0,
        }

    def _log_tensor_stats(self, mode, epoch, batch_idx, name, x):
        s = self._tensor_stats_dict(x)
        logger.info(
            '[TensorStat] mode=%s epoch=%s batch=%s name=%s shape=%s min=%.6g max=%.6g mean=%.6g std=%.6g',
            mode, epoch, batch_idx, name, s['shape'], s['min'], s['max'], s['mean'], s['std']
        )

    def _log_first_batch_snapshot(self, mode, epoch, batch_idx, labels, outputs):
        if not self.dump_first_batch_stats:
            return
        if epoch != 1 or batch_idx != 0:
            return
        self._log_tensor_stats(mode, epoch, batch_idx, 'labels', labels)
        self._log_tensor_stats(mode, epoch, batch_idx, 'y_hat', outputs.get('M', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'mu0', outputs.get('mu0', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'e', outputs.get('e', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'omega', outputs.get('omega', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'w', outputs.get('w', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'alpha_post_t', outputs.get('alpha_post_t', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'denom_w', outputs.get('denom_w', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'f_T', outputs.get('f_T', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'f_A', outputs.get('f_A', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'f_V', outputs.get('f_V', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'A_tav', outputs.get('A_tav', None))
        self._log_tensor_stats(mode, epoch, batch_idx, 'A_ratio_tav', outputs.get('A_ratio_tav', None))

    def _build_batch_meta(self, batch_data, text, audio, vision):
        meta = {
            'ids': self._to_list(batch_data.get('id', None)),
            'indices': self._to_list(batch_data.get('index', None)),
            'text_shape': tuple(text.shape),
            'audio_shape': tuple(audio.shape),
            'vision_shape': tuple(vision.shape),
            'text_mask_sum': None,
            'audio_lengths': self._to_list(batch_data.get('audio_lengths', None)),
            'vision_lengths': self._to_list(batch_data.get('vision_lengths', None)),
            'zero_len_idx': [],
        }
        text_mask = None
        if 'text_missing_mask' in batch_data:
            text_mask = batch_data['text_missing_mask']
        elif text.dim() == 3 and text.size(1) == 3:
            text_mask = text[:, 1, :]
        if text_mask is not None:
            text_mask = text_mask.detach()
            if torch.is_tensor(text_mask):
                mask_sum = text_mask.sum(dim=1).detach().cpu().view(-1)
                meta['text_mask_sum'] = mask_sum.tolist()
                meta['zero_len_idx'] = (mask_sum <= 0).nonzero(as_tuple=False).view(-1).tolist()
        return meta

    def _slice_optional(self, x, keep):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x[keep]
        if isinstance(x, np.ndarray):
            return x[keep.detach().cpu().numpy()]
        if isinstance(x, (list, tuple)):
            idx = keep.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            return [x[i] for i in idx]
        return x

    def _filter_zero_len_samples(self, text, audio, vision, labels, masks, valid_lengths, batch_data, meta, mode):
        zero_len_count = len(meta['zero_len_idx'])
        if zero_len_count == 0:
            return text, audio, vision, labels, masks, valid_lengths, batch_data, meta, 0

        bad_ids = [meta['ids'][i] for i in meta['zero_len_idx']] if meta['ids'] else None
        bad_indices = [meta['indices'][i] for i in meta['zero_len_idx']] if meta['indices'] else meta['zero_len_idx']
        logger.error(
            '[NaNGuard] zero_len samples | mode=%s epoch=%s idx=%s ids=%s text_mask_sum=%s '
            'audio_lengths=%s vision_lengths=%s text_shape=%s audio_shape=%s vision_shape=%s',
            mode,
            _get(self.args, 'cur_epoch', None),
            bad_indices,
            bad_ids,
            meta['text_mask_sum'],
            meta['audio_lengths'],
            meta['vision_lengths'],
            meta['text_shape'],
            meta['audio_shape'],
            meta['vision_shape'],
        )

        should_filter = (mode in ('VAL', 'TEST') and self.skip_zero_len_eval) or (mode == 'TRAIN' and self.skip_zero_len_train)
        if not should_filter:
            return text, audio, vision, labels, masks, valid_lengths, batch_data, meta, 0

        keep = torch.ones(text.size(0), dtype=torch.bool, device=text.device)
        keep[torch.tensor(meta['zero_len_idx'], device=text.device)] = False
        if keep.sum().item() == 0:
            return None, None, None, None, None, None, None, meta, zero_len_count

        text = text[keep]
        audio = audio[keep]
        vision = vision[keep]
        labels = labels[keep]
        if masks is not None:
            for k, v in list(masks.items()):
                masks[k] = self._slice_optional(v, keep)
        if valid_lengths is not None:
            for k, v in list(valid_lengths.items()):
                valid_lengths[k] = self._slice_optional(v, keep)
        filtered_batch = dict(batch_data)
        filtered_batch['id'] = self._slice_optional(batch_data.get('id', None), keep)
        filtered_batch['index'] = self._slice_optional(batch_data.get('index', None), keep)
        meta = self._build_batch_meta(filtered_batch, text, audio, vision)
        return text, audio, vision, labels, masks, valid_lengths, filtered_batch, meta, zero_len_count

    def _check_finite(self, name, x, mode, epoch, batch_idx, meta):
        if not self.nan_guard or x is None:
            return
        if torch.isfinite(x).all():
            return

        bad = ~torch.isfinite(x)
        nan_cnt = torch.isnan(x).sum().item()
        inf_cnt = torch.isinf(x).sum().item()
        min_val, max_val = self._tensor_minmax(x.detach())
        bad_idx = None
        if x.dim() >= 1:
            bad_idx = bad.view(bad.size(0), -1).any(dim=1).nonzero(as_tuple=False).view(-1).tolist()
        ids = [meta['ids'][i] for i in bad_idx] if bad_idx is not None and meta.get('ids') else None
        indices = [meta['indices'][i] for i in bad_idx] if bad_idx is not None and meta.get('indices') else bad_idx

        msg = (
            f'[NaNGuard] first non-finite tensor={name} mode={mode} epoch={epoch} batch={batch_idx} '
            f'shape={tuple(x.shape)} nan={nan_cnt} inf={inf_cnt} min={min_val} max={max_val} '
            f'bad_batch_idx={bad_idx} sample_ids={ids} sample_indices={indices} '
            f'text_mask_sum={meta.get("text_mask_sum")} audio_lengths={meta.get("audio_lengths")} '
            f'vision_lengths={meta.get("vision_lengths")} text_shape={meta.get("text_shape")} '
            f'audio_shape={meta.get("audio_shape")} vision_shape={meta.get("vision_shape")}'
        )
        logger.error(msg)
        raise RuntimeError(msg)

    def _save_last_good(self, model, reason):
        if not self.save_last_good_ckpt:
            return
        save_path = self.args.model_save_dir / str(self.args.dataset_name) / self.last_good_ckpt_name
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.error('[NaNGuard] saved last_good_ckpt=%s reason=%s', save_path, reason)

    def _check_grad_finite(self, model):
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return n
        return None

    def _check_param_finite(self, model, prefixes=None):
        if not self.check_param_finite:
            return None
        for n, p in model.named_parameters():
            if prefixes is not None and not any(n.startswith(pref) for pref in prefixes):
                continue
            if not torch.isfinite(p).all():
                return n
        return None

    def _parse_batch(self, batch_data):
        text = batch_data['text'].to(self.args.device)
        audio = batch_data['audio'].to(self.args.device)
        vision = batch_data['vision'].to(self.args.device)

        if _get(self.args, 'data_missing', False):
            if 'text_m' in batch_data:
                text = batch_data['text_m'].to(self.args.device)
            if 'audio_m' in batch_data:
                audio = batch_data['audio_m'].to(self.args.device)
            if 'vision_m' in batch_data:
                vision = batch_data['vision_m'].to(self.args.device)

        labels = batch_data['labels']['M'].to(self.args.device)
        if self.args.train_mode == 'classification':
            labels = labels.view(-1).long()
        else:
            labels = labels.view(-1, 1)

        masks = {}
        if _get(self.args, 'data_missing', False):
            if 'text_missing_mask' in batch_data:
                masks['text'] = batch_data['text_missing_mask'].to(self.args.device)
            if 'audio_mask' in batch_data:
                masks['audio'] = batch_data['audio_mask'].to(self.args.device)
            if 'vision_mask' in batch_data:
                masks['vision'] = batch_data['vision_mask'].to(self.args.device)

        valid_lengths = {}
        if not _get(self.args, 'need_data_aligned', False):
            if 'audio_lengths' in batch_data:
                valid_lengths['audio'] = batch_data['audio_lengths']
            if 'vision_lengths' in batch_data:
                valid_lengths['vision'] = batch_data['vision_lengths']

        if len(masks) == 0:
            masks = None
        if len(valid_lengths) == 0:
            valid_lengths = None
        return text, audio, vision, labels, masks, valid_lengths

    def _check_input_tensors(self, mode, epoch, step_idx, text, audio, vision, labels, meta):
        self._check_finite('text', text, mode, epoch, step_idx, meta)
        self._check_finite('audio', audio, mode, epoch, step_idx, meta)
        self._check_finite('vision', vision, mode, epoch, step_idx, meta)
        self._check_finite('labels', labels, mode, epoch, step_idx, meta)

    def _check_output_tensors(self, outputs, mode, epoch, step_idx, meta):
        for k in ['M', 'e', 'omega', 'f_T', 'f_A', 'f_V', 'A_tav', 'A_ratio_tav', 'S_tav', 'alpha_post_t', 'denom_w', 'w', 'w_sum', 'z_fuse']:
            if k in outputs:
                self._check_finite(k, outputs[k], mode, epoch, step_idx, meta)

    def _build_optimizer(self, model):
        base_lr = float(_get(self.args, 'learning_rate', 1e-3))
        weight_decay = float(_get(self.args, 'weight_decay', 0.0))
        betas = tuple(self.adam_betas) if isinstance(self.adam_betas, (list, tuple)) and len(self.adam_betas) == 2 else (0.9, 0.999)
        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if len(named_params) == 0:
            raise RuntimeError('[Optimizer] no trainable parameters found')

        if not self.use_param_group_decay:
            params = [p for _, p in named_params]
            if self.optimizer_name == 'adamw':
                return optim.AdamW(params, lr=base_lr, betas=betas, eps=self.adam_eps, weight_decay=weight_decay)
            return optim.Adam(params, lr=base_lr, betas=betas, eps=self.adam_eps, weight_decay=weight_decay)

        no_decay_terms = ('bias', 'layernorm.weight', 'layernorm.bias', 'layer_norm.weight', 'layer_norm.bias', 'norm.weight', 'norm.bias')
        groups = {}
        group_stats = {}
        for name, param in named_params:
            lname = name.lower()
            is_bert = ('text_model' in lname)
            is_head = any(k in lname for k in ('fuse_mlp', 'prior_head', 'evidence_heads', 'cond_heads', 'base_head', 's_head', 'p_head'))
            no_decay = any(t in lname for t in no_decay_terms)

            lr = base_lr * (self.bert_lr_ratio if is_bert else 1.0)
            if is_head:
                lr = lr * self.head_lr_ratio
            wd = 0.0 if no_decay else weight_decay
            key = (float(lr), float(wd))
            if key not in groups:
                groups[key] = []
                group_stats[key] = {'n_param': 0, 'n_bert': 0, 'n_head': 0}
            groups[key].append(param)
            group_stats[key]['n_param'] += int(param.numel())
            if is_bert:
                group_stats[key]['n_bert'] += int(param.numel())
            if is_head:
                group_stats[key]['n_head'] += int(param.numel())

        param_groups = []
        for (lr, wd), params in groups.items():
            param_groups.append({'params': params, 'lr': lr, 'weight_decay': wd})
            st = group_stats[(lr, wd)]
            logger.info(
                '[Optimizer] group lr=%.8f wd=%.6f params=%d bert_params=%d head_params=%d',
                float(lr),
                float(wd),
                int(st['n_param']),
                int(st['n_bert']),
                int(st['n_head']),
            )

        if self.optimizer_name == 'adamw':
            return optim.AdamW(param_groups, lr=base_lr, betas=betas, eps=self.adam_eps)
        return optim.Adam(param_groups, lr=base_lr, betas=betas, eps=self.adam_eps)

    def _build_scheduler(self, optimizer, total_updates: int):
        if self.scheduler_name in ('none', '', 'off'):
            return None
        total_updates = max(1, int(total_updates))
        warmup_steps = int(max(0, round(self.warmup_ratio * total_updates)))
        warmup_steps = min(warmup_steps, total_updates - 1) if total_updates > 1 else 0
        min_lr = float(np.clip(self.min_lr_ratio, 0.0, 1.0))

        def _lr_lambda(step):
            step = int(step)
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if total_updates <= warmup_steps:
                return 1.0
            progress = float(step - warmup_steps) / float(max(1, total_updates - warmup_steps))
            progress = float(np.clip(progress, 0.0, 1.0))
            if self.scheduler_name in ('linear', 'lin'):
                return min_lr + (1.0 - min_lr) * (1.0 - progress)
            # default cosine decay
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr + (1.0 - min_lr) * cosine

        logger.info(
            '[Scheduler] name=%s total_updates=%d warmup_steps=%d warmup_ratio=%.4f min_lr_ratio=%.4f',
            self.scheduler_name,
            total_updates,
            warmup_steps,
            float(self.warmup_ratio),
            float(min_lr),
        )
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    def do_train(self, model, dataloader, return_epoch_results=False):
        if self.autograd_detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        self._log_sims_dataset_stats_once(dataloader)

        optimizer = self._build_optimizer(model)
        steps_per_epoch = int(math.ceil(float(len(dataloader['train'])) / max(1, int(self.update_epochs))))
        total_updates = max(1, int(self.args.epochs) * max(1, steps_per_epoch))
        scheduler = self._build_scheduler(optimizer, total_updates)
        amp_enabled = bool(
            self.use_amp
            and str(self.args.device).startswith('cuda')
            and int(self.update_epochs) == 1
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        if self.use_amp and not amp_enabled:
            logger.info(
                '[AMP] requested but disabled | device=%s update_epochs=%d',
                str(self.args.device),
                int(self.update_epochs),
            )
        elif amp_enabled:
            logger.info('[AMP] enabled for EmotionFlow training')

        best_epoch = 0
        epoch_results = {'train': [], 'valid': [], 'test': []} if return_epoch_results else None
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        for epochs in range(1, self.args.epochs + 1):
            self.args.cur_epoch = epochs
            model.train()
            y_pred, y_true = [], []
            train_loss = 0.0
            train_loss_sum = 0.0
            left_epochs = self.update_epochs
            step_idx = 0
            good_batches = 0
            zero_len_seen_train = 0
            zero_len_filtered_train = 0
            zero_len_skipped_batches = 0
            sims_nonfinite_skipped_batches = 0
            audit_running_sum = 0.0
            audit_count = 0

            warmup_epochs = int(_get(self.args, 'loss_warmup_epochs', 0))
            if self.is_sims:
                warmup_epochs = int(_get(self.args, 'sims_loss_warmup_epochs', self.sims_loss_warmup_epochs))
            loss_scale = min(1.0, epochs / warmup_epochs) if warmup_epochs > 0 else 1.0
            method_scale = self._method_lambda_scale(epochs, int(self.args.epochs))
            logger.info(
                '[LossSchedule] epoch=%d/%d warmup_scale=%.4f method_scale=%.4f mode=%s',
                int(epochs),
                int(self.args.epochs),
                float(loss_scale),
                float(method_scale),
                self.method_lambda_schedule,
            )

            if epochs == 1 and _get(self.args, 'print_loss_breakdown', False):
                loss_sums = {'main': 0.0, 'l0': 0.0, 'lm': 0.0, 'le': 0.0, 'ls': 0.0, 'lc': 0.0, 'lstd': 0.0, 'cnt': 0}

            with tqdm(dataloader['train'], disable=not self.show_progress) as td:
                for batch_data in td:
                    if left_epochs == self.update_epochs:
                        optimizer.zero_grad(set_to_none=True)
                    left_epochs -= 1

                    text, audio, vision, labels, masks, valid_lengths = self._parse_batch(batch_data)
                    meta = self._build_batch_meta(batch_data, text, audio, vision)
                    zero_len_seen_train += len(meta.get('zero_len_idx', []))

                    text, audio, vision, labels, masks, valid_lengths, batch_data, meta, removed_zero_len = self._filter_zero_len_samples(
                        text, audio, vision, labels, masks, valid_lengths, batch_data, meta, 'TRAIN'
                    )
                    zero_len_filtered_train += int(removed_zero_len)
                    if text is None:
                        zero_len_skipped_batches += 1
                        step_idx += 1
                        left_epochs = self.update_epochs
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    if self.is_sims and self.sims_skip_nonfinite_batch and (not torch.isfinite(labels).all()):
                        logger.error(
                            '[SIMS-Guard] skip non-finite label batch | mode=train epoch=%d batch=%d ids=%s indices=%s',
                            epochs, step_idx, batch_data.get('id', None), batch_data.get('index', None)
                        )
                        sims_nonfinite_skipped_batches += 1
                        step_idx += 1
                        left_epochs = self.update_epochs
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    self._check_input_tensors('train', epochs, step_idx, text, audio, vision, labels, meta)

                    dbg_ctx = {
                        'mode': 'train',
                        'epoch': epochs,
                        'batch_idx': step_idx,
                        'ids': batch_data.get('id', None),
                        'indices': batch_data.get('index', None),
                        'text_shape': tuple(text.shape),
                        'audio_shape': tuple(audio.shape),
                        'vision_shape': tuple(vision.shape),
                        'text_mask_sum': meta.get('text_mask_sum'),
                        'audio_lengths': meta.get('audio_lengths'),
                        'vision_lengths': meta.get('vision_lengths'),
                        'valid_lengths': valid_lengths,
                    }
                    try:
                        with torch.cuda.amp.autocast(enabled=amp_enabled):
                            outputs = model(
                                text,
                                audio,
                                vision,
                                labels=None,
                                masks=masks,
                                valid_lengths=valid_lengths,
                                dbg_ctx=dbg_ctx,
                            )
                            self._check_output_tensors(outputs, 'train', epochs, step_idx, meta)
                            self._log_first_batch_snapshot('train', epochs, step_idx, labels, outputs)

                            preds = outputs['M']

                            preds_loss_main, labels_loss_main = self._loss_inputs(preds, labels)
                            loss_main = self.criterion(preds_loss_main, labels_loss_main)
                            loss0 = torch.tensor(0.0, device=preds.device)
                            if 'mu0' in outputs:
                                mu0_mean = outputs['mu0'].mean(dim=1)
                                mu0_loss, labels_loss_aux = self._loss_inputs(mu0_mean, labels)
                                loss0 = self.criterion(mu0_loss, labels_loss_aux)
                            lossm = torch.tensor(0.0, device=preds.device)
                            for k in ('mu_T', 'mu_A', 'mu_V'):
                                if k in outputs:
                                    mu_mean = outputs[k].mean(dim=1)
                                    mu_loss, labels_loss_aux = self._loss_inputs(mu_mean, labels)
                                    lossm = lossm + self.criterion(mu_loss, labels_loss_aux)

                            self._log_sims_shape_audit('train', epochs, step_idx, labels_loss_main, outputs, loss_main)
                            self._log_sims_spike_once('train', epochs, step_idx, loss_main, labels_loss_main, preds_loss_main, meta)

                            loss_evid = torch.tensor(0.0, device=preds.device)
                            if outputs.get('f_T_target', None) is not None:
                                loss_evid = loss_evid + self.evid_criterion(outputs['f_T'], outputs['f_T_target'])
                            if outputs.get('f_A_target', None) is not None:
                                loss_evid = loss_evid + self.evid_criterion(outputs['f_A'], outputs['f_A_target'])
                            if outputs.get('f_V_target', None) is not None:
                                loss_evid = loss_evid + self.evid_criterion(outputs['f_V'], outputs['f_V_target'])

                            loss_smooth = torch.tensor(0.0, device=preds.device)
                            if outputs.get('loss_w_smooth', None) is not None and self.w_smooth_lambda > 0.0:
                                loss_smooth = outputs['loss_w_smooth']

                            loss_corr = torch.tensor(0.0, device=preds.device)
                            if self.args.train_mode == 'regression' and self.corr_loss_lambda > 0.0:
                                loss_corr = self._corr_loss(preds_loss_main, labels_loss_main)

                            loss_std = torch.tensor(0.0, device=preds.device)
                            if self.args.train_mode == 'regression' and self.pred_std_lambda > 0.0:
                                pred_std = preds_loss_main.view(-1).std(unbiased=False)
                                if self.pred_std_target > 0.0:
                                    target_std = torch.tensor(float(self.pred_std_target), device=preds.device)
                                else:
                                    target_std = labels_loss_main.view(-1).std(unbiased=False).detach()
                                loss_std = (pred_std - target_std).pow(2)

                            loss = (
                                loss_main
                                + (self.lambda0 * loss_scale * method_scale) * loss0
                                + (self.lambda1 * loss_scale * method_scale) * lossm
                                + (self.lambda_e * loss_scale * method_scale) * loss_evid
                                + self.w_smooth_lambda * loss_smooth
                                + self.corr_loss_lambda * loss_corr
                                + self.pred_std_lambda * loss_std
                            )
                    except RuntimeError as ex:
                        self._save_last_good(model, f'non-finite forward output: {ex}')
                        optimizer.zero_grad(set_to_none=True)
                        if self.nan_guard_raise:
                            raise
                        step_idx += 1
                        continue
                    if not torch.isfinite(loss):
                        self._save_last_good(model, 'non-finite train loss')
                        optimizer.zero_grad(set_to_none=True)
                        msg = f'[NaNGuard] non-finite loss | mode=train epoch={epochs} batch={step_idx}'
                        logger.error(msg)
                        if self.nan_guard_raise:
                            raise RuntimeError(msg)
                        step_idx += 1
                        continue

                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()

                    bad_grad_name = self._check_grad_finite(model)
                    if bad_grad_name is not None:
                        if amp_enabled:
                            logger.warning(
                                '[AMP] skip overflow batch | epoch=%d batch=%d param=%s',
                                epochs,
                                step_idx,
                                bad_grad_name,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            left_epochs = self.update_epochs
                            step_idx += 1
                            continue
                        self._save_last_good(model, f'non-finite gradient at {bad_grad_name}')
                        optimizer.zero_grad(set_to_none=True)
                        msg = f'[NaNGuard] non-finite gradient param={bad_grad_name} mode=train epoch={epochs} batch={step_idx}'
                        logger.error(msg)
                        if self.nan_guard_raise:
                            raise RuntimeError(msg)
                        step_idx += 1
                        continue

                    if epochs == 1 and step_idx < 50 and _get(self.args, 'print_grad_norm', False):
                        total_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_norm += (p.grad.detach().norm(2).item() ** 2)
                        logger.info('[GradNorm] step=%d norm=%.4f', step_idx, total_norm ** 0.5)

                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            self.grad_clip,
                        )

                    if not left_epochs:
                        bad_param_before = self._check_param_finite(model, prefixes=['Model.fuse_mlp', 'Model.prior_head', 'Model.evidence_heads'])
                        if bad_param_before is not None:
                            self._save_last_good(model, f'non-finite param before step at {bad_param_before}')
                            optimizer.zero_grad(set_to_none=True)
                            raise RuntimeError(f'[NaNGuard] non-finite param before step: {bad_param_before}')

                        if amp_enabled:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        left_epochs = self.update_epochs

                        bad_param_after = self._check_param_finite(model, prefixes=['Model.fuse_mlp', 'Model.prior_head', 'Model.evidence_heads'])
                        if bad_param_after is not None:
                            self._save_last_good(model, f'non-finite param after step at {bad_param_after}')
                            optimizer.zero_grad(set_to_none=True)
                            raise RuntimeError(f'[NaNGuard] non-finite param after step: {bad_param_after}')

                    train_loss += loss.item()
                    train_loss_sum += loss.item()
                    good_batches += 1
                    y_pred.append(preds.detach().cpu())
                    y_true.append(labels.detach().cpu())

                    if self.loss_audit_enabled and epochs == 1 and audit_count < self.loss_audit_batches:
                        audit_running_sum += float(loss.item())
                        audit_count += 1
                        cur_mean = audit_running_sum / max(1, audit_count)
                        text_len = text.size(2) if (text.dim() == 3 and text.size(1) == 3) else text.size(1)
                        logger.info(
                            '[LossAudit] mode=train epoch=%d batch=%d batch_loss=%.8f epoch_loss_sum=%.8f '
                            'epoch_loss_mean=%.8f batch_size=%d sample_num=%d text_len=%d audio_len=%d vision_len=%d',
                            epochs,
                            step_idx,
                            float(loss.item()),
                            audit_running_sum,
                            cur_mean,
                            int(text.size(0)),
                            int(labels.size(0)),
                            int(text_len),
                            int(audio.size(1)),
                            int(vision.size(1)),
                        )

                    if epochs == 1 and _get(self.args, 'print_loss_breakdown', False):
                        loss_sums['main'] += float(loss_main.item())
                        loss_sums['l0'] += float(loss0.item())
                        loss_sums['lm'] += float(lossm.item())
                        loss_sums['le'] += float(loss_evid.item())
                        loss_sums['ls'] += float(loss_smooth.item())
                        loss_sums['lc'] += float(loss_corr.item())
                        loss_sums['lstd'] += float(loss_std.item())
                        loss_sums['cnt'] += 1

                    step_idx += 1

                if not left_epochs:
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

            if good_batches == 0:
                raise RuntimeError('[NaNGuard] no valid train batches remained after filtering')
            train_loss = train_loss / max(1, good_batches)
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            self._update_metric_calibration(pred, true, epochs)
            train_results = self.metrics(pred, true)
            train_results = self._sanitize_metrics(train_results, mode='TRAIN')
            logger.info(
                '>> Epoch:%d :TRAIN-(%s) [%d/%d/%d] >> loss: %s %s',
                epochs,
                self.args.model_name,
                epochs - best_epoch,
                epochs,
                self.args.cur_seed,
                round(train_loss, 8),
                dict_to_str(train_results),
            )
            logger.info(
                '[ZeroLen] mode=TRAIN epoch=%d seen=%d filtered=%d skipped_batches=%d good_batches=%d',
                epochs,
                int(zero_len_seen_train),
                int(zero_len_filtered_train),
                int(zero_len_skipped_batches),
                int(good_batches),
            )
            if self.is_sims:
                logger.info(
                    '[SIMS-Guard] mode=TRAIN epoch=%d nonfinite_label_skipped_batches=%d warmup_epochs=%d grad_clip=%.4f '
                    'loss_clip_preds=%s pred_clip_mode=%s label_clip=%.4f',
                    epochs,
                    int(sims_nonfinite_skipped_batches),
                    int(warmup_epochs),
                    float(self.grad_clip),
                    bool(self.sims_loss_clip_preds),
                    str(self.sims_pred_clip_mode),
                    float(self.sims_label_clip),
                )

            if epochs == 1 and _get(self.args, 'print_loss_breakdown', False):
                cnt = max(1, loss_sums['cnt'])
                main = loss_sums['main'] / cnt
                l0 = loss_sums['l0'] / cnt
                lm = loss_sums['lm'] / cnt
                le = loss_sums['le'] / cnt
                ls = loss_sums['ls'] / cnt
                lc = loss_sums['lc'] / cnt
                lstd = loss_sums['lstd'] / cnt
                total = main + l0 + lm + le + ls + lc + lstd + 1e-8
                logger.info(
                    '[Epoch1 LossAvg] main=%.4f(%.2f%%) L0=%.4f(%.2f%%) '
                    'Lm=%.4f(%.2f%%) Le=%.4f(%.2f%%) Ls=%.4f(%.2f%%) '
                    'Lc=%.4f(%.2f%%) Lstd=%.4f(%.2f%%)',
                    main, main / total * 100.0,
                    l0, l0 / total * 100.0,
                    lm, lm / total * 100.0,
                    le, le / total * 100.0,
                    ls, ls / total * 100.0,
                    lc, lc / total * 100.0,
                    lstd, lstd / total * 100.0,
                )

            val_results = self.do_test(model, dataloader['valid'], mode='VAL')
            cur_valid = val_results[self.args.KeyEval]

            if epochs == self.args.epochs:
                save_path = self.args.model_save_dir / str(self.args.dataset_name) / f'{self.args.model_name}_{self.args.dataset_name}_{epochs}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

            is_better = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if is_better:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if return_epoch_results:
                train_results['Loss'] = train_loss
                train_results['LossSum'] = float(train_loss_sum)
                train_results['LossMean'] = float(train_loss)
                train_results['TrainBatches'] = int(good_batches)
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode='TEST')
                epoch_results['test'].append(test_results)

        return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode='VAL', return_sample_results=False):
        model.eval()
        amp_enabled = bool(self.use_amp and str(self.args.device).startswith('cuda'))
        y_pred, y_true = [], []
        eval_loss = 0.0
        valid_batch_count = 0
        zero_len_seen_eval = 0
        zero_len_filtered_eval = 0
        zero_len_skipped_batches = 0
        sims_nonfinite_skipped_batches = 0

        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {'Feature_t': [], 'Feature_a': [], 'Feature_v': [], 'Feature_f': []}

        step_idx = 0
        with torch.no_grad():
            with tqdm(dataloader, disable=not self.show_progress) as td:
                for batch_data in td:
                    text, audio, vision, labels, masks, valid_lengths = self._parse_batch(batch_data)
                    meta = self._build_batch_meta(batch_data, text, audio, vision)
                    zero_len_seen_eval += len(meta.get('zero_len_idx', []))

                    text, audio, vision, labels, masks, valid_lengths, batch_data, meta, removed_zero_len = self._filter_zero_len_samples(
                        text, audio, vision, labels, masks, valid_lengths, batch_data, meta, mode
                    )
                    zero_len_filtered_eval += int(removed_zero_len)
                    if text is None:
                        zero_len_skipped_batches += 1
                        step_idx += 1
                        continue

                    if self.is_sims and self.sims_skip_nonfinite_batch and (not torch.isfinite(labels).all()):
                        logger.error(
                            '[SIMS-Guard] skip non-finite label batch | mode=%s epoch=%s batch=%d ids=%s indices=%s',
                            mode,
                            _get(self.args, 'cur_epoch', None),
                            step_idx,
                            batch_data.get('id', None),
                            batch_data.get('index', None),
                        )
                        sims_nonfinite_skipped_batches += 1
                        step_idx += 1
                        continue

                    self._check_input_tensors(mode.lower(), _get(self.args, 'cur_epoch', None), step_idx, text, audio, vision, labels, meta)

                    dbg_ctx = {
                        'mode': mode.lower(),
                        'epoch': _get(self.args, 'cur_epoch', None),
                        'batch_idx': step_idx,
                        'ids': batch_data.get('id', None),
                        'indices': batch_data.get('index', None),
                        'text_shape': tuple(text.shape),
                        'audio_shape': tuple(audio.shape),
                        'vision_shape': tuple(vision.shape),
                        'text_mask_sum': meta.get('text_mask_sum'),
                        'audio_lengths': meta.get('audio_lengths'),
                        'vision_lengths': meta.get('vision_lengths'),
                        'valid_lengths': valid_lengths,
                    }

                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        outputs = model(
                            text,
                            audio,
                            vision,
                            labels=None,
                            masks=masks,
                            valid_lengths=valid_lengths,
                            dbg_ctx=dbg_ctx,
                        )
                        self._check_output_tensors(outputs, mode.lower(), _get(self.args, 'cur_epoch', None), step_idx, meta)
                        self._log_first_batch_snapshot(mode.lower(), _get(self.args, 'cur_epoch', None), step_idx, labels, outputs)

                        preds_loss, labels_loss = self._loss_inputs(outputs['M'], labels)
                        loss = self.criterion(preds_loss, labels_loss)
                    self._log_sims_shape_audit(mode.lower(), _get(self.args, 'cur_epoch', 0), step_idx, labels_loss, outputs, loss)
                    self._log_sims_spike_once(mode.lower(), int(_get(self.args, 'cur_epoch', 0) or 0), step_idx, loss, labels_loss, preds_loss, meta)
                    if not torch.isfinite(loss):
                        self._check_finite('val_loss', loss.view(1), mode.lower(), _get(self.args, 'cur_epoch', None), step_idx, meta)
                    eval_loss += loss.item()
                    valid_batch_count += 1

                    if return_sample_results:
                        ids.extend(self._to_list(batch_data.get('id', None)) or [])
                        for item in features.keys():
                            if item in outputs:
                                features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds_np = outputs['M'].cpu().detach().numpy()
                        sample_results.extend(preds_np.squeeze())

                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
                    step_idx += 1

        if len(y_pred) == 0:
            raise RuntimeError(f'[NaNGuard] no valid batches in {mode} after filtering')

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        self._check_finite('pred_concat', pred, mode.lower(), _get(self.args, 'cur_epoch', None), -1, {'ids': None, 'indices': None})
        self._check_finite('true_concat', true, mode.lower(), _get(self.args, 'cur_epoch', None), -1, {'ids': None, 'indices': None})
        pred_metric = self._apply_metric_calibration(pred, mode=mode)
        self._check_finite('pred_metric', pred_metric, mode.lower(), _get(self.args, 'cur_epoch', None), -1, {'ids': None, 'indices': None})
        eval_results = self.metrics(pred_metric, true)
        eval_results = self._sanitize_metrics(eval_results, mode=mode)

        eval_loss = eval_loss / max(1, valid_batch_count)
        eval_results['Loss'] = round(eval_loss, 4)
        eval_results['LossSum'] = float(eval_loss * max(1, valid_batch_count))
        eval_results['EvalBatches'] = int(valid_batch_count)
        self._audit_eval(
            mode=mode,
            epoch=int(_get(self.args, 'cur_epoch', 0) or 0),
            pred=pred_metric,
            true=true,
            eval_results=eval_results,
        )
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        logger.info(
            '[ZeroLen] mode=%s epoch=%s seen=%d filtered=%d skipped_batches=%d valid_batches=%d',
            mode,
            _get(self.args, 'cur_epoch', None),
            int(zero_len_seen_eval),
            int(zero_len_filtered_eval),
            int(zero_len_skipped_batches),
            int(valid_batch_count),
        )
        if self.is_sims:
            logger.info(
                '[SIMS-Guard] mode=%s epoch=%s nonfinite_label_skipped_batches=%d loss_clip_preds=%s pred_clip_mode=%s label_clip=%.4f',
                mode,
                _get(self.args, 'cur_epoch', None),
                int(sims_nonfinite_skipped_batches),
                bool(self.sims_loss_clip_preds),
                str(self.sims_pred_clip_mode),
                float(self.sims_label_clip),
            )

        if return_sample_results:
            eval_results['Ids'] = ids
            eval_results['SResults'] = sample_results
            for k in features.keys():
                if len(features[k]) > 0:
                    features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
