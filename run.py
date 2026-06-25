import errno
import gc
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
import argparse

from config import get_config_regression, get_config_tune
from data_loader import MMDataLoader
from models import AMIO
from trains import ATIO
from utils import assign_gpu, count_parameters, setup_seed

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2" # This is crucial for reproducibility


SUPPORTED_MODELS = [
    'LF_DNN', 'EF_LSTM', 'TFN', 'LMF', 'MFN', 'Graph_MFN', 'MFM',
    'MulT', 'MISA', 'BERT_MAG', 'MLF_DNN', 'MTFN', 'MLMF', 'Self_MM', 'MMIM',
    'FF', 'EmotionFlow', 'SELF_EmotionFlow', 'EmotionFlow_FF'
]
SUPPORTED_DATASETS = ['MOSI', 'MOSEI', 'SIMS', 'SIMSV2']

logger = logging.getLogger('MSA')

from datetime import datetime      
now = datetime.now()
format = "%Y%m%d_%H%M%S"
format_1 = "%y%m%d_%H%M%S"
formatted_now = now.strftime(format)
formatted_now_1 = now.strftime(format_1)
epoch_num = 100
ResultName = ""


def _normalize_dataset_name(dataset_name: str) -> str:
    key = str(dataset_name).lower().strip().replace("-", "_")
    dataset_alias = {
        "sismv2": "simsv2",
        "sims_v2": "simsv2",
    }
    return dataset_alias.get(key, key)


SUMMARY_COLUMNS = [
    'epoch',
    'Val_Acc_7',
    'Val_Acc_5',
    'Val_Acc_3',
    'Val_Acc_2',
    'Val_F1_score',
    'Val_Corr',
    'Val_MAE',
    'Train_Loss',
    'Val_Loss',
]


def _safe_float(value, field_name, epoch_idx):
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    try:
        out = float(value)
    except Exception as ex:
        raise RuntimeError(f"[CSVGuard] {field_name} cannot cast to float at epoch={epoch_idx}: {value}") from ex
    if not np.isfinite(out):
        raise RuntimeError(f"[CSVGuard] {field_name} is non-finite at epoch={epoch_idx}: {out}")
    return out


def _audit_csv_file(csv_file: Path, expected_rows: int):
    df = pd.read_csv(csv_file)
    logger.info("[CSVAudit] file=%s shape=%s columns=%s", csv_file, df.shape, list(df.columns))
    logger.info("[CSVAudit] dtypes=%s", {c: str(t) for c, t in df.dtypes.items()})

    if list(df.columns) != SUMMARY_COLUMNS:
        raise RuntimeError(f"[CSVAudit] columns mismatch: got={list(df.columns)} expect={SUMMARY_COLUMNS}")
    if len(df) != int(expected_rows):
        raise RuntimeError(f"[CSVAudit] row count mismatch: got={len(df)} expect={expected_rows}")

    for col in SUMMARY_COLUMNS:
        series = pd.to_numeric(df[col], errors='coerce')
        bad_mask = series.isna() | ~np.isfinite(series.values)
        if bad_mask.any():
            bad_rows = df.index[bad_mask].tolist()
            logger.error("[CSVAudit] non-numeric rows in col=%s rows=%s", col, bad_rows)
            for idx in bad_rows:
                logger.error("[CSVAudit] row[%s]=%s", idx, df.loc[idx].to_dict())
            raise RuntimeError(f"[CSVAudit] non-numeric value in {col}: rows={bad_rows}")

    epoch_num = pd.to_numeric(df['epoch'], errors='coerce')
    if ((epoch_num % 1) != 0).any():
        bad_rows = df.index[((epoch_num % 1) != 0)].tolist()
        raise RuntimeError(f"[CSVAudit] epoch not integer rows={bad_rows}")

    logger.info("[CSVAudit] passed: summary csv numeric/integer checks all passed.")


def _audit_summary_dynamics(df: pd.DataFrame, stall_window: int = 3, raise_on_fail: bool = False):
    metric_cols = [c for c in SUMMARY_COLUMNS if c != 'epoch']
    issues = []

    for col in metric_cols:
        arr = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float64)
        if arr.size < stall_window:
            continue
        for i in range(0, arr.size - stall_window + 1):
            w = arr[i:i + stall_window]
            if np.all(np.isfinite(w)) and np.max(np.abs(w - w[0])) <= 1e-12:
                issues.append(f'constant_window col={col} epoch={i + 1}-{i + stall_window} value={w[0]:.8f}')
                break

    pair_cols = ['Val_Acc_7', 'Val_Acc_5', 'Val_Acc_3', 'Val_Acc_2', 'Val_F1_score']
    for i in range(len(pair_cols)):
        for j in range(i + 1, len(pair_cols)):
            c1, c2 = pair_cols[i], pair_cols[j]
            a = pd.to_numeric(df[c1], errors='coerce').to_numpy(dtype=np.float64)
            b = pd.to_numeric(df[c2], errors='coerce').to_numpy(dtype=np.float64)
            n = min(len(a), len(b))
            if n < stall_window:
                continue
            same = np.isfinite(a[:n]) & np.isfinite(b[:n]) & (np.abs(a[:n] - b[:n]) <= 1e-12)
            cur = 0
            best = 0
            for flag in same.tolist():
                cur = cur + 1 if flag else 0
                best = max(best, cur)
            if best >= stall_window:
                issues.append(f'same_columns col_pair={c1},{c2} longest_run={best}')

    if len(issues) == 0:
        logger.info("[MetricDynamics] passed: no constant-window or repeated-column issues.")
        return

    for msg in issues:
        logger.warning("[MetricDynamics] %s", msg)
    if raise_on_fail:
        raise RuntimeError(f"[MetricDynamics] failed with {len(issues)} issues.")


def _pick_metric(metric_dict, field_aliases, save_field, epoch_idx):
    for key in field_aliases:
        if key in metric_dict:
            return _safe_float(metric_dict[key], save_field, epoch_idx)
    raise RuntimeError(f"[CSVGuard] missing metric {save_field} at epoch={epoch_idx}, keys={list(metric_dict.keys())}")


def _build_summary_rows(epoch_results):
    rows = []
    num_epochs = len(epoch_results['train'])
    for epoch in range(num_epochs):
        train_epoch = epoch_results['train'][epoch]
        valid_epoch = epoch_results['valid'][epoch]

        if 'TrainBatches' in train_epoch and int(train_epoch['TrainBatches']) <= 0:
            raise RuntimeError(f"[CSVGuard] TrainBatches <= 0 at epoch={epoch + 1}")

        train_loss = _safe_float(train_epoch.get('LossMean', train_epoch.get('Loss')), 'Train_Loss', epoch + 1)
        row = {
            'epoch': int(epoch + 1),
            'Val_Acc_7': _pick_metric(valid_epoch, ('Acc_7', 'acc_7'), 'Val_Acc_7', epoch + 1),
            'Val_Acc_5': _pick_metric(valid_epoch, ('Acc_5', 'acc_5'), 'Val_Acc_5', epoch + 1),
            'Val_Acc_3': _pick_metric(valid_epoch, ('Acc_3', 'acc_3'), 'Val_Acc_3', epoch + 1),
            'Val_Acc_2': _pick_metric(valid_epoch, ('Acc_2', 'acc_2'), 'Val_Acc_2', epoch + 1),
            'Val_F1_score': _pick_metric(valid_epoch, ('F1_score', 'f1_score'), 'Val_F1_score', epoch + 1),
            'Val_Corr': _pick_metric(valid_epoch, ('Corr', 'corr'), 'Val_Corr', epoch + 1),
            'Val_MAE': _pick_metric(valid_epoch, ('MAE', 'mae'), 'Val_MAE', epoch + 1),
            'Train_Loss': train_loss,
            'Val_Loss': _safe_float(valid_epoch.get('Loss'), 'Val_Loss', epoch + 1),
        }
        rows.append(row)
    return rows

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}-{cur_time}.log"
    logger = logging.getLogger('MSA')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # prevent handler accumulation when MSA_run is called repeatedly in one process
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # file handler
    fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def MSA_run(
    model_name: str, 
    dataset_name: str, 
    config_file: str = None,
    config: dict = None, 
    seeds: list = [], 
    is_tune: bool = False,
    tune_times: int = 50, 
    custom_feature: str = None, 
    feature_T: str = None, 
    feature_A: str = None, 
    feature_V: str = None, 
    gpu_ids: list = [0],
    num_workers: int = 1, 
    verbose_level: int = 1,
    model_save_dir: str = Path().home() / "MSA" / "saved_models",
    res_save_dir: str = Path().home() / "MSA" / "results",
    log_dir: str = Path().home() / "MSA" / "logs",
):
    # Initialization
    model_name = model_name.lower()
    MODEL_NAME = model_name.upper()
    dataset_name = _normalize_dataset_name(dataset_name)

    if config_file is not None:
        config_file = Path(config_file)
    else:
        if is_tune:
            config_file = Path(__file__).parent / "config" / "config_tune.json"
        else:
            config_file = Path(__file__).parent / "config" / "config_regression.json"

    if not config_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

    if model_save_dir is None:
        model_save_dir = Path.home() / "MSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    if res_save_dir is None:
        res_save_dir = Path.home() / "MSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)

    if log_dir is None:
        log_dir = Path.home() / "MSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    logger.info("======================================== Program Start ========================================")

    if is_tune:
        logger.info(f"Tuning with seed {seeds[0]}")
        initial_args = get_config_tune(model_name, dataset_name, config_file)
        initial_args['model_save_path'] = Path(model_save_dir) / f"{initial_args['model_name']}-{initial_args['dataset_name']}.pth"
        initial_args['device'] = assign_gpu(gpu_ids)
        initial_args['train_mode'] = 'regression'
        initial_args['custom_feature'] = custom_feature
        initial_args['feature_T'] = feature_T
        initial_args['feature_A'] = feature_A
        initial_args['feature_V'] = feature_V

        if str(initial_args['device']).startswith('cuda'):
            torch.cuda.set_device(initial_args['device'])

        res_save_dir = Path(res_save_dir) / "tune"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        has_debuged = []
        csv_file = res_save_dir / f"{MODEL_NAME}_{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                has_debuged.append([df.loc[i, k] for k in initial_args['d_paras']])

        for i in range(tune_times):
            args = edict(**initial_args)
            random.seed(time.time())
            new_args = get_config_tune(model_name, dataset_name, config_file)
            args.update(new_args)
            if config:
                if config.get('model_name'):
                    assert (config['model_name'] == args['model_name'])
                args.update(config)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Tuning [{i + 1}/{tune_times}] {'-'*30}")
            logger.info(f"Args: {args}")
            cur_param = [args[k] for k in args['d_paras']]
            if cur_param in has_debuged:
                logger.info("This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
            setup_seed(seeds[0])
            result = _run(args, num_workers, is_tune)
            has_debuged.append(cur_param)

            if Path(csv_file).is_file():
                df2 = pd.read_csv(csv_file)
            else:
                df2 = pd.DataFrame(columns=[k for k in args.d_paras] + [k for k in result.keys()])
            res = [args[c] for c in args.d_paras]
            for col in result.keys():
                res.append(result[col])
            df2.loc[len(df2)] = res
            df2.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")
    else:
        args = get_config_regression(model_name, dataset_name, config_file)
        args['result_name'] = ResultName
        args['model_name'] = model_name
        args['model_save_dir'] = Path(model_save_dir) / f"{args['dataset_name']}"
        args['model_save_path'] = Path(model_save_dir) / f"{args['dataset_name']}" / f"{args['model_name']}{args.result_name}_{args['dataset_name']}.pth"
        Path(args['model_save_path']).parent.mkdir(parents=True, exist_ok=True)
        args['device'] = assign_gpu(gpu_ids)
        args['train_mode'] = 'regression'
        args['custom_feature'] = custom_feature
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        args['epochs'] = epoch_num

        if config:
            if config.get('model_name'):
                assert (config['model_name'] == args['model_name'])
            args.update(config)
        if int(args['epochs']) <= 1:
            logger.warning(
                "epochs=%s is very small; regression Acc_7/5/3 may collapse to identical values early.",
                args['epochs']
            )

        if str(args['device']).startswith('cuda'):
            torch.cuda.set_device(args['device'])

        logger.info(f"Model Name: {args.model_name}")
        logger.info(f"Dataset Name: {args.dataset_name}")
        logger.info(f"Task Type: {args.train_mode}")
        logger.info(f"Seeds: {seeds}")

        res_save_dir = Path(res_save_dir) / f"{model_name}"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        args['res_save_dir'] = str(res_save_dir)
        args['log_dir'] = str(log_dir)

        for i, seed in enumerate(seeds):
            setup_seed(seed)
            args['cur_seed'] = i + 1
            args['run_id'] = f"{model_name}_{dataset_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"{'-'*30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-'*30}")
            logger.info("Run ID: %s", args['run_id'])
            epoch_results = _run(args, num_workers, is_tune)

        model_csv_file = res_save_dir / f"{args['model_name']}_{args['dataset_name']}.csv"
        rows = _build_summary_rows(epoch_results)
        if len(rows) != int(args['epochs']):
            raise RuntimeError(f"[CSVGuard] summary rows != epochs: rows={len(rows)} epochs={args['epochs']}")

        df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        for col in SUMMARY_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        bad_numeric = df[SUMMARY_COLUMNS].isna().any(axis=1)
        if bad_numeric.any():
            bad_rows = df.index[bad_numeric].tolist()
            for idx in bad_rows:
                logger.error("[CSVGuard] invalid summary row[%s]=%s", idx, df.loc[idx].to_dict())
            raise RuntimeError(f"[CSVGuard] invalid summary rows: {bad_rows}")

        df['epoch'] = df['epoch'].astype(int)
        if (df['epoch'] <= 0).any():
            bad_rows = df.index[df['epoch'] <= 0].tolist()
            raise RuntimeError(f"[CSVGuard] invalid epoch<=0 rows: {bad_rows}")

        _audit_summary_dynamics(
            df,
            stall_window=int(getattr(args, 'metric_stall_window', 3)),
            raise_on_fail=bool(getattr(args, 'metric_audit_raise_on_constant_summary', False)),
        )
        df.to_csv(model_csv_file, index=False, float_format='%.6f')

        if getattr(args, 'csv_audit_enabled', True):
            _audit_csv_file(model_csv_file, expected_rows=len(rows))

        logger.info(f"Results saved to {model_csv_file}.")

def _run(args, num_workers=4, is_tune=False, from_sena=True):
    # load data and models
    loader_workers = int(getattr(args, 'num_workers', num_workers))
    logger.info(
        "DataLoader config: num_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        loader_workers,
        bool(getattr(args, 'pin_memory', False)),
        bool(getattr(args, 'persistent_workers', False)),
        getattr(args, 'prefetch_factor', None),
    )
    dataloader = MMDataLoader(args, loader_workers)
    model = AMIO(args).to(args['device'])

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # TODO: use multiple gpus
    # if using_cuda and len(args.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model,
    #                                   device_ids=args.gpu_ids,
    #                                   output_device=args.gpu_ids[0])
    trainer = ATIO().getTrain(args)
    real_model_cls = type(model.Model).__name__ if hasattr(model, 'Model') else type(model).__name__
    real_trainer_cls = type(trainer).__name__
    logger.info(f"Instantiated Model Class: {real_model_cls}")
    logger.info(f"Instantiated Trainer Class: {real_trainer_cls}")
    # do train
    # epoch_results = trainer.do_train(model, dataloader)
    epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
    # epoch_results = trainer.do_train(model, dataloader)

    
    # load trained model & do test
    assert Path(args['model_save_path']).exists()
    model.load_state_dict(torch.load(args['model_save_path']))
    model.to(args['device'])
    # ===== Step 1: 取一个测试样本 =====
    test_loader = dataloader['test']

    for batch in test_loader:
        sample_batch = batch
        break

    # 取第一个样本
    text = sample_batch['text'][0].unsqueeze(0).to(args['device'])
    audio = sample_batch['audio'][0].unsqueeze(0).to(args['device'])
    vision = sample_batch['vision'][0].unsqueeze(0).to(args['device'])
    label_obj = sample_batch['labels']
    if isinstance(label_obj, dict):
        label_obj = label_obj.get('M', next(iter(label_obj.values())))
    label = label_obj.reshape(-1)[0].item()

    print("Label:", label)
    # ===== Step 2: 当前模型 prediction =====
    model.eval()
    with torch.no_grad():
        pred = model(text, audio, vision)

        if isinstance(pred, dict):
            pred = pred['M']

        pred_value = pred.cpu().detach().numpy()[0][0]

    print(f"{args['model_name']} Prediction:", pred_value)
    if from_sena:
        final_results = {}
        # final_results['train'] = trainer.do_test(model, dataloader['train'], mode="TRAIN", return_sample_results=True)
        # final_results['valid'] = trainer.do_test(model, dataloader['valid'], mode="VALID", return_sample_results=True)
        # final_results['test'] = trainer.do_test(model, dataloader['test'], mode="TEST", return_sample_results=True)
    elif is_tune:
        # use valid set to tune hyper parameters
        # results = trainer.do_test(model, dataloader['valid'], mode="VALID")
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        # delete saved model
        Path(args['model_save_path']).unlink(missing_ok=True)
    else:
        results = trainer.do_test(model, dataloader['test'], mode="TEST")

    del model
    if str(args['device']).startswith('cuda'):
        torch.cuda.empty_cache()
    gc.collect()
    # time.sleep(1)


    # return {"epoch_results": epoch_results, 'final_results': final_results} if from_sena else results
    return epoch_results if from_sena else results



def MSA_test(
    config: dict | str,
    weights_path: str,
    feature_path: str, 
    # seeds: list = [], 
    gpu_id: int = 0, 
):
    """Test MSA models on a single sample.

    Load weights and configs of a saved model, input pre-extracted
    features of a video, then get sentiment prediction results.

    Args:
        model_name: Name of MSA model.
        config: Config dict or path to config file. 
        weights_path: Pkl file path of saved model weights.
        feature_path: Pkl file path of pre-extracted features.
        gpu_id: Specify which gpu to use. Use cpu if value < 0.
    """
    if type(config) == str or type(config) == Path:
        config = Path(config)
        with open(config, 'r') as f:
            args = json.load(f)
    elif type(config) == dict or type(config) == edict:
        args = config
    else:
        raise ValueError(f"'config' should be string or dict, not {type(config)}")
    args['train_mode'] = 'regression' # backward compatibility.

    if gpu_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')
    args['device'] = device
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)
    # args['feature_dims'] = [feature['text'].shape[1], feature['audio'].shape[1], feature['vision'].shape[1]]
    args['feature_dims'] = [feature['text'].shape[1], feature['audio'].shape[1], feature['vision'].shape[1], feature['audio_LLD'].shape[1]]
    # args['seq_lens'] = [feature['text'].shape[0], feature['audio'].shape[0], feature['vision'].shape[0]]
    args['seq_lens'] = [feature['text'].shape[0], feature['audio'].shape[0], feature['vision'].shape[0], feature['audio_LLD'].shape[0]]
    model = AMIO(args)
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        if args.get('use_bert', None):
            if type(text := feature['text_bert']) == np.ndarray:
                text = torch.from_numpy(text).float()
        else:
            if type(text := feature['text']) == np.ndarray:
                text = torch.from_numpy(text).float()
        if type(audio := feature['audio']) == np.ndarray:
            audio = torch.from_numpy(audio).float()
        if type(vision := feature['vision']) == np.ndarray:
            vision = torch.from_numpy(vision).float()
        text = text.unsqueeze(0).to(device)
        audio = audio.unsqueeze(0).to(device)
        vision = vision.unsqueeze(0).to(device)
        if args.get('need_normalized', None):
            audio = torch.mean(audio, dim=1, keepdims=True)
            vision = torch.mean(vision, dim=1, keepdims=True)
        # TODO: write a do_single_test function for each model in trains
        if args['model_name'] == 'self_mm' or args['model_name'] == 'mmim':
            output = model(text, (audio, torch.tensor(audio.shape[1]).unsqueeze(0)), (vision, torch.tensor(vision.shape[1]).unsqueeze(0)))
        elif args['model_name'] == 'tfr_net':
            input_mask = torch.tensor(feature['text_bert'][1]).unsqueeze(0).to(device)
            output, _ = model((text, text, None), (audio, audio, input_mask, None), (vision, vision, input_mask, None))
        else:
            output = model(text, audio, vision)
        if type(output) == dict:
            output = output['M']
    return output.cpu().detach().numpy()[0][0]
# mlf_dnn銆乵tfn銆乵lmf銆佸彧鏈塻ims鍜宻ims v2

def train(dataset_name, model_name, epochs=None, key_eval=None):
    runtime_cfg = {}
    if epochs is not None:
        runtime_cfg['epochs'] = int(epochs)
    if key_eval:
        runtime_cfg['KeyEval'] = str(key_eval)
    MSA_run(
        model_name=model_name,  
        dataset_name=dataset_name,
        # ['sims', 'mosi', 'mosei', 'simsv2']
        config_file='config/config_regression.json',
        config=runtime_cfg if runtime_cfg else None,
        seeds=[1111],
        is_tune=False,
        model_save_dir="./saved_models",
        res_save_dir="./results",
        log_dir="./logs",
        # gpu_ids=[0]
    )

def test(dataset_name, model_name, epochs=None, key_eval=None):
    runtime_cfg = {}
    if epochs is not None:
        runtime_cfg['epochs'] = int(epochs)
    if key_eval:
        runtime_cfg['KeyEval'] = str(key_eval)
    MSA_run(
        model_name=model_name, 
        dataset_name=dataset_name, 
        config_file='config/config_regression.json',
        config=runtime_cfg if runtime_cfg else None,
        seeds=[1111],
        is_tune=False, 
        model_save_dir="./saved_models", 
        res_save_dir="./results", 
        log_dir="./logs"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
# ...existing code...
    parser.add_argument('--model_name', type=str, default='emotionflow_ff', choices=[
        'lf_dnn', 'ef_lstm', 'tfn', 'lmf', 'mfn', 'graph_mfn', 'mfm',
        'mult', 'misa', 'bert_mag', 'mlf_dnn', 'mtfn', 'mlmf', 'self_mm', 'mmim',
        'mctn', 'cenet', 'almt', '-tetfn', 'tfr_net', 'ff',
        'emotion_flow', 'emotionflow', 'self_emotion_flow',
        'emotionflow_ff', 'emotion_flow_ff'
    ])
    parser.add_argument('--dataset_name', type=str, default='mosei', choices=['mosi', 'mosei', 'sims', 'simsv2', 'sismv2'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--key_eval', type=str, default=None, choices=['Loss', 'Corr', 'MAE'])

    # parser.add_argument('--feature_A', type=str, default='D:/DL/MSA/paper_code/dataset/MOSI/Processed/audio_LowLevelDescriptors.pkl')
    # parser.add_argument('--feature_A', type=str, default='/public/home/sixinheyi/MSA/dataset/MOSI/Processed/audio_LowLevelDescriptors.pkl')
    args = parser.parse_args()

    if args.mode == 'train':
        train(dataset_name=args.dataset_name, model_name=args.model_name, epochs=args.epochs, key_eval=args.key_eval)
    else:
        test(dataset_name=args.dataset_name, model_name=args.model_name, epochs=args.epochs, key_eval=args.key_eval)
