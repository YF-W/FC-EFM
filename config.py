import json
import os
from pathlib import Path
import random
from easydict import EasyDict as edict
from typing import Union



def _normalize_dataset_name(dataset_name: str) -> str:
    key = str(dataset_name).lower().strip().replace("-", "_")
    dataset_alias = {
        "sismv2": "simsv2",
        "sims_v2": "simsv2",
    }
    return dataset_alias.get(key, key)


def get_config_regression(
    
    model_name: str, dataset_name: str, config_file: Union[str, Path] = ""
) -> dict:
    key_std = model_name.lower().replace("-", "_").strip()
    dataset_name = _normalize_dataset_name(dataset_name)
    alias_map = {
        "emotionflow": "emotion_flow",
        "emotion-flow": "emotion_flow",
        "ef": "emotion_flow",
        "self_emotionflow": "emotion_flow",
        "self_emotion_flow": "emotion_flow",
        "selfflow": "emotion_flow",
        "emotionflow_ff": "emotionflow_ff",
        "emotion_flow_ff": "emotionflow_ff",
        "emotion-flow-ff": "emotionflow_ff",
    }
    config_key = alias_map.get(key_std, key_std)
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        model_name: Name of model.
        dataset_name: Name of dataset.
        config_file: Path to config file, if given an empty string, will use default config file.

    Returns:
        config (dict): config of the given dataset and model
    """
    # 处理配置文件路径
    if not config_file:
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    
    # 验证配置文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    # 加载配置文件
    try:
        with open(config_file, 'r') as f:
            config_all = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"配置文件格式错误: {config_file}")
    
    # 验证模型名称是否存在
    if config_key not in config_all:
        raise KeyError(f"配置文件中未找到模型: {config_key}")
    
    model_config = config_all[config_key]
    
    # 验证并获取公共参数
    if 'commonParams' not in model_config:
        raise KeyError(f"模型 {model_name} 缺少 commonParams 配置")
    model_common_args = model_config['commonParams']
    
    # 验证并获取数据集参数
    if 'datasetParams' not in model_config:
        raise KeyError(f"模型 {model_name} 缺少 datasetParams 配置")
    if dataset_name not in model_config['datasetParams']:
        raise KeyError(f"模型 {model_name} 的 datasetParams 中未找到数据集: {dataset_name}")
    model_dataset_args = model_config['datasetParams'][dataset_name]
    
    # 验证并获取通用数据集参数
    if 'datasetCommonParams' not in config_all:
        raise KeyError("配置文件中缺少 datasetCommonParams 配置")
    if dataset_name not in config_all['datasetCommonParams']:
        raise KeyError(f"datasetCommonParams 中未找到数据集: {dataset_name}")
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    
    # 处理对齐/未对齐特征
    need_aligned = model_common_args.get('need_data_aligned', False)
    if need_aligned:
        if 'aligned' in dataset_args:
            dataset_args = dataset_args['aligned']
        else:
            raise KeyError(f"数据集 {dataset_name} 需要对齐特征，但配置中未找到 'aligned' 选项")
    else:
        if 'unaligned' in dataset_args:
            dataset_args = dataset_args['unaligned']
        else:
            raise KeyError(f"数据集 {dataset_name} 需要未对齐特征，但配置中未找到 'unaligned' 选项")
    
    # 合并配置参数
    config = {}
    # 保留用户使用的模型名（含别名），便于后续模型/训练器选择
    config['model_name'] = key_std
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    
    # 处理特征路径
    if 'datasetCommonParams' in config_all and 'dataset_root_dir' in config_all['datasetCommonParams']:
        if 'featurePath' in config:
            config['featurePath'] = os.path.join(
                config_all['datasetCommonParams']['dataset_root_dir'], 
                config['featurePath']
            )
        else:
            print(f"警告: 配置中缺少 'featurePath' 参数")
    else:
        print(f"警告: 配置文件中缺少 'datasetCommonParams.dataset_root_dir' 参数")
    
    # 使用edict便于访问
    config = edict(config)
    # ==== EmotionFlow defaults (return 前一行粘这一段) ====
    def _set_default(d, k, v):
        if k not in d:
            d[k] = v

    _set_default(config, 'ef_hidden', 128)
    _set_default(config, 'ef_layers', 2)
    _set_default(config, 'ef_heads', 1)
    _set_default(config, 'ef_dropout', 0.10)

    _set_default(config, 'score_hidden', 128)
    _set_default(config, 'tau', 1.0)
    _set_default(config, 'smooth_lambda', 0.20)
    _set_default(config, 'ema_alpha', 0.90)
    _set_default(config, 'ema_beta', 0.90)
    _set_default(config, 'evidence_k', 5)
    _set_default(config, 'evidence_k_eval', 5)
    _set_default(config, 'evidence_eta', 0.10)
    _set_default(config, 'evidence_temp', 1.0)
    _set_default(config, 'omega_kappa', 5.0)
    _set_default(config, 'dirichlet_c', 1.0)
    _set_default(config, 'lambda0', 0.2)
    _set_default(config, 'lambda1', 0.2)
    _set_default(config, 'mc_dropout_eval', False)
    _set_default(config, 'lambda_e', 0.1)
    _set_default(config, 'evidence_use_net', True)
    _set_default(config, 'evidence_hidden', 64)
    _set_default(config, 'w_smooth_lambda', 0.0)

    _set_default(config, 'tv_lambda', 0.05)
    _set_default(config, 'entropy_lambda', 0.01)
    _set_default(config, 'pooling', 'mean')
    _set_default(config, 'loss_name', 'mse')
    _set_default(config, 'smooth_l1_beta', 1.0)
    _set_default(config, 'label_clip', 0.0)
    _set_default(config, 'loss_warmup_epochs', 0)
    _set_default(config, 'print_epoch1_stats', False)
    _set_default(config, 'print_loss_breakdown', False)
    _set_default(config, 'print_grad_norm', False)
    _set_default(config, 'loss_audit_enabled', False)
    _set_default(config, 'loss_audit_batches', 3)
    _set_default(config, 'dump_first_batch_stats', False)
    _set_default(config, 'csv_audit_enabled', True)
    _set_default(config, 'nan_guard', True)
    _set_default(config, 'nan_guard_raise', True)
    _set_default(config, 'skip_zero_len_eval', True)
    _set_default(config, 'skip_zero_len_train', False)
    _set_default(config, 'save_last_good_ckpt', False)
    _set_default(config, 'last_good_ckpt_name', 'last_good.pth')
    _set_default(config, 'check_param_finite', True)
    _set_default(config, 'return_debug_tensors', True)
    _set_default(config, 'autograd_detect_anomaly', False)
    _set_default(config, 'warmup_ratio', 0.1)
    _set_default(config, 'bert_lr_ratio', 0.05)
    _set_default(config, 'show_progress', True)
    _set_default(config, 'show_progress', True)
    _set_default(config, 'smooth_l1_beta', 1.0)
    _set_default(config, 'label_clip', 0.0)
    _set_default(config, 'warmup_ratio', 0.1)
    _set_default(config, 'bert_lr_ratio', 0.05)
    # ==== EmotionFlow defaults END ====


    return config


def get_config_tune(
    model_name: str, dataset_name: str, config_file: str = "",
    random_choice: bool = True
) -> dict:
    """
    Get the tuning config of given dataset and model from config file.

    Parameters:
        model_name: Name of model.
        dataset_name: Name of dataset.
        config_file: Path to config file, if given an empty string, will use default config file.
        random_choice: If True, will randomly choose a config from the list of configs.

    Returns:
        config (dict): config of the given dataset and model
    """
    dataset_name = _normalize_dataset_name(dataset_name)

    # 处理配置文件路径
    if not config_file:
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    
    # 验证配置文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    # 加载配置文件
    try:
        with open(config_file, 'r') as f:
            config_all = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"配置文件格式错误: {config_file}")
    
    # 验证模型名称是否存在
    if model_name not in config_all:
        raise KeyError(f"配置文件中未找到模型: {model_name}")
    
    model_config = config_all[model_name]
    
    # 验证并获取公共参数
    if 'commonParams' not in model_config:
        raise KeyError(f"模型 {model_name} 缺少 commonParams 配置")
    model_common_args = model_config['commonParams']
    
    # 获取数据集参数（可选）
    model_dataset_args = model_config.get('datasetParams', {}).get(dataset_name, {})
    
    # 验证并获取调试参数
    if 'debugParams' not in model_config:
        raise KeyError(f"模型 {model_name} 缺少 debugParams 配置")
    model_debug_args = model_config['debugParams']
    
    # 验证并获取通用数据集参数
    if 'datasetCommonParams' not in config_all:
        raise KeyError("配置文件中缺少 datasetCommonParams 配置")
    if dataset_name not in config_all['datasetCommonParams']:
        raise KeyError(f"datasetCommonParams 中未找到数据集: {dataset_name}")
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    
    # 处理对齐/未对齐特征
    need_aligned = model_common_args.get('need_data_aligned', False)
    if need_aligned:
        if 'aligned' in dataset_args:
            dataset_args = dataset_args['aligned']
        else:
            raise KeyError(f"数据集 {dataset_name} 需要对齐特征，但配置中未找到 'aligned' 选项")
    else:
        if 'unaligned' in dataset_args:
            dataset_args = dataset_args['unaligned']
        else:
            raise KeyError(f"数据集 {dataset_name} 需要未对齐特征，但配置中未找到 'unaligned' 选项")
    
    # 随机选择调试参数
    if random_choice:
        for item in model_debug_args.get('d_paras', []):
            if item in model_debug_args:
                if isinstance(model_debug_args[item], list):
                    model_debug_args[item] = random.choice(model_debug_args[item])
                elif isinstance(model_debug_args[item], dict):
                    # 嵌套参数，最多处理2层
                    for k, v in model_debug_args[item].items():
                        if isinstance(v, list):
                            model_debug_args[item][k] = random.choice(v)
    
    # 合并配置参数
    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(model_debug_args)
    
    # 处理特征路径
    if 'datasetCommonParams' in config_all and 'dataset_root_dir' in config_all['datasetCommonParams']:
        if 'featurePath' in config:
            config['featurePath'] = os.path.join(
                config_all['datasetCommonParams']['dataset_root_dir'], 
                config['featurePath']
            )
        else:
            print(f"警告: 配置中缺少 'featurePath' 参数")
    else:
        print(f"警告: 配置文件中缺少 'datasetCommonParams.dataset_root_dir' 参数")
    
    # 使用edict便于访问
    config = edict(config)
    def _set_default(d, k, v):
        if k not in d:
            d[k] = v

    _set_default(config, 'ef_hidden', 128)
    _set_default(config, 'ef_layers', 2)
    _set_default(config, 'ef_heads', 1)
    _set_default(config, 'ef_dropout', 0.10)

    _set_default(config, 'score_hidden', 128)
    _set_default(config, 'tau', 1.0)
    _set_default(config, 'smooth_lambda', 0.20)
    _set_default(config, 'ema_alpha', 0.90)
    _set_default(config, 'ema_beta', 0.90)
    _set_default(config, 'evidence_k', 5)
    _set_default(config, 'evidence_k_eval', 5)
    _set_default(config, 'evidence_eta', 0.10)
    _set_default(config, 'evidence_temp', 1.0)
    _set_default(config, 'omega_kappa', 5.0)
    _set_default(config, 'dirichlet_c', 1.0)
    _set_default(config, 'lambda0', 0.2)
    _set_default(config, 'lambda1', 0.2)
    _set_default(config, 'mc_dropout_eval', False)
    _set_default(config, 'lambda_e', 0.1)
    _set_default(config, 'evidence_use_net', True)
    _set_default(config, 'evidence_hidden', 64)
    _set_default(config, 'w_smooth_lambda', 0.0)

    _set_default(config, 'tv_lambda', 0.05)
    _set_default(config, 'entropy_lambda', 0.01)
    _set_default(config, 'pooling', 'mean')
    _set_default(config, 'loss_name', 'mse')
    _set_default(config, 'nan_guard', True)
    _set_default(config, 'nan_guard_raise', True)
    _set_default(config, 'skip_zero_len_eval', True)
    _set_default(config, 'skip_zero_len_train', False)
    _set_default(config, 'save_last_good_ckpt', False)
    _set_default(config, 'last_good_ckpt_name', 'last_good.pth')
    _set_default(config, 'check_param_finite', True)
    _set_default(config, 'return_debug_tensors', True)
    _set_default(config, 'autograd_detect_anomaly', False)
    _set_default(config, 'loss_audit_enabled', False)
    _set_default(config, 'loss_audit_batches', 3)
    _set_default(config, 'dump_first_batch_stats', False)
    _set_default(config, 'csv_audit_enabled', True)
    # ==== EmotionFlow defaults END ====

    return config


def get_config_all(config_file: str) -> dict:
    """
    Get all default configs. This function is used to export default config file. 
    If you want to get config for a specific model, use "get_config_regression" or "get_config_tune" instead.

    Parameters:
        config_file: "regression" or "tune"
    
    Returns:
        config: all default configs
    """
    # 处理配置文件路径
    if config_file == "regression":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    elif config_file == "tune":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    else:
        raise ValueError("config_file should be 'regression' or 'tune'")
    
    # 验证配置文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    # 加载配置文件
    try:
        with open(config_file, 'r') as f:
            config_all = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"配置文件格式错误: {config_file}")
    
    # 使用edict便于访问
    return edict(config_all)


def get_citations() -> dict:
    """
    Get paper titles and citations for models and datasets.

    Returns:
        cites (dict): {
            models: {
                tfn: {
                    title: "xxx",
                    paper_url: "xxx",
                    citation: "xxx",
                    description: "xxx"
                },
                ...
            },
            datasets: {
                ...
            },
        }
    """
    config_file = Path(__file__).parent / "config" / "citations.json"
    
    # 验证配置文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"引用配置文件不存在: {config_file}")
    
    # 加载配置文件
    try:
        with open(config_file, 'r') as f:
            cites = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"引用配置文件格式错误: {config_file}")
    
    return cites
