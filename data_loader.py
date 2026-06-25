import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2':self.__init_simsv2,
        }
        DATASET_MAP[args['dataset_name']]()

    def __infer_lengths_from_feature(self, feature):
        # feature: [N, T, D]
        if feature.ndim < 3:
            return np.array([feature.shape[1]] * feature.shape[0], dtype=np.int32)
        valid = np.any(np.abs(feature) > 1e-8, axis=-1)  # [N, T]
        return valid.sum(axis=1).astype(np.int32)

    def __resolve_lengths(self, split_data, key, feature, source_name):
        # 兼容不同预处理字段命名
        cand_keys = [key, key.replace('_lengths', '_lens'), key.replace('_lengths', '_length')]
        for k in cand_keys:
            if k in split_data:
                return np.asarray(split_data[k], dtype=np.int32)
        inferred = self.__infer_lengths_from_feature(feature)
        logger.warning(
            f"{self.mode}: missing '{key}' in {source_name}, fallback to inferred lengths from feature "
            f"(min={int(inferred.min())}, max={int(inferred.max())})"
        )
        return inferred

    def __text_lengths(self):
        if self.args.get('use_bert', None) and self.text.ndim == 3 and self.text.shape[1] >= 2:
            return np.sum(self.text[:, 1, :] > 0, axis=1).astype(np.int32)
        valid = np.any(np.abs(self.text) > 1e-8, axis=-1)
        return valid.sum(axis=1).astype(np.int32)

    def __log_zero_len_stats(self):
        if len(self.ids) == 0:
            return
        text_lengths = self.__text_lengths()
        audio_lengths = self.audio_lengths if hasattr(self, 'audio_lengths') else self.__infer_lengths_from_feature(self.audio)
        vision_lengths = self.vision_lengths if hasattr(self, 'vision_lengths') else self.__infer_lengths_from_feature(self.vision)

        def _log_one(name, lengths):
            arr = np.asarray(lengths).reshape(-1)
            zero_idx = np.where(arr <= 0)[0].tolist()
            if len(zero_idx) == 0:
                logger.info(
                    "%s: %s length ok (min=%d max=%d)",
                    self.mode,
                    name,
                    int(arr.min()) if arr.size > 0 else -1,
                    int(arr.max()) if arr.size > 0 else -1,
                )
                return
            preview_idx = zero_idx[:10]
            preview_ids = [self.ids[i] for i in preview_idx]
            logger.warning(
                "%s: %s zero_len=%d/%d preview_idx=%s preview_ids=%s",
                self.mode,
                name,
                len(zero_idx),
                len(arr),
                preview_idx,
                preview_ids,
            )

        _log_one('text', text_lengths)
        _log_one('audio', audio_lengths)
        _log_one('vision', vision_lengths)

    def __init_mosi(self):
        if self.args['custom_feature']:
            # use custom feature file extracted with MSA-FET
            with open(self.args['custom_feature'], 'rb') as f:
                data = pickle.load(f)
        else:
            # use deault feature file specified in config file
            with open(self.args['featurePath'], 'rb') as f:
                data = pickle.load(f)
        
        if self.args.get('use_bert', None):
            self.text = data[self.mode]['text_bert'].astype(np.float32)
            self.args['feature_dims'][0] = 768
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
            self.args['feature_dims'][0] = self.text.shape[2]
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.args['feature_dims'][2] = self.vision.shape[2]
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        # Overide with custom modality features
        if self.args['feature_T']:
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if self.args.get('use_bert', None):
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A']:
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V']:
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        self.labels = {
            # 'M': data[self.mode][self.args['train_mode']+'_labels'].astype(np.float32)
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        if 'sims' in self.args['dataset_name']:
            for m in "TAV":
                self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args['need_data_aligned']:
            if self.args['feature_A']:
                self.audio_lengths = self.__resolve_lengths(
                    data_A[self.mode], 'audio_lengths', self.audio, 'feature_A'
                )
            else:
                self.audio_lengths = self.__resolve_lengths(
                    data[self.mode], 'audio_lengths', self.audio, 'featurePath'
                )
            if self.args['feature_V']:
                self.vision_lengths = self.__resolve_lengths(
                    data_V[self.mode], 'vision_lengths', self.vision, 'feature_V'
                )
            else:
                self.vision_lengths = self.__resolve_lengths(
                    data[self.mode], 'vision_lengths', self.vision, 'featurePath'
                )
        self.audio[self.audio == -np.inf] = 0

        if self.args.get('data_missing'):
            # Currently only support unaligned data missing.
            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                        self.args.missing_rate[0], self.args.missing_seed[0], mode='text')
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:,2,:], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

            if self.args['need_data_aligned']:
                self.audio_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)
                self.vision_lengths = np.sum(self.text[:,1,:], axis=1, dtype=np.int32)

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                        self.args.missing_rate[1], self.args.missing_seed[1], mode='audio')
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                        self.args.missing_rate[2], self.args.missing_seed[2], mode='vision')

        if self.args.get('need_normalized'):
            self.__normalize()

        if self.args.get('log_zero_len_dataset', True):
            self.__log_zero_len_stats()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()
    
    def __init_simsv2(self):
        return self.__init_mosi()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        
        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        np.random.seed(missing_seed)
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
        
        assert missing_mask.shape == input_mask.shape
        
        if mode == 'text':
            # CLS SEG Token unchanged.
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask) # UNK token: 100.
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
        
        return modality_m, input_len, input_mask, missing_mask

    def __truncate(self):
        # NOTE: truncate input to specific length.
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        if self.args.get('data_missing'):
            sample['text_m'] = torch.Tensor(self.text_m[index])
            sample['text_missing_mask'] = torch.Tensor(self.text_missing_mask[index])
            sample['audio_m'] = torch.Tensor(self.audio_m[index])
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['audio_mask'] = self.audio_mask[index]
            sample['audio_missing_mask'] = torch.Tensor(self.audio_missing_mask[index])
            sample['vision_m'] = torch.Tensor(self.vision_m[index])
            sample['vision_lengths'] = self.vision_lengths[index]
            sample['vision_mask'] = self.vision_mask[index]
            sample['vision_missing_mask'] = torch.Tensor(self.vision_missing_mask[index])

        return sample

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    pin_memory = bool(args.get('pin_memory', False))
    persistent_workers = bool(args.get('persistent_workers', False)) and int(num_workers) > 0
    prefetch_factor = args.get('prefetch_factor', None)

    dataLoader = {}
    for ds in datasets.keys():
        loader_kwargs = {
            'batch_size': args['batch_size'],
            'num_workers': num_workers,
            'shuffle': (ds == 'train'),
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
        }
        if int(num_workers) > 0 and prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = int(prefetch_factor)
        dataLoader[ds] = DataLoader(datasets[ds], **loader_kwargs)
    
    return dataLoader
