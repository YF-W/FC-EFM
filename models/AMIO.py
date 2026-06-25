"""
AMIO -- All Model in One
"""
import torch.nn as nn

from .singleTask import *
from .subNets import AlignSubNet
# from transformers import BertConfig

from transformers import BertModel, BertConfig

# config = BertConfig.from_pretrained(r'C:\Users\sixinheyi\.cache\huggingface\transformers\bert-base-uncased-local')
# model = BertModel.from_pretrained(r'C:\Users\sixinheyi\.cache\huggingface\transformers\bert-base-uncased-local', config=config)

# tokenizer = BertTokenizer.from_pretrained(r"/public/home/sixinheyi/.cache/huggingface/transformers/bert-base-uncased-local")
# model = BertModel.from_pretrained(r"/public/home/sixinheyi/.cache/huggingface/transformers/bert-base-uncased-local")

import os
from transformers import BertTokenizer

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=True):
        super().__init__()

        # 设定路径
        if os.name == 'posix':
            bert_path = '/public/home/sixinheyi/MSA/MSA_Project/Project_misa/bert_models/bert-base-chinese'
        elif os.name == 'nt':
            bert_path = 'D:\code\Project_misa\need_model'
        else:
            raise EnvironmentError("无法识别系统")

        # ✅ 正确使用 self：在类的方法内使用
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
        self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)





class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            'ff': FF,
            'self_emotion_flow': SELF_EmotionFlow,
            'emotionflow_ff': EmotionFlowFF,
            'emotion_flow_ff': EmotionFlowFF,
            "emotion_flow": EmotionFlow,
        }
        self.need_model_aligned = args.get('need_model_aligned', None)
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_model_aligned):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()
        lastModel = self.MODEL_MAP[args['model_name']]

        if args.model_name == 'cenet':
            config = BertConfig.from_pretrained(args.pretrained, num_labels=1, finetuning_task='sst')
            self.Model = CENET.from_pretrained(args.pretrained, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True, args=args)
        else:
            self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)
