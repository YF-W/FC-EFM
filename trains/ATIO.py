"""
ATIO -- All Trains in One
"""
from .multiTask import *
from .singleTask import *
from .missingTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'ff': FF,
            'emotion_flow': EmotionFlowTrain,
            'emotionflow': EmotionFlowTrain,
            'self_emotion_flow': EmotionFlowTrain,
            'emotionflow_ff': EmotionFlowFFTrain,
            'emotion_flow_ff': EmotionFlowFFTrain,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
