# Copyright (c) OpenMMLab. All rights reserved.
from .inferencers.inference_functions import init_model
from .mmvsr_inferencer import mmvsrInferencer

__all__ = ['mmvsrInferencer', 'init_model']
