# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmvsr.models import BaseEditModel
from mmvsr.registry import MODELS

@MODELS.register_module()
class NaiveVSR(BaseEditModel):
    """Naive model for video super-resolution. There is no particular 
    configuration about training. 

    The purpose is to test existing VSR algorithms like VRT and RVRT

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):

        super().__init__(
            generator=generator,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
