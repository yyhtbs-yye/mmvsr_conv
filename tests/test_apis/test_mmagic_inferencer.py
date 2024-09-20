# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmvsr.apis import mmvsrInferencer
from mmvsr.utils import register_all_modules

register_all_modules()


def test_edit():
    with pytest.raises(Exception):
        mmvsrInferencer('dog', ['error_type'], None)

    with pytest.raises(Exception):
        mmvsrInferencer()

    with pytest.raises(Exception):
        mmvsrInferencer(model_setting=1)

    supported_models = mmvsrInferencer.get_inference_supported_models()
    mmvsrInferencer.inference_supported_models_cfg_inited = False
    supported_models = mmvsrInferencer.get_inference_supported_models()

    supported_tasks = mmvsrInferencer.get_inference_supported_tasks()
    mmvsrInferencer.inference_supported_models_cfg_inited = False
    supported_tasks = mmvsrInferencer.get_inference_supported_tasks()

    task_supported_models = \
        mmvsrInferencer.get_task_supported_models('Image2Image Translation')
    mmvsrInferencer.inference_supported_models_cfg_inited = False
    task_supported_models = \
        mmvsrInferencer.get_task_supported_models('Image2Image Translation')

    print(supported_models)
    print(supported_tasks)
    print(task_supported_models)

    cfg = osp.join(
        osp.dirname(__file__), '..', '..', 'configs', 'biggan',
        'biggan_2xb25-500kiters_cifar10-32x32.py')

    mmvsr_instance = mmvsrInferencer(
        'biggan',
        model_ckpt='',
        model_config=cfg,
        extra_parameters={'sample_model': 'ema'})
    mmvsr_instance.print_extra_parameters()
    inference_result = mmvsr_instance.infer(label=1)
    result_img = inference_result[1]
    assert result_img.shape == (4, 3, 32, 32)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
