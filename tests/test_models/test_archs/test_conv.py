# Copyright (c) OpenMMLab. All rights reserved.
from mmvsr.models.archs import conv


def test_conv():
    assert 'Deconv' in conv.MODELS.module_dict


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
