from mmvsr.models.utils import flow_warp, make_layer
from mmengine.model import BaseModule


class Warper(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, feat, flow):
        return flow_warp(feat, flow)
