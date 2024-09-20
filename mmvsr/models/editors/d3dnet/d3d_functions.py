import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _triple
from torch.autograd.function import once_differentiable

import os
import torch
from torch.utils.cpp_extension import load



module_path = os.path.dirname(__file__)

# Manually specify the location of source files if globbing is not needed
main_file = os.path.join(module_path, "vision.cpp")
source_cpu = [os.path.join(module_path, "cpu", "deform_cpu.cpp")]  # Add any CPU-specific source files
source_cuda = [os.path.join(module_path, "cuda", "deform_conv_cuda.cu")]

# Combine all source files
sources = [main_file] + source_cpu + source_cuda

# Define macros and extra compile arguments
extra_cflags = []
extra_cuda_cflags = [
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]

# Load the extension
D3D = load(
    name="D3D",
    sources=sources,
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True
)

class DeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias,
                stride, padding, dilation, group, deformable_groups, im2col_step):
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.kernel_size = _triple(weight.shape[2:5])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = D3D.deform_conv_forward(input, weight, bias, offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1],ctx.kernel_size[2],
                                         ctx.stride[0], ctx.stride[1],ctx.stride[2],
                                         ctx.padding[0], ctx.padding[1],ctx.padding[2],
                                         ctx.dilation[0], ctx.dilation[1],ctx.dilation[2],
                                         ctx.group,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = \
            D3D.deform_conv_backward(input, weight,
                                     bias,
                                     offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                     ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                     ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                     ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                                     ctx.group,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None
