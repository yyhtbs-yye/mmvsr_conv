import torch
import torch.nn.functional as F

from einops import rearrange
eps = 1e-6
def corr(f1, f2, md=4):
    b, c, h, w = f1.shape
    # Normalize features
    f1 = f1 / (torch.norm(f1, dim=1, keepdim=True) + eps)  # added epsilon to avoid division by zero
    f2 = f2 / (torch.norm(f2, dim=1, keepdim=True) + eps)

    # Compute correlation matrix
    # Unfold patches from f1 with a size of (2*md+1)x(2*md+1) and a padding of md
    f1_unfold = F.unfold(f1, kernel_size=(2*md+1, 2*md+1), padding=(md, md))
    f1_unfold = rearrange(f1_unfold, "b (c q) (h w) -> b c q h w", h=h, w=w, c=c)

    # Expand f2 for broadcasting
    f2 = f2.view(b, c, 1, h, w)

    # Sum over channels to get the correlation volume
    corr_volume = torch.sum(f1_unfold * f2, dim=1)

    return corr_volume

# Example usage
feature_map1 = torch.randn(1, 256, 128, 128)
feature_map2 = torch.randn(1, 256, 128, 128)
correlation_output = corr(feature_map1, feature_map2, md=4)
print(correlation_output.shape)  # Expected shape: (batch_size, (2*md+1)^2, H, W)
