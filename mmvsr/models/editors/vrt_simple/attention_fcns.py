import torch.nn.functional as F

def swin_attention(q, k, v, mask, scale, rpe_table=None, rpe_index=None):
    """
    # w_B: num_windows * batch_size 
    # N: number of patches/pixels in a window 
    # C: number of embeddings per patch/pixel
    # nW: num_windows

    Parameters:
    q : Tensor
        The query tensor of shape (w_B, N, C).          
    k : Tensor
        The key tensor of shape (w_B, N, C).
    v : Tensor
        The value tensor of shape (w_B, N, C).
    mask : Tensor or None
        The mask tensor of shape (nW, N, N) or None.    
    rpe_table : None or Tensor
        A tensor containing the relative position biases.
    rpe_index : Tensor
        Tensor of relative position indices.
    scale : float
        Scaling factor (often 1 / sqrt(d_k)).

    Returns:
    Tensor
        The output after applying the attention mechanism.
    """

    (w_B, nH, N, CdnH) = q.shape                            # CdnH: C/nH

    C = CdnH*nH

    # Compute scaled dot-product attention, k.transpose(-2, -1) -> shape=(w_B, nH, C/nH, N)
    # (w_B, nH, N, C/nH) @ (w_B, nH, C/nH, N) = (w_B, nH, N, N), it cancellates C 
    beta = (q * scale) @ k.transpose(-2, -1)

    if rpe_table is not None:
        rpe = rpe_table[rpe_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
        beta = beta + rpe.permute(2, 0, 1).unsqueeze(0)  # w_B, nH, N, N

    # The mask is needed for swin transformer s-msa
    if mask is None:
        alpha = F.softmax(beta, dim=-1)
    else:
        nW = mask.shape[0]
        alpha = beta.view(w_B // nW, nW, nH, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
        alpha = alpha.view(-1, nH, N, N)
        alpha = F.softmax(alpha, dim=-1)

    # alpha.shape=(w_B, nH, N, N), v.shape=(w_B, nH, N, C)
    # (alpha @ v).transpose(1, 2).shape=(w_B, N, nH, C)
    # $final.shape=(w_B, N, nH*C)
    x = (alpha @ v).transpose(1, 2).reshape(w_B, N, C)

    return x