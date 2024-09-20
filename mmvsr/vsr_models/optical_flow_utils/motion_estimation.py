def compute_flow(x, spynet):

    b, n, c, h, w = x.size()
    lrs_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
    lrs_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

    # Calculate backward flow
    flows_backward = spynet(lrs_1, lrs_2)
    flows_backward = [flow.view(b, n - 1, 2, 
                                h // (2 ** i), 
                                w // (2 ** i)) 
                      for flow, i in zip(flows_backward, range(4))]

    # Calculate forward flow
    flows_forward = spynet(lrs_2, lrs_1)
    flows_forward = [flow.view(b, n - 1, 2, 
                               h // (2 ** i), 
                               w // (2 ** i)) 
                     for flow, i in zip(flows_forward, range(4))]

    return flows_backward, flows_forward