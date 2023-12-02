from functools import partial

import torch

from ...util import default, instantiate_from_config


class VanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None):
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {
                    "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
                },
            )
        )

    def __call__(self, x, sigma):
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma)
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "add_crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out
    

class DualCFG:

    def __init__(self, scale):
        self.scale = scale
        self.dyn_thresh = instantiate_from_config(
            {
                "target": "sgm.modules.diffusionmodules.sampling_utils.DualThresholding"
            },
        )

    def __call__(self, x, sigma):
        x_u_1, x_u_2, x_c = x.chunk(3)
        x_pred = self.dyn_thresh(x_u_1, x_u_2, x_c, self.scale)
        return x_pred

    def prepare_inputs(self, x, s, c, uc_1, uc_2):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat", "add_crossattn"]:
                c_out[k] = torch.cat((uc_1[k], uc_2[k], c[k]), 0)
            else:
                assert c[k] == uc_1[k]
                c_out[k] = c[k]
        return torch.cat([x] * 3), torch.cat([s] * 3), c_out



class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out
