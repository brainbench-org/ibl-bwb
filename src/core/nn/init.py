import torch
from torch import nn


def tfixup_init_(encoder: nn.TransformerEncoder, scale_base: float, v_scale_factor: float) -> None:
    r"""Scale a Transformer encoder's weights in place for optimization without warmup.

    Sources:
    Transformers without Tears https://arxiv.org/pdf/1910.05895.pdf
    T-Fixup http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf

    Args:
        encoder: encoder whose ``layers`` use the stock ``nn.TransformerEncoderLayer``
            parameter names (``linear1``/``linear2``/``self_attn``).
        scale_base: T-Fixup base scale, applied as ``scale_base * N**-0.25``.
        v_scale_factor: extra factor on the value weights, on top of ``scale``.
    """
    with torch.no_grad():
        num_layers = len(encoder.layers)

        # scale = 0.67 * N^(-1/4), v_scale = 0.67 * N^(-1/4) * sqrt(2)
        scale = scale_base * (num_layers**-0.25)
        v_scale = scale * v_scale_factor

        for name, param in encoder.named_parameters():
            # Scale linear layers and the attention output projection
            if any(
                x in name
                for x in [
                    "linear1.weight",
                    "linear2.weight",
                    "self_attn.out_proj.weight",
                ]
            ):
                param.mul_(scale)

            # Scale ONLY the Value weights inside the bundled attention input projection
            elif "self_attn.in_proj_weight" in name:
                H = param.size(1)
                # param shape is (3 * hidden_dim, hidden_dim).
                # Q is param[0 : hidden_dim], K is param[hidden_dim : 2 * hidden_dim], V is param[2 * hidden_dim :]
                param[2 * H :].mul_(v_scale)
