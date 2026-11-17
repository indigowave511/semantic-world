"""Q/K position reinjection for the 1-D AR baseline, outside the original model.

Layer 0 is the parent forward. From ``reinject_from_layer`` (default 1), Q and K
read ``LN(H) + c PE(i)``. V and the residual read ``LN(H)`` and ``H``. ``PE(i)``
is the same 1-D sinusoid ``_encode_layers`` adds at the input, including PAD
columns of the packed batch. No new parameters.

``c`` defaults to the scale whose energy ``c² · d/2`` equals the mean Frobenius
energy of the 4-tensor ``P + Temb`` on this ``N × T`` grid. Pass ``position_scale``
to override it. ``baseline_ar_attention_model.py`` is patched only during
``__init__``, then restored.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

import baseline_ar_attention_model as bm
from baseline_ar_attention_model import BaselineARAttention


def mean_four_tensor_pos_energy(N: int, T: int, d_x: int, d_t: int) -> float:
    """Mean over (x, τ) of ||P_x 1^T + 1 T_{x,τ}^T||_F^2, including the cross term."""
    from four_tensor_model import _sinusoidal_pe

    x = torch.arange(N, dtype=torch.float64)
    tau = torch.arange(T, dtype=torch.float64)
    P = _sinusoidal_pe(x, d_x)
    Temb = _sinusoidal_pe(x[:, None] + tau[None, :], d_t)
    diagonal = d_t * (P * P).sum(-1)[:, None] + d_x * (Temb * Temb).sum(-1)
    cross = 2 * P.sum(-1)[:, None] * Temb.sum(-1)
    return float((diagonal + cross).mean())


def mean_energy_position_scale(N: int, T: int, d_x: int, d_t: int) -> float:
    """c such that ||c PE||^2 matches the mean 4-tensor Pos energy. ||PE||^2 = d/2."""
    d = d_x * d_t
    energy = mean_four_tensor_pos_energy(N, T, d_x, d_t)
    return (energy / (d / 2)) ** 0.5


class QKReinjectionCausalAttention(bm.CausalSelfAttention):
    def forward(self, H, ids, padding_idx, H_qk=None):
        if H_qk is None:
            return super().forward(H, ids, padding_idx)
        if H_qk.shape != H.shape:
            raise ValueError(
                f"H_qk shape {tuple(H_qk.shape)} does not match H {tuple(H.shape)}"
            )
        B, L, d = H.shape
        blocked = bm.causal_pad_mask(ids, padding_idx)
        neg = torch.finfo(H.dtype).min
        Q = self.W_Q(H_qk)
        K = self.W_K(H_qk)
        V = self.W_V(H)
        if self.n_heads == 1:
            scores = (Q @ K.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(blocked.squeeze(1), neg)
            A = torch.softmax(scores, dim=-1)
            return A @ V
        Q = Q.view(B, L, self.n_heads, self.d_h).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.d_h).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_h).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(blocked, neg)
        A = torch.softmax(scores, dim=-1)
        out = (A @ V).transpose(1, 2).contiguous().view(B, L, d)
        return self.W_O(out)


class QKReinjectionDecoderLayer(bm.CausalDecoderLayer):
    def forward(self, H, ids, pe=None):
        U = self.ln_attn(H)
        H_qk = None if pe is None else U + pe
        attn = self.attn(U, ids, self.padding_idx, H_qk)
        H = H + self.dropout(attn)
        Z = self.fc2(F.gelu(self.fc1(self.ln_mlp(H))))
        return H + self.dropout(Z)


@contextmanager
def _reinjection_classes():
    saved = (bm.CausalSelfAttention, bm.CausalDecoderLayer)
    bm.CausalSelfAttention = QKReinjectionCausalAttention
    bm.CausalDecoderLayer = QKReinjectionDecoderLayer
    try:
        yield
    finally:
        bm.CausalSelfAttention, bm.CausalDecoderLayer = saved


class BaselineARAttentionQKReinjection(BaselineARAttention):
    """BaselineARAttention with Q/K position reinjection from ``reinject_from_layer``."""

    reinject_from_layer = 1
    position_scale = None

    def __init__(self, *args, reinject_from_layer: int | None = None, position_scale: float | None = None, **kwargs):
        if reinject_from_layer is None:
            reinject_from_layer = self.reinject_from_layer
        if position_scale is None:
            position_scale = self.position_scale
        if reinject_from_layer < 0:
            raise ValueError(
                f"reinject_from_layer must be >= 0, got {reinject_from_layer}"
            )
        with _reinjection_classes():
            super().__init__(*args, **kwargs)
        if position_scale is None:
            position_scale = mean_energy_position_scale(self.N, self.T, self.d_x, self.d_t)
        self.reinject_from_layer = reinject_from_layer
        self.position_scale = float(position_scale)
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, QKReinjectionDecoderLayer):
                raise TypeError(
                    f"layer {index} is {type(layer).__name__}, expected QKReinjectionDecoderLayer"
                )
            if not isinstance(layer.attn, QKReinjectionCausalAttention):
                raise TypeError(
                    f"layer {index} attention is {type(layer.attn).__name__}"
                )
            layer.layer_index = index

    def _encode_layers(self, ids: torch.Tensor) -> torch.Tensor:
        _, L = ids.shape
        pos = torch.arange(L, device=ids.device, dtype=self.E.weight.dtype)
        pe = bm._sinusoidal_pe(pos, self.d)
        H = self.E(ids) + pe
        pe_qk = None
        if self.reinject_from_layer < self.n_layers:
            pe_qk = pe * self.position_scale
        for layer in self.layers:
            use = pe_qk if layer.layer_index >= self.reinject_from_layer else None
            H = layer(H, ids, use)
        return self.final_norm(H)


def _self_check() -> None:
    scale = mean_energy_position_scale(16, 4, 64, 32)
    if not 1.66 < scale < 1.68:
        raise SystemExit(f"paper-grid position scale {scale} is outside 1.66–1.68")

    kwargs = dict(
        vocab_size=11, N=4, T=2, d_x=4, d_t=4,
        n_layers=2, n_heads=2, dropout=0.0, padding_idx=0,
    )
    torch.manual_seed(0)
    plain = BaselineARAttention(**kwargs)
    torch.manual_seed(0)
    rein = BaselineARAttentionQKReinjection(**kwargs)
    plain_sd = plain.state_dict()
    rein_sd = rein.state_dict()
    if plain_sd.keys() != rein_sd.keys():
        raise SystemExit("state_dict keys differ from BaselineARAttention")
    for key in plain_sd:
        if not torch.equal(plain_sd[key], rein_sd[key]):
            raise SystemExit(f"init mismatch at {key}")

    ids = torch.randint(1, 11, (2, 6))
    plain.eval()
    rein.eval()
    with torch.no_grad():
        rein.reinject_from_layer = rein.n_layers
        if not torch.allclose(plain._encode_layers(ids), rein._encode_layers(ids), atol=1e-6, rtol=1e-6):
            raise SystemExit("reinject_from_layer=n_layers disagrees with BaselineARAttention")
        rein.reinject_from_layer = 1
        if torch.allclose(plain._encode_layers(ids), rein._encode_layers(ids), atol=1e-5, rtol=1e-5):
            raise SystemExit("layer-1 reinjection did not change the forward")
        for module in (plain, rein):
            for name, parameter in module.named_parameters():
                if name.endswith(("W_Q.weight", "W_K.weight", "W_V.weight")):
                    parameter.zero_()
        if not torch.allclose(plain._encode_layers(ids), rein._encode_layers(ids), atol=1e-6, rtol=1e-6):
            raise SystemExit("zero QKV weights still differ; PE entered V or the residual")
    if bm.CausalSelfAttention is QKReinjectionCausalAttention:
        raise SystemExit("baseline CausalSelfAttention was left patched")
    print(f"baseline qk reinjection self-check ok  c={scale:.6f}")


if __name__ == "__main__":
    _self_check()
