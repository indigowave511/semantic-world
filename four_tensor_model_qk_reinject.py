"""Q/K position reinjection from layer 1, outside four_tensor_model.py.

Layer 0 is the parent forward. From ``reinject_from_layer`` (default 1) onward,
Q and K are contracted with ``LN(X) + P + Temb``. V and the residual see ``LN(X)``
and ``X`` only. P and Temb are the existing input buffers, indexed by the
window-local ``(x, τ)``. No new parameters.

``four_tensor_model.py`` is not modified. Its classes are swapped only while
this module's ``__init__`` runs, then restored, so a later plain
``FourTensorModel`` in the same process keeps the original forward.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

import four_tensor_model as fm
from four_tensor_model import FourTensorModel


def broadcast_position(P: torch.Tensor, Temb: torch.Tensor) -> torch.Tensor:
    """P (N, d_x), Temb (N, T, d_t) -> (1, N, T, d_x, d_t).

    Same broadcast as FourTensorInputConstruction.build_input_tensor.
    """
    return P[None, :, None, :, None] + Temb[None, :, :, None, :]


class QKReinjectionAttention(fm.FourTensorAttention):
    """Parent attention when X_qk is omitted. Otherwise Q and K read X_qk, V reads X."""

    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor, X_qk: torch.Tensor | None = None):
        if X_qk is None:
            return super().forward(X, pad_mask)
        if X_qk.shape != X.shape:
            raise ValueError(
                f"X_qk shape {tuple(X_qk.shape)} does not match X {tuple(X.shape)}"
            )
        if X_qk.device != X.device or X_qk.dtype != X.dtype:
            raise ValueError(
                f"X_qk device/dtype ({X_qk.device}, {X_qk.dtype}) does not match "
                f"X ({X.device}, {X.dtype})"
            )
        Q = self._contract(X_qk, self.W_Q)
        K = self._contract(X_qk, self.W_K)
        V = self._contract(X, self.W_V)
        scores = self.compute_scores(Q, K)
        A = self.joint_softmax(scores, pad_mask)
        return self.attention_output(A, V)


class QKReinjectionMultiHead(fm.FourTensorMultiHeadAttention):
    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor, X_qk: torch.Tensor | None = None):
        if X.dim() != 5:
            raise ValueError(
                f"X must be (B, N, T, d_x, d_t), got {tuple(X.shape)}"
            )
        attn = self.heads[0](X, pad_mask, X_qk)
        for head in self.heads[1:]:
            attn = attn + head(X, pad_mask, X_qk)
        return attn


class QKReinjectionLayer(fm.FourTensorTransformerLayer):
    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor, pos: torch.Tensor | None = None):
        U = self._norm_attn(X)
        X_qk = None if pos is None else U + pos
        attn = self.attention(U, pad_mask, X_qk)
        return self.mlp(X, attn)


@contextmanager
def _reinjection_classes():
    saved = (
        fm.FourTensorAttention,
        fm.FourTensorMultiHeadAttention,
        fm.FourTensorTransformerLayer,
    )
    fm.FourTensorAttention = QKReinjectionAttention
    fm.FourTensorMultiHeadAttention = QKReinjectionMultiHead
    fm.FourTensorTransformerLayer = QKReinjectionLayer
    try:
        yield
    finally:
        (
            fm.FourTensorAttention,
            fm.FourTensorMultiHeadAttention,
            fm.FourTensorTransformerLayer,
        ) = saved


class FourTensorPositionReinjectionModel(FourTensorModel):
    """FourTensorModel with Q/K position reinjection from ``reinject_from_layer``."""

    reinject_from_layer = 1

    def __init__(self, *args, reinject_from_layer: int | None = None, **kwargs):
        if reinject_from_layer is None:
            reinject_from_layer = self.reinject_from_layer
        if reinject_from_layer < 0:
            raise ValueError(
                f"reinject_from_layer must be >= 0, got {reinject_from_layer}"
            )
        with _reinjection_classes():
            super().__init__(*args, **kwargs)
        self.reinject_from_layer = reinject_from_layer
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, QKReinjectionLayer):
                raise TypeError(
                    f"layer {index} is {type(layer).__name__}, expected QKReinjectionLayer"
                )
            if not isinstance(layer.attention, QKReinjectionMultiHead):
                raise TypeError(
                    f"layer {index} attention is {type(layer.attention).__name__}"
                )
            layer.layer_index = index

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 3:
            raise ValueError(
                f"token_ids must be (B, T, N), got {tuple(token_ids.shape)}"
            )
        pad_mask = self.build_pad_mask(token_ids)
        X = self.input_construction(token_ids)
        pos = None
        if self.reinject_from_layer < self.n_layers:
            pos = broadcast_position(
                self.input_construction.P, self.input_construction.Temb,
            )
        for layer in self.layers:
            use = pos if layer.layer_index >= self.reinject_from_layer else None
            X = layer(X, pad_mask, use)
        B, N, T, d_x, d_t = X.shape
        X_norm = self.final_norm(X.reshape(B, N, T, d_x * d_t)).reshape(
            B, N, T, d_x, d_t,
        )
        return self.readout(X_norm, self.input_construction.E_sem.weight)


def _self_check() -> None:
    torch.manual_seed(0)
    plain = FourTensorModel(
        vocab_size=11, N=4, T=3, d_x=4, d_t=4,
        n_layers=2, n_heads=2, dropout=0.0, padding_idx=0,
    )
    torch.manual_seed(0)
    rein = FourTensorPositionReinjectionModel(
        vocab_size=11, N=4, T=3, d_x=4, d_t=4,
        n_layers=2, n_heads=2, dropout=0.0, padding_idx=0,
    )
    plain_sd = plain.state_dict()
    rein_sd = rein.state_dict()
    if plain_sd.keys() != rein_sd.keys():
        raise SystemExit("state_dict keys differ from FourTensorModel")
    for key in plain_sd:
        if not torch.equal(plain_sd[key], rein_sd[key]):
            raise SystemExit(f"init mismatch at {key}")
    if sum(p.numel() for p in plain.parameters()) != sum(p.numel() for p in rein.parameters()):
        raise SystemExit("parameter count changed")

    ids = torch.randint(1, 11, (2, 3, 4))
    pad_ids = torch.zeros(2, 3, 4, dtype=torch.long)
    built = rein.input_construction(pad_ids)
    pos = broadcast_position(rein.input_construction.P, rein.input_construction.Temb)
    if not torch.allclose(built, pos.expand_as(built)):
        raise SystemExit("Pos broadcast does not match input construction on PAD tokens")

    plain.eval()
    rein.eval()
    with torch.no_grad():
        rein.reinject_from_layer = rein.n_layers
        off = rein(ids)
        base = plain(ids)
        if not torch.allclose(off, base, atol=1e-6, rtol=1e-6):
            raise SystemExit("reinject_from_layer=n_layers disagrees with FourTensorModel")
        rein.reinject_from_layer = 1
        on = rein(ids)
        if torch.allclose(on, base, atol=1e-5, rtol=1e-5):
            raise SystemExit("layer-1 reinjection did not change the forward")

        for module in (plain, rein):
            for name, parameter in module.named_parameters():
                if name.endswith(("W_Q", "W_K", "W_V")):
                    parameter.zero_()
        rein.reinject_from_layer = 1
        if not torch.allclose(plain(ids), rein(ids), atol=1e-6, rtol=1e-6):
            raise SystemExit("zero QKV weights still differ; Pos entered V or the residual")
    restored = fm.FourTensorAttention is not QKReinjectionAttention
    if not restored:
        raise SystemExit("four_tensor_model.FourTensorAttention was left patched")
    print("position reinjection self-check ok")


if __name__ == "__main__":
    _self_check()
