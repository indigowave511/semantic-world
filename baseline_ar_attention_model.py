"""
Standard AR (autoregressive) QKV baseline for X(T-1) -> S(T).

Same windows as four_tensor_model.py: token_ids_full (B, T+1, N).
X(T-1) is the concatenated token sequence of sentences 0..T-1 (PAD stripped).
S(T) is generated left-to-right from that premise only.

d = d_x * d_t. Residual dropout on both branches, Pre-LN on Attn and MLP,
one LayerNorm before the tied readout. See baseline_ar_attention_model_readme.md.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _sinusoidal_pe(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Same formula as four_tensor_model._sinusoidal_pe, 1-D index."""
    half = dim // 2
    i = torch.arange(half, device=positions.device, dtype=positions.dtype)
    div = 10000.0 ** (2 * i / dim)
    angle = positions.unsqueeze(-1) / div
    pe = positions.new_zeros(*positions.shape, dim)
    pe[..., 0:2 * half:2] = torch.sin(angle)
    pe[..., 1:2 * half:2] = torch.cos(angle)
    return pe


def compute_d(d_x: int, d_t: int) -> int:
    return d_x * d_t


def compute_dff(d: int) -> int:
    """d_ff = 4 * d, matching four_tensor §1 with d = d_x d_t."""
    return 4 * d


def bos_eos_ids(content_vocab_size: int) -> tuple[int, int]:
    """Append BOS/EOS at the end of the pool vocab. Do not reuse PAD."""
    return content_vocab_size, content_vocab_size + 1


def strip_pad(row: torch.Tensor, pad_id: int) -> torch.Tensor:
    """row : (N,)  ->  1-D content ids, PAD removed."""
    return row[row != pad_id]


@dataclass
class PackedBatch:
    """Linearized windows for one batch.

    prefix      : (B, L_pre)   [BOS]+X(T-1), right-PAD
    prefix_len  : (B,)
    gold        : (B, L_gold)  S(T)+EOS, right-PAD
    gold_len    : (B,)
    """

    prefix: torch.Tensor
    prefix_len: torch.Tensor
    gold: torch.Tensor
    gold_len: torch.Tensor


def pack_batch(
    token_ids_full: torch.Tensor,
    padding_idx: int,
    bos_id: int,
    eos_id: int,
) -> PackedBatch:
    """token_ids_full : (B, T+1, N)  ->  PackedBatch.

    Per sentence, strip PAD. No SEP. BOS/EOS are the only added tokens.
    """
    if token_ids_full.dim() != 3:
        raise ValueError(
            f"token_ids_full must be (B, T+1, N), got {tuple(token_ids_full.shape)}"
        )
    B, T1, _N = token_ids_full.shape
    if T1 < 2:
        raise ValueError(f"T+1 must be >= 2, got {T1}")
    T = T1 - 1
    device = token_ids_full.device
    dtype = token_ids_full.dtype
    bos = torch.tensor([bos_id], device=device, dtype=dtype)
    eos = torch.tensor([eos_id], device=device, dtype=dtype)

    prefixes: list[torch.Tensor] = []
    golds: list[torch.Tensor] = []
    for b in range(B):
        parts = [strip_pad(token_ids_full[b, t], padding_idx) for t in range(T1)]
        prefix = torch.cat([bos, *parts[:T]])
        gold = torch.cat([parts[T], eos])
        prefixes.append(prefix)
        golds.append(gold)

    def _pad(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor(
            [int(s.numel()) for s in seqs], device=device, dtype=torch.long,
        )
        L = int(lengths.max().item())
        out = token_ids_full.new_full((B, L), padding_idx)
        for i, s in enumerate(seqs):
            out[i, : s.numel()] = s
        return out, lengths

    prefix, prefix_len = _pad(prefixes)
    gold, gold_len = _pad(golds)
    return PackedBatch(
        prefix=prefix,
        prefix_len=prefix_len,
        gold=gold,
        gold_len=gold_len,
    )


def causal_pad_mask(ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """ids : (B, L)  ->  blocked : (B, 1, L, L)  True = blocked key.

    Causal j>i, plus batch-PAD keys. finfo.min is applied by the caller.
    """
    if ids.dim() != 2:
        raise ValueError(f"ids must be (B, L), got {tuple(ids.shape)}")
    B, L = ids.shape
    pad = ids == padding_idx
    causal = torch.triu(
        torch.ones(L, L, device=ids.device, dtype=torch.bool), diagonal=1,
    )
    return causal[None, None, :, :] | pad[:, None, None, :]


class CausalSelfAttention(nn.Module):
    """Standard QKV. n_heads=1 is W^{Q,K,V} ∈ R^{d×d} and no W^O."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")
        self.d = d
        self.n_heads = n_heads
        self.d_h = d // n_heads
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False) if n_heads > 1 else None
        self.scale = 1.0 / math.sqrt(self.d_h if n_heads > 1 else d)

    def forward(self, H: torch.Tensor, ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
        """H : (B, L, d)  ids : (B, L)  ->  Attn : (B, L, d)"""
        B, L, d = H.shape
        blocked = causal_pad_mask(ids, padding_idx)
        neg = torch.finfo(H.dtype).min
        Q = self.W_Q(H)
        K = self.W_K(H)
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


class CausalDecoderLayer(nn.Module):
    """Pre-LN: H + drop(Attn(LN(H))), then H + drop(MLP(LN(H)))."""

    def __init__(self, d: int, n_heads: int, dropout: float, padding_idx: int):
        super().__init__()
        self.padding_idx = padding_idx
        self.ln_attn = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln_mlp = nn.LayerNorm(d)
        d_ff = compute_dff(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        Attn = self.attn(self.ln_attn(H), ids, self.padding_idx)
        H = H + self.dropout(Attn)
        Z = self.fc2(F.gelu(self.fc1(self.ln_mlp(H))))
        return H + self.dropout(Z)


class BaselineARAttention(nn.Module):
    """
    Decoder-only AR baseline.

        token_ids_full (B, T+1, N)
            pack: strip PAD, [BOS]+X(T-1)  and  S(T)+[EOS]
            d = d_x * d_t
            causal QKV, Pre-LN Attn/MLP, residual dropout, final LayerNorm, tied E^T

    training_step / training_step_fair : free-running AR (comparison path)
    generate_S3                        : eval, own tokens only
    forward_last                       : last-position tied readout (readme §9)
    forward(token_ids_full)            : DataParallel train hook; (CE sum, n_tok)
    """

    def __init__(
        self,
        vocab_size: int,
        N: int,
        T: int,
        d_x: int,
        d_t: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        padding_idx: int,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if d_x < 1 or d_t < 1:
            raise ValueError(f"d_x and d_t must be >= 1, got {d_x}, {d_t}")
        d = compute_d(d_x, d_t)
        if d < 2:
            raise ValueError(f"d=d_x*d_t must be >= 2 for sinusoidal PE, got {d}")

        self.content_vocab_size = vocab_size
        self.bos_id, self.eos_id = bos_eos_ids(vocab_size)
        self.vocab_size = vocab_size + 2
        self.N = N
        self.T = T
        self.d_x = d_x
        self.d_t = d_t
        self.d = d
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.max_len = 2 + (T + 1) * N

        self.E = nn.Embedding(self.vocab_size, d, padding_idx=padding_idx)
        self.layers = nn.ModuleList([
            CausalDecoderLayer(d, n_heads, dropout, padding_idx)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d)

    def pack(self, token_ids_full: torch.Tensor) -> PackedBatch:
        if token_ids_full.dim() != 3:
            raise ValueError(
                f"token_ids_full must be (B, T+1, N), got {tuple(token_ids_full.shape)}"
            )
        _, T1, N = token_ids_full.shape
        if N != self.N or T1 != self.T + 1:
            raise ValueError(
                f"token_ids_full (T+1, N)=({T1}, {N}) does not match "
                f"module ({self.T + 1}, {self.N})"
            )
        return pack_batch(
            token_ids_full, self.padding_idx, self.bos_id, self.eos_id,
        )

    def _encode_layers(self, ids: torch.Tensor) -> torch.Tensor:
        """ids : (B, L)  ->  H_norm : (B, L, d). Layers + LN, no readout."""
        _, L = ids.shape
        pos = torch.arange(L, device=ids.device, dtype=self.E.weight.dtype)
        H = self.E(ids) + _sinusoidal_pe(pos, self.d)
        for layer in self.layers:
            H = layer(H, ids)
        return self.final_norm(H)

    def _encode(self, ids: torch.Tensor) -> torch.Tensor:
        """ids : (B, L)  ->  H_norm : (B, L, d). Checkpointed in train (K-unroll)."""
        if ids.dim() != 2:
            raise ValueError(f"ids must be (B, L), got {tuple(ids.shape)}")
        if (ids >= self.vocab_size).any() or (ids < 0).any():
            raise ValueError(
                f"ids must be in [0, {self.vocab_size}), "
                f"got min={int(ids.min())} max={int(ids.max())}"
            )
        if self.training:
            return checkpoint(self._encode_layers, ids, use_reentrant=False)
        return self._encode_layers(ids)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """2-D ids -> logits (B, L, V). 3-D windows -> (CE sum, n_tok) for DataParallel."""
        if x.dim() == 3:
            return self._fair_stats(x)
        if lengths is None:
            return F.linear(self._encode(x), self.E.weight)
        return self.forward_last(x, lengths)

    def forward_last(
        self,
        ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tied readout at the last real token. ids : (B, L) -> (B, V).

        Slice H before E^T (readme §9). Packed rows pass lengths; else index L-1.
        Identity: forward(ids)[b, i] == forward_last(ids, lengths)[b]
        with i = lengths[b]-1 (or L-1).
        """
        H = self._encode(ids)
        B, L, _d = H.shape
        if lengths is None:
            H_last = H[:, -1, :]
        else:
            if lengths.shape != (B,):
                raise ValueError(
                    f"lengths must be (B,)={B}, got {tuple(lengths.shape)}"
                )
            idx = (lengths - 1).clamp(min=0, max=L - 1)
            H_last = H[torch.arange(B, device=ids.device), idx]
        return F.linear(H_last, self.E.weight)

    def _fair_unroll(self, token_ids_full: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor, int]:
        """Free-running CE sums. Gold is the target only.

        Returns (all_sum, n_all, content_sum, n_content). content_* drops EOS
        so the denominator matches four-tensor last-slice (non-PAD S(T) tokens).
        """
        packed = self.pack(token_ids_full)
        B = packed.prefix.size(0)
        gold_max = int(packed.gold_len.max().item())
        cap = min(self.max_len, int(packed.prefix_len.max().item()) + gold_max)
        ids = packed.prefix.new_full((B, cap), self.padding_idx)
        lengths = packed.prefix_len.clone()
        for b in range(B):
            L_pre = int(packed.prefix_len[b])
            ids[b, :L_pre] = packed.prefix[b, :L_pre]

        loss_sum = torch.zeros((), device=ids.device, dtype=torch.float32)
        content_sum = torch.zeros((), device=ids.device, dtype=torch.float32)
        n_tok = 0
        n_content = 0
        for step in range(gold_max):
            active = packed.gold_len > step
            if not bool(active.any()):
                break
            L_now = max(int(lengths.max().item()), 1)
            last = self.forward_last(ids[:, :L_now], lengths)
            tgt = packed.gold[:, step]
            loss_sum = loss_sum + F.cross_entropy(
                last[active], tgt[active], reduction="sum",
            )
            n_tok += int(active.sum().item())
            is_content = active & (tgt != self.eos_id)
            if bool(is_content.any()):
                content_sum = content_sum + F.cross_entropy(
                    last[is_content], tgt[is_content], reduction="sum",
                )
                n_content += int(is_content.sum().item())
            hat = last.argmax(dim=-1).detach()
            write_at = lengths.clamp(max=cap - 1)
            ids = ids.clone()
            ids[active, write_at[active]] = hat[active]
            lengths = torch.where(active, (lengths + 1).clamp(max=cap), lengths)
        return loss_sum, n_tok, content_sum, n_content

    def _fair_stats(self, token_ids_full: torch.Tensor) -> torch.Tensor:
        """(2,) = (CE sum, n_tok). DataParallel gathers to (n_gpus, 2)."""
        loss_sum, n_tok, _content_sum, _n_content = self._fair_unroll(token_ids_full)
        n = loss_sum.new_tensor(float(n_tok))
        return torch.stack((loss_sum, n))

    def training_step_fair(self, token_ids_full: torch.Tensor):
        """Free-running: feed own tokens for S(T). Gold is the target only.

        token_ids_full : (B, T+1, N)
        returns (loss, None)
        """
        loss_sum, n_tok, _content_sum, _n_content = self._fair_unroll(token_ids_full)
        if n_tok == 0:
            raise RuntimeError("training_step_fair: no gold tokens in the batch")
        return loss_sum / n_tok, None

    def training_step(self, token_ids_full: torch.Tensor):
        """Comparison path: free-running (Section 3)."""
        return self.training_step_fair(token_ids_full)

    @torch.no_grad()
    def generate_S3(
        self,
        token_ids_full: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Eval only. Premise [BOS]+X(T-1). Own tokens until EOS or N content.

        token_ids_full : (B, T+1, N)
        returns hat_S  : (B, N)  content tokens, right-PAD, EOS not stored.
        """
        packed = self.pack(token_ids_full)
        B = packed.prefix.size(0)
        cap = min(self.max_len, int(packed.prefix_len.max().item()) + self.N + 1)
        ids = packed.prefix.new_full((B, cap), self.padding_idx)
        lengths = packed.prefix_len.clone()
        for b in range(B):
            L_pre = int(packed.prefix_len[b])
            ids[b, :L_pre] = packed.prefix[b, :L_pre]

        hat = packed.prefix.new_full((B, self.N), self.padding_idx)
        n_content = packed.prefix_len.new_zeros(B)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        pad_hits = 0

        for _ in range(self.N):
            live = ~done
            if not bool(live.any()):
                break
            L_now = max(int(lengths.max().item()), 1)
            last = self.forward_last(ids[:, :L_now], lengths) / temperature
            nxt = last.argmax(dim=-1)
            pad_hits += int(((nxt == self.padding_idx) & live).sum().item())

            emit = live & (nxt != self.eos_id) & (n_content < self.N)
            rows = emit.nonzero(as_tuple=False).squeeze(-1)
            if rows.numel() > 0:
                hat[rows, n_content[rows]] = nxt[rows]
                n_content[rows] += 1

            write_at = lengths.clamp(max=cap - 1)
            ids = ids.clone()
            ids[live, write_at[live]] = nxt[live]
            lengths = torch.where(live, (lengths + 1).clamp(max=cap), lengths)
            done = done | (nxt == self.eos_id) | (n_content >= self.N)

        if pad_hits > 0:
            warnings.warn(
                f"[generate_S3] argmax landed on PAD at {pad_hits} step(s). "
                f"BOS/EOS must not reuse padding_idx={self.padding_idx}.",
                RuntimeWarning,
            )
        return hat

    def gold_S_last(self, token_ids_full: torch.Tensor) -> torch.Tensor:
        """Gold S(T), PAD stripped then right-padded to N. (B, N)."""
        packed = self.pack(token_ids_full)
        B = packed.gold.size(0)
        out = packed.gold.new_full((B, self.N), self.padding_idx)
        for b in range(B):
            # gold is S(T)+EOS; drop EOS
            g = packed.gold[b, : int(packed.gold_len[b])]
            if g.numel() > 0 and int(g[-1]) == self.eos_id:
                g = g[:-1]
            n = min(self.N, int(g.numel()))
            if n > 0:
                out[b, :n] = g[:n]
        return out


class BaselineARUpdater:
    """AdamW / SGD. Train path is AR (autoregressive) free-running only."""

    def __init__(
        self,
        model: "BaselineARAttention",
        lr: float = 3e-4,
        optimizer: str = "adamw",
        weight_decay: float = 0.01,
    ):
        self.model = model
        params = [p for p in model.parameters() if p.requires_grad]
        name = optimizer.lower()
        if name == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=lr)
        elif name == "adamw":
            self.optimizer = torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f"unknown optimizer {optimizer!r}; use 'adamw' or 'sgd'"
            )

    def step(self, token_ids_full: torch.Tensor):
        """Fair-AR update through self.model so DataParallel can scatter the batch.

        Do not call training_step_fair on the unwrapped module — that puts the
        full loader batch on GPU 0 (four_tensor_model.FourTensorUpdater.step).
        """
        raw = self.model.module if hasattr(self.model, "module") else self.model
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        runner = self.model
        device_ids = getattr(self.model, "device_ids", None)
        if device_ids is not None and token_ids_full.size(0) < len(device_ids):
            runner = raw
        stats = runner(token_ids_full)
        if stats.dim() == 1:
            n_tok = stats[1]
            if float(n_tok.item()) <= 0:
                raise RuntimeError("training_step_fair: no gold tokens in the batch")
            loss = stats[0] / n_tok
        else:
            n_tok = stats[:, 1].sum()
            if float(n_tok.item()) <= 0:
                raise RuntimeError("training_step_fair: no gold tokens in the batch")
            loss = stats[:, 0].sum() / n_tok
        loss.backward()
        self.optimizer.step()
        return loss, None
