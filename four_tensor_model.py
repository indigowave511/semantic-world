import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pe(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard transformer sinusoidal encoding.

    positions : float tensor of any shape
    returns   : same shape + (dim,), with
        even indices  2i     -> sin(pos / 10000^{2i/dim})
        odd  indices  2i+1   -> cos(pos / 10000^{2i/dim})
    Unused last slot (odd dim) stays 0, matching the document's `dim // 2` loop.
    """
    half = dim // 2
    i = torch.arange(half, device=positions.device, dtype=positions.dtype)
    div = 10000.0 ** (2 * i / dim)
    angle = positions.unsqueeze(-1) / div
    pe = positions.new_zeros(*positions.shape, dim)
    pe[..., 0:2 * half:2] = torch.sin(angle)
    pe[..., 1:2 * half:2] = torch.cos(angle)
    return pe


class FourTensorInputConstruction(nn.Module):
    """
    Step-1 pipeline: token ids -> rank-4 input tensor X_{xn;τq}.

        S(τ)  --+P_xn-->  S̃(τ)  --stack τ-->  X^{(0)}_{xn;τq}  --+T_{xτq}-->  X_{xn;τq}

    token_ids : (B, T, N)                 -- token_ids[..., τ, x]
    X         : (B, N, T, d_x, d_t)       -- X_{xn;τq}

    P_{xn} and T_{xτq} are fixed sinusoids (not learned). Only E_sem, E_temp are parameters.
    """

    def __init__(self, vocab_size: int, N: int, T: int, d_x: int, d_t: int,
                 padding_idx: int):
        super().__init__()
        if d_x < 2 or d_t < 2:
            raise ValueError("d_x and d_t must be >= 2 so sinusoidal pairs exist.")

        self.vocab_size = vocab_size
        self.N = N
        self.T = T
        self.d_x = d_x
        self.d_t = d_t

        # Stage A dictionaries: independent lookups into the n-fiber and the q-fiber.
        self.E_sem = nn.Embedding(vocab_size, d_x, padding_idx=padding_idx)
        self.E_temp = nn.Embedding(vocab_size, d_t, padding_idx=padding_idx)

        # Stage B: P_{xn} ∈ R^{N × d_x}, phase = x. Broadcast later over (τ, q).
        x_idx = torch.arange(N, dtype=torch.float32)
        P = _sinusoidal_pe(x_idx, d_x)                    # (N, d_x)
        self.register_buffer("P", P, persistent=False)

        # Stage D: T_{xτq} ∈ R^{N × T × d_t}, joint phase = x + τ. Broadcast later over n.
        tau_idx = torch.arange(T, dtype=torch.float32)
        phase = x_idx[:, None] + tau_idx[None, :]         # (N, T)
        Temb = _sinusoidal_pe(phase, d_t)                 # (N, T, d_t)
        self.register_buffer("Temb", Temb, persistent=False)

    def build_token_tensor(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Stage A + Stage C: dual-dictionary outer product, stacked over τ.

            S_{xnq}(τ) = E_sem(tok_{x,τ})_n · E_temp(tok_{x,τ})_q

        token_ids : (B, T, N)
        returns S : (B, N, T, d_x, d_t)  == S_{xn;τq}
        """
        e_n = self.E_sem(token_ids)                       # (B, T, N, d_x)
        e_q = self.E_temp(token_ids)                      # (B, T, N, d_t)
        # rank-1 (n, q) slice per token; stacking over τ is just the layout permute.
        S = e_n.unsqueeze(-1) * e_q.unsqueeze(-2)         # (B, T, N, d_x, d_t)
        return S.permute(0, 2, 1, 3, 4).contiguous()      # (B, N, T, d_x, d_t)

    def build_input_tensor(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : (B, T, N)  ->  X : (B, N, T, d_x, d_t)"""
        if token_ids.dim() != 3:
            raise ValueError(
                f"token_ids must be (B, T, N), got {tuple(token_ids.shape)}"
            )
        _, T, N = token_ids.shape
        if N != self.N or T != self.T:
            raise ValueError(
                f"token_ids (T, N)=({T}, {N}) does not match "
                f"module (T, N)=({self.T}, {self.N})"
            )

        # Stage A (+ C): S_{xnq}(τ) stacked into (B, N, T, d_x, d_t)
        S = self.build_token_tensor(token_ids)

        # Stage B: spatial position P_{xn}, broadcast over (τ, q)
        #   S̃_{xnq}(τ) = S_{xnq}(τ) + P_{xn}
        S_tilde = S + self.P[None, :, None, :, None]

        # Stage C is already realized by the stacked layout of S; no extra op.

        # Stage D: temporal-order embedding T_{xτq}, broadcast over n
        #   X_{xn;τq} = S̃_{xnq}(τ) + T_{xτq}
        return S_tilde + self.Temb[None, :, :, None, :]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.build_input_tensor(token_ids)


def _rank4_weight(d_x_out: int, d_x_in: int, d_t_out: int, d_t_in: int) -> nn.Parameter:
    """
    W_{mn,pq} as a linear map (n,q) -> (m,p).
    Stored as (m, p, n, q) so .view(d_x_out*d_t_out, d_x_in*d_t_in) is
    the correct flattening with zero copy.
    Xavier is computed on the flat (mp, nq) matrix — applying it to the
    4D tensor would mis-estimate fan_in.
    Theory notation stays W_{mn,pq}; storage order is (m,p,n,q) only.
    """
    mat = torch.empty(d_x_out * d_t_out, d_x_in * d_t_in)
    nn.init.xavier_uniform_(mat)
    return nn.Parameter(
        mat.view(d_x_out, d_t_out, d_x_in, d_t_in).contiguous()
    )


class FourTensorAttention(nn.Module):
    """
    Steps 2-5, one head: X_{xn;τq} -> Attn_{xm;τp}.

        Q_{xm;τp} = Σ_{n,q} W^Q_{mn,pq} X_{xn;τq}     (and analogously K, V)
        S_{xτ,x'τ'} = (1/√(d_x_out d_t_out)) Σ_{m,p} Q_{xm;τp} K_{x'm;τ'p}
        A = softmax_{x',τ'}(S)
        Attn_{xm;τp} = Σ_{x',τ'} A_{xτ,x'τ'} V_{x'm;τ'p}

    Input  X    : (B, N, T, d_x_in,  d_t_in)
    Output Attn : (B, N, T, d_x_out, d_t_out)
    For a full-size single head, d_x_out = d_x_in and d_t_out = d_t_in.
    """

    def __init__(
        self,
        d_x: int,
        d_t: int,
        d_x_out: int | None = None,
        d_t_out: int | None = None,
    ):
        super().__init__()
        d_x_out = d_x if d_x_out is None else d_x_out
        d_t_out = d_t if d_t_out is None else d_t_out
        self.d_x_in = d_x
        self.d_t_in = d_t
        self.d_x_out = d_x_out
        self.d_t_out = d_t_out
        self.d_x = d_x
        self.d_t = d_t
        self.scale = 1.0 / math.sqrt(d_x_out * d_t_out)
        self.W_Q = _rank4_weight(d_x_out, d_x, d_t_out, d_t)
        self.W_K = _rank4_weight(d_x_out, d_x, d_t_out, d_t)
        self.W_V = _rank4_weight(d_x_out, d_x, d_t_out, d_t)

    def _contract(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Y_{xm;τp} = Σ_{n,q} W_{mn,pq} X_{xn;τq}
        X : (B, N, T, d_x_in, d_t_in)
        W : (d_x_out, d_t_out, d_x_in, d_t_in)  -- stored (m, p, n, q)
        """
        B, N, T, d_x_in, d_t_in = X.shape
        d_x_out, d_t_out, w_n, w_q = W.shape
        if (w_n, w_q) != (d_x_in, d_t_in):
            raise ValueError(
                f"W in-fiber (n, q)=({w_n}, {w_q}) does not match "
                f"X (d_x, d_t)=({d_x_in}, {d_t_in})"
            )
        X_flat = X.reshape(B * N * T, d_x_in * d_t_in)
        W_flat = W.view(d_x_out * d_t_out, d_x_in * d_t_in)
        Y_flat = F.linear(X_flat, W_flat)
        return Y_flat.view(B, N, T, d_x_out, d_t_out)

    def build_qkv(self, X: torch.Tensor):
        """X -> Q, K, V, each (B, N, T, d_x_out, d_t_out)."""
        Q = self._contract(X, self.W_Q)
        K = self._contract(X, self.W_K)
        V = self._contract(X, self.W_V)
        return Q, K, V

    def compute_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Doubled Frobenius inner product over the (m, p) slice.
        Q, K    : (B, N, T, d_x_out, d_t_out)
        Scores  : (B, N, T, N, T) == S_{xτ, x'τ'}
        """
        B, N, T, d_x_out, d_t_out = Q.shape
        Qf = Q.reshape(B, N * T, d_x_out * d_t_out)
        Kf = K.reshape(B, N * T, d_x_out * d_t_out)
        scores = self.scale * torch.bmm(Qf, Kf.transpose(1, 2))
        return scores.view(B, N, T, N, T)

    @staticmethod
    def joint_softmax(
        Scores: torch.Tensor, pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Softmax jointly over the (x', τ') grid. Scores, A : (B, N, T, N, T).
        pad_mask : (B, N, T) bool, True = PAD key. Applied on the key axes
        only. finfo.min (not -inf) so an all-PAD key row becomes uniform,
        not NaN.
        """
        B, N, T, _, _ = Scores.shape
        if pad_mask is not None:
            if pad_mask.shape != (B, N, T):
                raise ValueError(
                    f"pad_mask must be (B, N, T)=({B}, {N}, {T}), "
                    f"got {tuple(pad_mask.shape)}"
                )
            neg = torch.finfo(Scores.dtype).min
            Scores = Scores.masked_fill(pad_mask[:, None, None, :, :], neg)
        return F.softmax(Scores.reshape(B, N, T, N * T), dim=-1).view(
            B, N, T, N, T,
        )

    def attention_output(self, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Attn_{xm;τp} = Σ_{x',τ'} A_{xτ,x'τ'} V_{x'm;τ'p}
        A    : (B, N, T, N, T)
        V    : (B, N, T, d_x_out, d_t_out)
        Attn : (B, N, T, d_x_out, d_t_out)
        """
        B, N, T, d_x_out, d_t_out = V.shape
        A_flat = A.reshape(B, N * T, N * T)
        Vf = V.reshape(B, N * T, d_x_out * d_t_out)
        Attn_flat = torch.bmm(A_flat, Vf)
        return Attn_flat.view(B, N, T, d_x_out, d_t_out)

    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """X : (B, N, T, d_x, d_t)  ->  Attn : (B, N, T, d_x_out, d_t_out)
        pad_mask : (B, N, T) bool, True = PAD key."""
        if X.dim() != 5:
            raise ValueError(
                f"X must be (B, N, T, d_x, d_t), got {tuple(X.shape)}"
            )
        B, N, T, d_x, d_t = X.shape
        if d_x != self.d_x_in or d_t != self.d_t_in:
            raise ValueError(
                f"X fiber (d_x, d_t)=({d_x}, {d_t}) does not match "
                f"module ({self.d_x_in}, {self.d_t_in})"
            )
        if pad_mask.shape != (B, N, T):
            raise ValueError(
                f"pad_mask must be (B, N, T)=({B}, {N}, {T}), "
                f"got {tuple(pad_mask.shape)}"
            )
        Q, K, V = self.build_qkv(X)
        Scores = self.compute_scores(Q, K)
        A = self.joint_softmax(Scores, pad_mask)
        return self.attention_output(A, V)


class FourTensorMultiHeadAttention(nn.Module):
    """
    Multi-head: H independent copies of the same rank-4 attention (Steps 2-5).

    Each head has its own W^Q, W^K, W^V ∈ R^{d_x × d_x × d_t × d_t} and
    produces a full Attn_{xm;τp} of shape (B, N, T, d_x, d_t). Nothing is split:
    both the n-fiber and the q-fiber stay size d_x and d_t in every head.

        Attn = Σ_h Attn^{(h)}

    n_heads=1 is exactly one FourTensorAttention.
    """

    def __init__(self, d_x: int, d_t: int, n_heads: int = 1):
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        self.d_x = d_x
        self.d_t = d_t
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            FourTensorAttention(d_x, d_t) for _ in range(n_heads)
        ])

    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """X, Attn : (B, N, T, d_x, d_t)   pad_mask : (B, N, T)"""
        if X.dim() != 5:
            raise ValueError(
                f"X must be (B, N, T, d_x, d_t), got {tuple(X.shape)}"
            )
        Attn = self.heads[0](X, pad_mask)
        for head in self.heads[1:]:
            Attn = Attn + head(X, pad_mask)
        return Attn


def compute_dff(d_x: int, d_t: int) -> int:
    """d_ff = 4 * d_x * d_t.
    MLP 입력 차원은 d_in = d_x*d_t 이므로, 표준 트랜스포머의
    "은닉폭 = 4*d_model" 관례를 d_model = d_x*d_t 에 그대로 적용한 것."""
    return 4 * d_x * d_t


class FourTensorMLPBlock(nn.Module):
    """
    Steps 6-8: residual after attention -> position-wise MLP -> residual after MLP.

        X^{attn}_{xn;τq} = X_{xn;τq} + dropout(Attn_{xn;τq})
        z = unflatten( W2 GELU(W1 flatten(LN(X^{attn})) + b1) + b2 )
        X_output = X^{attn} + dropout(z)

    One nn.Dropout(p) for both residual branches; each call draws an independent
    mask. p=0 is a no-op. model.eval() disables both. MLP acts independently at
    each (x, τ). The (n, q) slice is jointly flattened row-major into
    R^{d_x d_t}. Hidden width d_ff = 4 * d_x * d_t.
    """

    def __init__(self, d_x: int, d_t: int, dropout: float = 0.0):
        super().__init__()
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        self.d_x = d_x
        self.d_t = d_t
        d_in = d_x * d_t
        d_ff = compute_dff(d_x, d_t)
        self.d_ff = d_ff
        # W1 : (d_ff, d_in), W2 : (d_in, d_ff)  -- same layout as the document
        self.fc1 = nn.Linear(d_in, d_ff)
        self.fc2 = nn.Linear(d_ff, d_in)
        self.ln = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def residual_after_attention(self, X: torch.Tensor, Attn: torch.Tensor) -> torch.Tensor:
        """Step 6: X^{attn} = X + dropout(Attn)."""
        return X + self.dropout(Attn)

    def mlp_block(self, X_out_attn: torch.Tensor) -> torch.Tensor:
        """
        Step 7: position-wise MLP on the pre-normed flattened (n, q) fiber.
        X_out_attn, Z : (B, N, T, d_x, d_t)
        flatten: v[n * d_t + q] = X[x, τ, n, q]  (row-major)
        """
        B, N, T, d_x, d_t = X_out_attn.shape
        v = self.ln(X_out_attn.reshape(B, N, T, d_x * d_t)).reshape(
            B * N * T, d_x * d_t,
        )
        hidden = F.gelu(self.fc1(v))
        out = self.fc2(hidden)
        return out.view(B, N, T, d_x, d_t)

    def residual_after_mlp(self, X_out_attn: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Step 8: X_output = X^{attn} + dropout(z)."""
        return X_out_attn + self.dropout(Z)

    def forward(self, X: torch.Tensor, Attn: torch.Tensor) -> torch.Tensor:
        """X, Attn, out : (B, N, T, d_x, d_t)"""
        if X.dim() != 5 or Attn.dim() != 5:
            raise ValueError(
                f"X and Attn must be (B, N, T, d_x, d_t), "
                f"got X {tuple(X.shape)}, Attn {tuple(Attn.shape)}"
            )
        _, _, _, d_x, d_t = X.shape
        if d_x != self.d_x or d_t != self.d_t:
            raise ValueError(
                f"fiber (d_x, d_t)=({d_x}, {d_t}) does not match "
                f"module ({self.d_x}, {self.d_t})"
            )
        X_out_attn = self.residual_after_attention(X, Attn)
        Z = self.mlp_block(X_out_attn)
        return self.residual_after_mlp(X_out_attn, Z)


class FourTensorTransformerLayer(nn.Module):
    """
    One layer through Step 8, Pre-LN on each branch:

        Attn = multi-head attention(LN(X))
        X^{attn} = X + dropout(Attn)
        X_output = X^{attn} + dropout(MLP(LN(X^{attn})))

    Residual stream is the raw sum. §8 final_norm stays on the model, after
    the last layer.
    """

    def __init__(self, d_x: int, d_t: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_x = d_x
        self.d_t = d_t
        self.ln_attn = nn.LayerNorm(d_x * d_t)
        self.attention = FourTensorMultiHeadAttention(d_x, d_t, n_heads=n_heads)
        self.mlp = FourTensorMLPBlock(d_x, d_t, dropout=dropout)

    def _norm_attn(self, X: torch.Tensor) -> torch.Tensor:
        B, N, T, d_x, d_t = X.shape
        return self.ln_attn(X.reshape(B, N, T, d_x * d_t)).reshape(
            B, N, T, d_x, d_t,
        )

    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        Attn = self.attention(self._norm_attn(X), pad_mask)
        return self.mlp(X, Attn)


class FourTensorTiedReadout(nn.Module):
    """
    Step 9 (this document's revised form): tied-n / learned-q vocabulary readout.

        W_out_{nq;v} = E_sem[v,n] · W'_{q,v}
        L_{xv}(τ)    = Σ_{n,q} X_output_{xn;τq} E_sem[v,n] W'_{q,v}

    E_sem is the SAME Stage-A table (passed in, not re-created): zero new n-axis
    parameters. Only W' ∈ R^{d_t × |V|} is learned here. Logits are not yet
    probabilities.

    Contraction order: fold E_sem and W' into M_{nq,v} first (no batch axis),
    then one GEMM L = X_flat @ M. The dense [d_x, d_t, |V|] tensor is never
    stored as a parameter; M is rebuilt each forward.
    """

    def __init__(self, d_x: int, d_t: int, vocab_size: int):
        super().__init__()
        self.d_x = d_x
        self.d_t = d_t
        self.vocab_size = vocab_size
        self.W_prime = nn.Parameter(torch.empty(d_t, vocab_size))
        nn.init.xavier_uniform_(self.W_prime)

    def readout_logits_tied(
        self, X_output: torch.Tensor, E_sem: torch.Tensor,
    ) -> torch.Tensor:
        """
        X_output : (B, N, T, d_x, d_t)
        E_sem    : (vocab_size, d_x)  -- embedding.weight from Stage A
        returns L: (B, N, T, vocab_size)  == L_{xv}(τ)

        L_{iv} = <X_i, e_v ⊗ w_v>_F as a single GEMM.
        M is transient, not a learned parameter.
        """
        B, N, T, d_x, d_t = X_output.shape
        if (d_x, d_t) != (self.d_x, self.d_t):
            raise ValueError(
                f"X_output fiber (d_x, d_t)=({d_x}, {d_t}) does not match "
                f"module ({self.d_x}, {self.d_t})"
            )
        d = self.d_x * self.d_t
        X_flat = X_output.reshape(B * N * T, d)
        M = torch.einsum("vn,qv->nqv", E_sem, self.W_prime).reshape(
            d, self.vocab_size,
        )
        return (X_flat @ M).view(B, N, T, self.vocab_size)

    def forward(self, X_output: torch.Tensor, E_sem: torch.Tensor) -> torch.Tensor:
        if X_output.dim() != 5:
            raise ValueError(
                f"X_output must be (B, N, T, d_x, d_t), "
                f"got {tuple(X_output.shape)}"
            )
        _, _, _, d_x, d_t = X_output.shape
        if d_x != self.d_x or d_t != self.d_t:
            raise ValueError(
                f"X_output fiber (d_x, d_t)=({d_x}, {d_t}) does not match "
                f"module ({self.d_x}, {self.d_t})"
            )
        if E_sem.dim() != 2 or E_sem.shape != (self.vocab_size, self.d_x):
            raise ValueError(
                f"E_sem must be (vocab_size, d_x)=({self.vocab_size}, {self.d_x}), "
                f"got {tuple(E_sem.shape)}"
            )
        return self.readout_logits_tied(X_output, E_sem)

    def to_probabilities(self, X_output: torch.Tensor, E_sem: torch.Tensor):
        """Step 9 + 10: returns (Logits, P), P = softmax_v(Logits)."""
        logits = self.forward(X_output, E_sem)
        return logits, vocab_softmax(logits)


def vocab_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Step 10: softmax over the vocabulary axis, independently at each (x, τ).

        p_{xv}(τ) = exp(L_{xv}(τ)) / Σ_{v'} exp(L_{xv'}(τ))

    logits : (..., |V|)  typically (B, N, T, |V|)
    returns P with the same shape, each (x, τ) a distribution in Δ^{|V|-1}
    """
    return F.softmax(logits, dim=-1)


def build_targets(token_ids_full: torch.Tensor) -> torch.Tensor:
    """token_ids_full : (B, T+1, N)  ->  Y : (B, T, N)"""
    if token_ids_full.dim() != 3:
        raise ValueError(
            f"token_ids_full must be (B, T+1, N), got {tuple(token_ids_full.shape)}"
        )
    return token_ids_full[:, 1:, :].contiguous()


def cross_entropy_loss(
    logits: torch.Tensor,
    Y: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    logits : (B, N, T, |V|)
    Y      : (B, T, N)   -- build_targets 출력
    PAD 타깃은 ignore_index로 제외. log-sum-exp 내부 안정화.
    """
    if logits.dim() != 4:
        raise ValueError(
            f"logits must be (B, N, T, V), got {tuple(logits.shape)}"
        )
    if Y.dim() != 3:
        raise ValueError(f"Y must be (B, T, N), got {tuple(Y.shape)}")
    B, N, T, _ = logits.shape
    if Y.shape != (B, T, N):
        raise ValueError(
            f"Y must be (B, T, N)=({B}, {T}, {N}) to match logits "
            f"(B, N, T, V), got {tuple(Y.shape)}"
        )
    tgt = Y.permute(0, 2, 1).reshape(-1)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt, ignore_index=pad_id,
    )


def last_slice_token_stats(
    logits: torch.Tensor,
    token_ids_full: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token S(T) stats. PAD ignored, same denominator as CE/3 for [a,a,a,0,0].

    logits          : (B, N, T, V)  from forward({S(0)..S(T-1)})
    token_ids_full  : (B, T+1, N)
    returns (ce_sum, n_correct, n_tok) over gold S(T) non-PAD slots.
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be (B, N, T, V), got {tuple(logits.shape)}")
    if token_ids_full.dim() != 3:
        raise ValueError(
            f"token_ids_full must be (B, T+1, N), got {tuple(token_ids_full.shape)}"
        )
    last = logits[:, :, -1, :]
    gold = token_ids_full[:, -1, :]
    if last.shape[:2] != gold.shape:
        raise ValueError(
            f"last-slice (B, N)={tuple(last.shape[:2])} vs gold {tuple(gold.shape)}"
        )
    mask = gold != pad_id
    ce_sum = F.cross_entropy(
        last.reshape(-1, last.size(-1)),
        gold.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )
    n_correct = ((last.argmax(dim=-1) == gold) & mask).sum()
    n_tok = mask.sum()
    return ce_sum, n_correct, n_tok


class FourTensorModel(nn.Module):
    """
    Full forward: token ids -> X -> L layers of multi-head Attn/MLP -> logits -> CE.

        token_ids_full {S(0),...,S(T)}
            -> X_input {S(0),...,S(T-1)}   (Step 1)
            -> for layer in 1..n_layers:
                   Attn = Σ_h rank-4 attention_h(LN(X))   # each head: full (d_x, d_t)
                   X = X + dropout(Attn)
                   X = X + dropout(MLP(LN(X)))            # §9 residual dropout
            -> LayerNorm on flattened (n, q) fiber   (§8, readout boundary)
            -> Logits                       (Steps 9)
            -> Y = {S(1),...,S(T)}          (Step 11)
            -> L via F.cross_entropy        (Step 13, PAD ignore_index)
        Step 10 (softmax) and Step 12 (one-hot) are theory counterparts;
        the training path does not call them.
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
        self.vocab_size = vocab_size
        self.N = N
        self.T = T
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.input_construction = FourTensorInputConstruction(
            vocab_size, N, T, d_x, d_t, padding_idx=padding_idx,
        )
        self.layers = nn.ModuleList([
            FourTensorTransformerLayer(
                d_x, d_t, n_heads=n_heads, dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.readout = FourTensorTiedReadout(d_x, d_t, vocab_size)
        self.final_norm = nn.LayerNorm(d_x * d_t)

    def build_pad_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : (B, T, N)  ->  mask : (B, N, T) bool, True = PAD."""
        return (token_ids == self.padding_idx).permute(0, 2, 1)

    def forward_pass_to_probabilities(self, token_ids: torch.Tensor):
        """
        Diagnostic: input window {S(0),...,S(T-1)} -> (Logits, P).
        token_ids : (B, T, N)
        Logits, P : (B, N, T, |V|)
        Training uses forward() (logits only), not this method.
        """
        logits = self.forward(token_ids)
        return logits, vocab_softmax(logits)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : (B, T, N)  ->  logits : (B, N, T, |V|)"""
        if token_ids.dim() != 3:
            raise ValueError(
                f"token_ids must be (B, T, N), got {tuple(token_ids.shape)}"
            )
        pad_mask = self.build_pad_mask(token_ids)
        X = self.input_construction(token_ids)
        for layer in self.layers:
            X = layer(X, pad_mask)
        B, N, T, d_x, d_t = X.shape
        X_norm = self.final_norm(X.reshape(B, N, T, d_x * d_t)).reshape(
            B, N, T, d_x, d_t,
        )
        return self.readout(X_norm, self.input_construction.E_sem.weight)

    def generate_logits(
        self, token_ids: torch.Tensor, temperature: float = 1.0,
    ) -> torch.Tensor:
        """생성 전용. §6의 training_step logits와 텐서를 공유하지 않는다."""
        with torch.no_grad():
            logits = self.forward(token_ids)[:, :, -1, :] / temperature
            pad_hit = logits.argmax(dim=-1) == self.padding_idx
            if pad_hit.any():
                w_collapsed = self.readout.W_prime.abs().max().item() < 1e-6
                warnings.warn(
                    f"[generate_logits] argmax가 PAD(id={self.padding_idx})로 떨어진 위치 "
                    f"{pad_hit.sum().item()}/{pad_hit.numel()}개. W' collapsed={w_collapsed}. "
                    f"실제 클래스 로짓이 전부 0 미만이거나 readout이 붕괴했을 가능성 — 점검 필요.",
                    RuntimeWarning,
                )
        return logits

    def training_step(self, token_ids_full: torch.Tensor):
        """token_ids_full : (B, T+1, N)  ->  (loss, logits)"""
        if token_ids_full.dim() != 3:
            raise ValueError(
                f"token_ids_full must be (B, T+1, N), got {tuple(token_ids_full.shape)}"
            )
        token_ids = token_ids_full[:, :-1, :]
        logits = self.forward(token_ids)
        Y = build_targets(token_ids_full)
        loss = cross_entropy_loss(logits, Y, pad_id=self.padding_idx)
        return loss, logits


class FourTensorUpdater:
    """
    Learned-matrix update via autograd.

    The readme derives an explicit backward (Delta S, Delta Q/K/V, dL/dW^Q,K,V)
    and SGD W <- W - eta dL/dW. This class does the same job without that
    hand-written backward: loss.backward() fills .grad on every requires_grad
    tensor, then the optimizer writes the new values.

    Updated parameters (all of model.parameters()):
        E_sem, E_temp, per-layer ln_attn / MLP ln, per-head W_Q/W_K/W_V,
        MLP W1/b1 W2/b2, W', final_norm weight/bias
    Buffers P_{xn} and T_{xτq} are sinusoidal and are not trained.

    optimizer='sgd' is the document's plain GD rule.
    optimizer='adamw' (default) is the same autograd grads with AdamW.
    """

    def __init__(
        self,
        model: "FourTensorModel",
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
        """One teacher-forced update: forward -> autograd -> optimizer.step.

        Goes through self.model(token_ids) so DataParallel (and DDP) can
        hook the forward. Do not call training_step here — that bypasses the wrapper.
        """
        raw = self.model.module if hasattr(self.model, "module") else self.model
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        token_ids = token_ids_full[:, :-1]
        runner = self.model
        device_ids = getattr(self.model, "device_ids", None)
        if device_ids is not None and token_ids.size(0) < len(device_ids):
            runner = raw
        logits = runner(token_ids)
        Y = build_targets(token_ids_full)
        loss = cross_entropy_loss(logits, Y, pad_id=raw.padding_idx)
        loss.backward()
        self.optimizer.step()
        return loss, logits
