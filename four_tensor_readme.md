# PSF 4-Tensor Attention — Construction Document

This file starts from `Four_Tensor_Attention_readme.md` and incorporates the §1–§4 and §6–§9 revisions.

- **§1.** MLP hidden width $d_{ff} = 4\,d_x d_t$ (replacing $4\sqrt{d_x d_t}$).
- **§2.** Tied readout evaluated as $XM$, with the factored matrix $M$ materialized first (replacing the sequence-dependent intermediate $Y$).
- **§3.** Activation storage $(B, N, T, d_x, d_t)$ — $N$ and $T$ adjacent, fibers last. Theory indices $X_{xn;\tau q}$ are unchanged.
- **§4.** Weight storage $(m, p, n, q)$. Theory notation $W_{mn,pq}$ is unchanged.
- **§6.** Batch axis $B$ is always present. Training uses integer-target cross-entropy on logits, not a materialized one-hot / softmax path.
- **§7.** PAD: attention-key masking from `token_ids` is mandatory. Readout PAD-argmax is a generation-path warning, not a permanent logit mask.
- **§8.** One LayerNorm on the flattened $(n,q)$ fiber, after the last layer and before readout. Logit-scale only. Still present.
- **Pre-LN.** Per layer, `Attn(LN(X))` and `MLP(LN(X))` on the same flatten $\phi$. Residual stream is the raw sum. Does not replace §8. Live `four_tensor_model.py`; `four_tensor_model0921.py` is the pre-change backup.
- **§9.** The same dropout $p$ on both residual branches (`Attn` and MLP). Not on softmax, embeddings, `final_norm`, Pre-LN, or readout.

## Storage layout (§3, §4, §6–§9)

The map is invariant; only the in-memory axis order changes. Both layouts flatten the $(n,q)$ slice with the same $\phi$:

$$
W^{\text{new}}[m,p,n,q] \;=\; W^{\text{old}}[m,n,p,q] \;=\; \texttt{mat}\big[\,m\,d_t^{\text{out}}+p,\;\; n\,d_t^{\text{in}}+q\,\big]
$$

```
token_ids / Y     : (B, T, N)              -- [b, τ, x]
X, Q, K, V, Attn  : (B, N, T, d_x, d_t)    -- [b, x, τ, n, q]
pad_mask          : (B, N, T)              -- True = PAD key slot; from token_ids, not X
W                 : (d_x_out, d_t_out, d_x_in, d_t_in)  -- (m, p, n, q)
logits            : (B, N, T, |V|)         -- [b, x, τ, v]
phi(n, q)         = n * d_t + q
배치 B 는 항상 있다.
```

Public functions accept **only** `(B, ...)`. A single example is `unsqueeze(0)` / `[0]` at the caller. `DataLoader` yields `(B, T+1, N)` even when $B=1$. There is no `_as_batched` helper and no unbatched overload. Fiber checks read the last two axes (`shape[-2], shape[-1]` / `_, _, _, d_x, d_t = X.shape`); `dim() != 5` is not a substitute for a fiber check. Do **not** write the old layout $(N, d_x, T, d_t)$ or $(B, N, d_x, T, d_t)$ with a `permute` that swaps $d_x$ and $T$: if $T=d_x$ the shapes still match and the values are silently wrong. Verify with **all of** $d_x, d_t, T, N$ distinct. An unbatched `(T+1, N)` passed to `training_step` is a `ValueError`.

Inner pedagogical loops run independently over $b$. Every activation index below is `[b, x, τ, n, q]`.

$N$ adjacent to $T$ is what makes `reshape(B*N*T, d_x*d_t)` the flattening $x\,T+\tau$ with no permute, and makes $W.\mathrm{view}(d_x^{\text{out}} d_t^{\text{out}},\, d_x^{\text{in}} d_t^{\text{in}})$ a zero-copy map whose columns use the same $\phi$ as $X$.

Old $W$ checkpoints stored $(d_x^{\text{out}}, d_x^{\text{in}}, d_t^{\text{out}}, d_t^{\text{in}})$. Convert with `permute(0, 2, 1, 3)` on $W^Q,W^K,W^V$; if $d_x=d_t$, `load_state_dict` accepts the mixed axes without error. Hand-derived $\partial\mathcal{L}/\partial W_{mn,pq}$ compared against `W.grad` needs the same permute: `W.grad.permute(0, 2, 1, 3)`.

**Logits vs. targets.** Logits are $(B, N, T, |V|)$; $Y$ is $(B, T, N)$. These two batch layouts do **not** share axis order. `Y.permute(0, 2, 1)` is the bridge. If $N=T$, a missing permute still has the right shape — the guard is an explicit `Y.shape == (B, T, N)` check inside `cross_entropy_loss`.

---

# PSF 4-Tensor Input Construction: Step-1 Pipeline

## 1. Overview

The rank-4 input tensor $X_{xn;\tau q}$ is built through a strict sequential pipeline, not a single simultaneous sum. Each stage injects exactly one piece of information, into exactly one fiber, before the next stage proceeds:

$$
S(\tau)
\;\xrightarrow{\;+\,P_{xn}\;}\;
\tilde S(\tau)
\;\xrightarrow{\;\text{stack over }\tau\;}\;
X^{(0)}_{xn;\tau q}
\;\xrightarrow{\;+\,T_{x\tau q}\;}\;
X_{xn;\tau q}
$$

This ordering matters: it is what keeps the two positional signals (spatial $x$, temporal $\tau$) analytically separable even though they ultimately land in the same rank-4 object.

---

## 2. Stage-by-Stage Logic

### Stage A — Content embedding: $S(\tau)$

For each sentence (temporal slot) $\tau \in \{0,\dots,T-1\}$, each token at position $x \in \{0,\dots,N-1\}$ is embedded via **two separate dictionaries** (see Appendix for the full justification) and combined by outer product:

$$
S_{xnq}(\tau) = E_{\text{sem}}(\text{tok}_{x,\tau})_n \cdot E_{\text{temp}}(\text{tok}_{x,\tau})_q
\;\in\; \mathbb{R}^{N \times d_x \times d_t}
$$

$S(\tau)$ is therefore **rank-3 from the outset**, not rank-2: it already carries content along both the semantic fiber $n$ and the temporal-context fiber $q$. No positional information (spatial or temporal-order) has been added yet — only the token's own dual-dictionary content.

### Stage B — Add spatial position: $S(\tau) + P \to \tilde S(\tau)$

Spatial position embedding $P_{xn} \in \mathbb{R}^{N \times d_x}$ is added **independently, per sentence**:

$$
\tilde S_{xnq}(\tau) = S_{xnq}(\tau) + P_{xn}
$$

- $P_{xn}$ lives entirely in the **semantic fiber** $n$ (same space as content), since it is a property of *where in the sentence* a token sits — a spatial fact, not a temporal one.
- $P_{xn}$ does **not** depend on $\tau$, and — because $S_{xnq}(\tau)$ is already rank-3 — $P_{xn}$ must also broadcast over $q$: it is added identically across every $q$-slice, since spatial placement carries no temporal-context content of its own. 
- Result: $\tilde S(\tau) \in \mathbb{R}^{N \times d_x \times d_t}$ for each $\tau$, still rank-3 per sentence — $P$ does not raise or lower the rank, it only shifts content within the $(n,q)$ slice already established by Stage A.

### Stage C — Stack over $\tau$: $X^{(0)}_{xn;\tau q}$

The $T$ sentences $\{\tilde S(\tau)\}_{\tau=0}^{T-1}$ are stacked along a new temporal axis:

$$
X^{(0)}_{xn;\tau q} = \tilde S_{xnq}(\tau)
$$

Unlike the earlier rank-2 formulation, $X^{(0)}$ here is **not** empty along $q$ — it already carries the outer-product content from Stage A (Dictionary Embedding, see Appendix), now shifted by $P_{xn}$. Stacking over $\tau$ performs no computation; it simply arranges the $T$ already-rank-3 slices into the rank-4 **storage** shape $(B, N, T, d_x, d_t)$ (theory indices remain $X^{(0)}_{xn;\tau q}$). $N$ and $T$ are adjacent so a later `reshape(B*N*T, \cdot)` is the flattening $x\,T+\tau$.

**Order of operations is provably exchangeable.** Because $P_{xn}$ broadcasts identically over both $\tau$ and $q$ (Stage B), adding $P$ before stacking and adding $P$ after stacking give the same result:
$$
\left[S_{xnq}(\tau) + P_{xn}\right]_{\text{stacked over }\tau} \;=\; \left[S_{xnq}(\tau)\right]_{\text{stacked over }\tau} + P_{xn}
$$
This is precisely a consequence of $P$ being applied "independently, per sentence" — the same rule to every $\tau$, with no cross-$\tau$ coupling. It is *not* a new assumption; it falls out of Stage B's broadcast structure.

### Stage D — Add temporal-order embedding: $X^{(0)} + T \to X$

The temporal-order embedding $T_{x\tau q} \in \mathbb{R}^{N \times T \times d_t}$ is added, broadcasting over $n$ only:

$$
X_{xn;\tau q} = X^{(0)}_{xn;\tau q} + T_{x\tau q}
$$

The $q$-fiber already carries token-level temporal-context content from Stage A ($E_{\text{temp}}$). This stage adds a second, structurally distinct layer on top: an explicit **positional** signal along $q$, which is what distinguishes one $\tau$-slice from another *and* one $x$-column from another within the same $\tau$ — information that dictionary content alone cannot supply, since the same token at different $(x,\tau)$ would otherwise embed identically.

---

## 3. Why $T$ Must Depend on $x$, Not Just $\tau$

A naive choice $T_{\tau q}$ (broadcast over both $x$ and $n$) would assign the **same** temporal signal to every column $x$ at a given $\tau$. This collapses exactly the information the model needs to learn entity trajectories across the causal surface.

**Concrete example (elevator scenario):**

```
S(t=0): [I, am, in, front, of, the, elevator, 0]
S(t=2): [The, elevator, arrives, at, my, floor, 0, 0]
S(t=4): [I, step, inside, the, elevator, 0, 0, 0]
```

The token `I` occupies column $x=0$ at $\tau=0$, is absent (or displaced) at $\tau=2$, and returns to column $x=0$ at $\tau=4$ — but as part of a *different* causal moment. If $T_{\tau q}$ were column-independent, the model would have no positional handle for distinguishing "the `I` that is waiting" from "the `I` that is stepping in" beyond content alone. The whole point of the $q$-fiber is to let the model learn an arbitrary trajectory across the $(x,\tau)$ causal surface — which requires that the temporal signal itself vary with $x$.

### Resolution: joint-phase embedding $T_{x\tau q}$

$$
T_{x\tau q} =
\begin{cases}
\sin\!\left(\dfrac{x+\tau}{10000^{2i/d_t}}\right), & q = 2i \\[6pt]
\cos\!\left(\dfrac{x+\tau}{10000^{2i/d_t}}\right), & q = 2i+1
\end{cases}
\qquad i = 0,\dots,\lfloor d_t/2 \rfloor - 1
$$

This satisfies two requirements simultaneously:

1. **Column independence** (this session's requirement): for fixed $\tau$, $T_{x\tau q} \neq T_{x'\tau q}$ whenever $x \neq x'$, since $x$ enters the phase directly.
2. **Offset consistency** (carried over from the original sinusoidal-embedding argument): for any fixed offset $k$, there exists an $x$-independent linear map $M_k$ such that
$$
T_{x,\tau+k,q} = \sum_{q'} (M_k)_{qq'}\, T_{x,\tau,q'} \quad \text{for all } x,\tau.
$$
This holds because the phase shift under $\tau \to \tau+k$ is $\Delta = k/10000^{2i/d_t}$, which depends only on $k$ and $i$ — not on $x$ or $\tau$ — so the same rotation $M_k$ applies uniformly across every column. This is what guarantees the trained $W^Q, W^K, W^V$ remain valid as the window slides ($X \to X' \to X''$), regardless of which columns are involved.

---

## 4. Fiber Summary

| Signal | Symbol | Shape | Broadcast over | Lives in fiber |
|---|---|---|---|---|
| Content (dual-dictionary) | $S_{xnq}(\tau)$ | $(B, N, T, d_x, d_t)$ | — | both $n$ and $q$ |
| Spatial position | $P_{xn}$ | $(N, d_x)$ | $\tau$, $q$ (and $B$) | $n$ (semantic) |
| Temporal order | $T_{x\tau q}$ | $(N, T, d_t)$ | $n$ (and $B$) | $q$ (temporal-context) |

Shapes in the table are **storage** shapes for the stacked rank-4 object (content $S$ after Stage C). Per-sentence $S(\tau)$ remains rank-3, $(N, d_x, d_t)$, then stacked and batched to $(B, N, T, d_x, d_t)$. $P$ and $T$ keep the shapes shown; they broadcast into $(B, N, T, d_x, d_t)$ — `P[None, :, None, :, None]` and `Temb[None, :, :, None, :]`. The `None` slots must match this layout, not the old $(B, N, d_x, T, d_t)$.

The asymmetry is intentional: spatial position is a fact about **content placement**, so it joins the semantic fiber $n$ and stays blind to $q$; temporal order is a fact about **when along the causal surface**, so it joins the temporal-context fiber $q$ — but it must still carry the $x$-coordinate to distinguish trajectories, hence $T_{x\tau q}$ rather than $T_{\tau q}$. Content itself, by contrast, spans *both* fibers from the start, since the two dictionaries $E_{\text{sem}}, E_{\text{temp}}$ jointly define the token's initial $(n,q)$ slice (Appendix A.2).

---

## 5. Final Assembly

$$
\boxed{X_{xn;\tau q} = \underbrace{\left[S_{xnq}(\tau) + P_{xn}\right]}_{\text{Stages A+B, per-sentence}} + \underbrace{T_{x\tau q}}_{\text{Stage D, joint }x,\tau\text{ phase}}}
$$

$$
X \in \mathbb{R}^{N \times d_x \times T \times d_t}
\quad\text{(theory: $x,n,\tau,q$)},
\qquad
\text{stored as }(B, N, T, d_x, d_t)
$$

This is the tensor that feeds into $Q_{xm;\tau p} = W^Q_{mn,pq} X_{xn;\tau q}$ (and analogously $K, V$) in the next construction stage. Storage of $W$ is $(m,p,n,q)$; the contraction is the same map.

---

## Appendix: Dictionary Embedding — $S_{xnq}$ as a Rank-3 Tensor

### A.1 Motivation

$S(\tau)$ was originally treated as a pure $d_x$-space vector, i.e. a single dictionary lookup `token_id → R^{d_x}`. Since the $q$-fiber (temporal-context) is a first-class axis of the rank-4 formulation, the token's own embedding should itself be split across **two separate dictionaries** — one supplying its semantic content ($n$-fiber), one supplying its temporal-context signature ($q$-fiber):

$$
E_{\text{sem}} : V \to \mathbb{R}^{d_x} \qquad\qquad E_{\text{temp}} : V \to \mathbb{R}^{d_t}
$$

This makes $S(\tau)$ a rank-3 tensor from the outset:

$$
S_{xnq}(\tau) \in \mathbb{R}^{N \times d_x \times d_t}
$$

### A.2 Why Outer Product Is the Unique Initialization

Given two **independent** dictionaries $E_{\text{sem}}$, $E_{\text{temp}}$, the combination rule that produces the $(n,q)$ slice for a token must respect that independence at initialization:

$$
S_{xnq}(\tau) = E_{\text{sem}}(\text{tok}_{x,\tau})_n \cdot E_{\text{temp}}(\text{tok}_{x,\tau})_q
$$

This is the only logical choice, for three reasons:

1. **Any other combination collapses the separation.** If a full table $|V| \times d_x \times d_t$ were learned directly, there would be no reason to maintain two separate dictionaries in the first place — it reduces to a single, undifferentiated lookup, contradicting the premise that $E_{\text{sem}}$ and $E_{\text{temp}}$ are distinct information sources.
2. **Any mixing beyond the outer product requires extra machinery.** Concatenation-plus-projection, or a learned mixing matrix, would introduce additional parameters at the initialization stage itself — parameters whose role is exactly to re-couple what was just decoupled. This begs the question of why the split was introduced at all.
3. **It is consistent with "no shortcuts."** The rank-1 structure applies only to the *initialization* of $S$. Once $P_{xn}$, $T_{x\tau q}$ are added and the tensor passes through full $d_x^2 d_t^2$ attention, the rank-1 constraint is not preserved — the model is free to build arbitrary $(n,q)$ correlations downstream. Nothing is skipped in the computation graph; only the initial condition is minimal.

### A.3 Pseudo Code

```python
# ============================================================
# Appendix: Dictionary embedding for S_{xnq}(τ)
#   S(t) is a rank-3 tensor: S_{x,n,q} ∈ R^{N × d_x × d_t}
#   built from two SEPARATE dictionaries via outer product
#   §6: token_ids is always (B, T, N). No unbatched overload.
# ============================================================

def build_token_tensor(token_ids, N, d_x, d_t, T, E_sem, E_temp):
    """
    token_ids : int array, shape (B, T, N)      -- token_ids[b, tau, x]
    E_sem     : dictionary, token_id -> R^{d_x}   (semantic embedding table)
    E_temp    : dictionary, token_id -> R^{d_t}   (temporal-context embedding table)

    returns:
        S : rank-4 tensor, shape (B, N, T, d_x, d_t)  == S_{xn;τq}
            stored [b, x, τ, n, q]
    """
    if token_ids.ndim != 3:
        raise ValueError("token_ids must be (B, T, N)")
    B, T_in, N_in = token_ids.shape
    if (T_in, N_in) != (T, N):
        raise ValueError("token_ids (T, N) does not match module (T, N)")

    S = zeros(shape=(B, N, T, d_x, d_t))

    for b in range(B):
        for tau in range(T):
            for x in range(N):
                token_id = token_ids[b, tau, x]
                e_n = E_sem[token_id]     # -> R^{d_x}
                e_q = E_temp[token_id]    # -> R^{d_t}
                for n in range(d_x):
                    for q in range(d_t):
                        S[b, x, tau, n, q] = e_n[n] * e_q[q]

    return S   # shape (B, N, T, d_x, d_t) == S_{xn;τq}


# ============================================================
# Full Step-1 pipeline using rank-3 S_{xnq}(τ) as the starting point
# ============================================================

def build_input_tensor(token_ids, N, d_x, d_t, T, E_sem, E_temp):
    """token_ids : (B, T, N)  ->  X : (B, N, T, d_x, d_t)
    Public API is batched only. A single (T, N) example is unsqueeze(0)
    at the caller. No _as_batched, no squeeze on the way out.
    """
    if token_ids.ndim != 3:
        raise ValueError("token_ids must be (B, T, N)")
    B, T_in, N_in = token_ids.shape
    if (T_in, N_in) != (T, N):
        raise ValueError("token_ids (T, N) does not match module (T, N)")

    S = build_token_tensor(token_ids, N, d_x, d_t, T, E_sem, E_temp)
    # S : (B, N, T, d_x, d_t)

    P = zeros(shape=(N, d_x))
    for x in range(N):
        for i in range(d_x // 2):
            angle = x / (10000 ** (2*i / d_x))
            P[x, 2*i]     = sin(angle)
            P[x, 2*i + 1] = cos(angle)

    Temb = zeros(shape=(N, T, d_t))
    for x in range(N):
        for tau in range(T):
            for i in range(d_t // 2):
                angle = (x + tau) / (10000 ** (2*i / d_t))
                Temb[x, tau, 2*i]     = sin(angle)
                Temb[x, tau, 2*i + 1] = cos(angle)

    # batched broadcast, §3 layout — NOT P[None,:,:,None,None]
    S_tilde = zeros(shape=(B, N, T, d_x, d_t))
    X = zeros(shape=(B, N, T, d_x, d_t))
    for b in range(B):
        for x in range(N):
            for tau in range(T):
                for n in range(d_x):
                    for q in range(d_t):
                        S_tilde[b, x, tau, n, q] = S[b, x, tau, n, q] + P[x, n]
                        X[b, x, tau, n, q] = S_tilde[b, x, tau, n, q] + Temb[x, tau, q]
    return X
```


### A.7 Note on $P_{xn}$'s Broadcast Scope

Because $S(\tau)$ is now rank-3 (carrying its own $q$-content from the outset), $P_{xn}$ — which remains purely spatial and must stay independent of $q$-content — is now added with broadcast over **both** $\tau$ and $q$, rather than over $\tau$ alone as in the earlier rank-2 formulation. This is consistent with the original principle that $P_{xn}$ encodes a fact about token placement only, and must not carry any $q$-fiber information of its own.

**Layout trap (§3).** On stored $X$ of shape $(B, N, T, d_x, d_t)$, $P_{xn}$ expands in the $T$ and $d_t$ slots (pedagogically $P[x,n]$ added for every $\tau,q$; in batched code `P[None, :, None, :, None]`). The old expansion into $(N, d_x, T, d_t)$ used different `None` slots (`P[:, :, None, None]` unbatched; `P[None, :, :, None, None]` batched). Reusing the old pattern on the new layout is a silent bug whenever $d_x=T$. The same holds for $T_{x\tau q}$: it expands in the $d_x$ slot (`Temb[None, :, :, None, :]` batched), not the $T$ slot.

### A.8 PAD slots are not the zero tensor (§7)

$E_{\text{sem}}$ and $E_{\text{temp}}$ both use `padding_idx`, so Stage A's outer product is exactly $0$ at PAD. Stages B and D then add position/time **regardless of token content**:

$$
X_{\text{pad}} = 0 + P[x] + T_{\text{emb}}[x,\tau] \;\neq\; 0.
$$

$X_{\text{pad}}$ is a structured nonzero vector, and so are the $K,V$ built from it. A standard transformer can treat "PAD embedding $=0$ $\Rightarrow$ $Q,K,V=0$, only softmax-denominator dilution." That premise does **not** hold here. Without a key mask, PAD injects $P$ and $T_{\text{emb}}$ into every other position's attention. $X$-norms cannot recover which slots were PAD (they overlap real-token norms). The mask is therefore built from **raw `token_ids`**, not from $X$, and applied once on the score tensor (Step 4 / §7.1). $P$ and $T_{\text{emb}}$ are non-learned buffers, but they still enter $W^{Q,K,V}$, so an unmasked PAD path contaminates those projection gradients.



# PSF 4-Tensor Attention Forward Pass: Step-2 to Step-5 Pipeline

QKV 구성 → Attention Output (single layer, single head)

§6: $X, Q, K, V, \text{Attn}$ 는 항상 $(B, N, T, d_x, d_t)$ 이다. `_as_batched` 는 없다. `dim() != 5` 이면 거부하고, 파이버는 `_, _, _, d_x, d_t = X.shape` 로 검사한다. `build_qkv` 아래는 배치 루프를 명시한다. 그 다음 점수·소프트맥스·어텐션 출력 루프는 각 $b$ 의 슬라이스 $X[b]$ 위에서 도는 본문이다 (저장된 모양은 선두 $B$).

```python
# ============================================================
# Step 2: QKV tensor construction
# Theory: W^Q, W^K, W^V ∈ R^{d_x × d_x × d_t × d_t}  with W_{mn,pq}
# Storage (§4): (d_x_out, d_t_out, d_x_in, d_t_in) == (m, p, n, q)
#   so W.view(d_x_out*d_t_out, d_x_in*d_t_in) is zero-copy and its
#   columns use the same φ(n,q)=n*d_t+q as X's last two axes.
#   Xavier is on the flat (mp, nq) matrix, not the 4D tensor.
# ============================================================

def rank4_weight(d_x_out, d_x_in, d_t_out, d_t_in):
    """Theory W_{mn,pq}; store (m, p, n, q)."""
    mat = empty(shape=(d_x_out * d_t_out, d_x_in * d_t_in))
    xavier_uniform_(mat)
    return mat.view(d_x_out, d_t_out, d_x_in, d_t_in)

def build_qkv(X, W_Q, W_K, W_V, N, d_x, d_t, T):
    """
    X   : (B, N, T, d_x, d_t)   == X_{xn;τq} stored [b, x, τ, n, q]
    W_Q, W_K, W_V : shape (d_x, d_t, d_x, d_t)  -- stored (m, p, n, q)
                    theory W_{mn,pq}; last two axes of W must equal last two of X.

    returns:
        Q, K, V : each (B, N, T, d_x, d_t)  == {Q,K,V}_{xm;τp}
    """
    if X.ndim != 5:
        raise ValueError("X must be (B, N, T, d_x, d_t)")
    B, N_in, T_in, d_x_in, d_t_in = X.shape
    d_x_out, d_t_out, w_n, w_q = W_Q.shape
    if (w_n, w_q) != (d_x_in, d_t_in):
        raise ValueError("W in-fiber (n, q) does not match X (d_x, d_t)")

    Q = zeros(shape=(B, N, T, d_x, d_t))
    K = zeros(shape=(B, N, T, d_x, d_t))
    V = zeros(shape=(B, N, T, d_x, d_t))

    # Q_{xm;τp} = sum_{n,q} W^Q_{mn,pq} * X_{xn;τq}
    # stored: Q[b,x,tau,m,p] += W[m,p,n,q] * X[b,x,tau,n,q]
    for b in range(B):
        for x in range(N):
            for tau in range(T):
                for m in range(d_x):
                    for p in range(d_t):
                        q_sum = 0.0
                        k_sum = 0.0
                        v_sum = 0.0
                        for n in range(d_x):
                            for q in range(d_t):
                                x_val = X[b, x, tau, n, q]
                                q_sum += W_Q[m, p, n, q] * x_val
                                k_sum += W_K[m, p, n, q] * x_val
                                v_sum += W_V[m, p, n, q] * x_val
                        Q[b, x, tau, m, p] = q_sum
                        K[b, x, tau, m, p] = k_sum
                        V[b, x, tau, m, p] = v_sum

    return Q, K, V


# ============================================================
# Step 3: Score tensor
# S_{xτ, x'τ'} = (1 / sqrt(d_x * d_t)) * sum_{m,p} Q_{xm;τp} K_{x'm;τ'p}
#   (doubled Frobenius inner product over the (m,p) slice)
# ============================================================

def compute_scores(Q, K, N, d_x, d_t, T):
    """
    Q, K : shape (N, T, d_x, d_t)
           N,T adjacent => reshape(B*N*T, d_x*d_t) is x*T+τ, no permute.

    returns:
        Scores : shape (N, T, N, T)  == S_{xτ, x'τ'}
    """
    scale = 1.0 / sqrt(d_x * d_t)
    Scores = zeros(shape=(N, T, N, T))

    for x in range(N):
        for tau in range(T):
            for xp in range(N):
                for taup in range(T):
                    s = 0.0
                    for m in range(d_x):
                        for p in range(d_t):
                            s += Q[x, tau, m, p] * K[xp, taup, m, p]
                    Scores[x, tau, xp, taup] = scale * s

    return Scores   # S_{xτ, x'τ'}


# ============================================================
# Step 4: Joint softmax over the full (x', τ') grid  (§7.1 PAD)
# A_{xτ, x'τ'} = exp(S_{xτ,x'τ'}) / sum_{x'',τ''} exp(S_{xτ,x'',τ''})
# pad_mask : (B, N, T) bool, True = PAD. Applied on KEY axes only.
# Fill with finfo.min, not -inf: an all-PAD key row must softmax to
# uniform, not NaN (-inf - (-inf)).
# ============================================================

def build_pad_mask(token_ids, padding_idx=0):
    """token_ids : (B, T, N)  ->  mask : (B, N, T) bool, True = PAD.
    permute matches §3 layout (B, N, T, ...). Never infer PAD from X."""
    return (token_ids == padding_idx).permute(0, 2, 1)

def joint_softmax(Scores, pad_mask=None):
    """
    Scores   : (B, N, T, N, T)
    pad_mask : (B, N, T) bool, True = PAD key. Broadcast onto the last
               two axes: pad_mask[:, None, None, :, :]. Query axes are
               not masked — PAD queries are dropped by ignore_index.
    """
    B, N, T, _, _ = Scores.shape
    if pad_mask is not None:
        neg = finfo(Scores.dtype).min
        # key-axis fill: Scores[b, x, tau, xp, taup] <- neg if pad_mask[b, xp, taup]
        for b in range(B):
            for xp in range(N):
                for taup in range(T):
                    if pad_mask[b, xp, taup]:
                        for x in range(N):
                            for tau in range(T):
                                Scores[b, x, tau, xp, taup] = neg

    A = zeros(shape=(B, N, T, N, T))
    for b in range(B):
        for x in range(N):
            for tau in range(T):
                max_val = -inf
                for xpp in range(N):
                    for taupp in range(T):
                        if Scores[b, x, tau, xpp, taupp] > max_val:
                            max_val = Scores[b, x, tau, xpp, taupp]
                denom = 0.0
                for xpp in range(N):
                    for taupp in range(T):
                        denom += exp(Scores[b, x, tau, xpp, taupp] - max_val)
                for xp in range(N):
                    for taup in range(T):
                        A[b, x, tau, xp, taup] = (
                            exp(Scores[b, x, tau, xp, taup] - max_val) / denom
                        )
    return A


# ============================================================
# Step 5: Attention output (single layer, single head)
# Attn_{xm;τp} = sum_{x',τ'} A_{xτ,x'τ'} * V_{x'm;τ'p}
# ============================================================

def attention_output(A, V, N, d_x, d_t, T):
    """
    A : shape (N, T, N, T)      == A_{xτ, x'τ'}
    V : shape (N, T, d_x, d_t)  == V_{x'm;τ'p}

    returns:
        Attn : shape (N, T, d_x, d_t)  == Attn_{xm;τp}
    """
    Attn = zeros(shape=(N, T, d_x, d_t))

    for x in range(N):
        for tau in range(T):
            for m in range(d_x):
                for p in range(d_t):
                    acc = 0.0
                    for xp in range(N):
                        for taup in range(T):
                            acc += A[x, tau, xp, taup] * V[xp, taup, m, p]
                    Attn[x, tau, m, p] = acc

    return Attn   # Attn_{xm;τp}, shape (N, T, d_x, d_t)


# ============================================================
# Top-level: single layer, single head, up to attention output
# ============================================================

def single_layer_single_head_attention(X, W_Q, W_K, W_V, N, d_x, d_t, T, pad_mask=None):
    Q, K, V = build_qkv(X, W_Q, W_K, W_V, N, d_x, d_t, T)
    Scores  = compute_scores(Q, K, N, d_x, d_t, T)
    A       = joint_softmax(Scores, pad_mask)
    Attn    = attention_output(A, V, N, d_x, d_t, T)
    return Attn   # shape (B, N, T, d_x, d_t), same shape as X
```

## PAD in attention (§7.1)

Joint softmax normalizes over the full $(x',\tau')$ grid, so the mask is $(B, N, T)$, not a 1-D sequence mask, and broadcasts onto `Scores` of shape $(B, N, T, N, T)$ as `pad_mask[:, None, None, :, :]`.

**Keys only.** PAD must not pollute other positions' attention. PAD **queries** need not be masked: their outputs are dropped by `ignore_index` in the loss. MLP is position-wise and has no cross-slot path, so it does not take a mask.

**`finfo.min`, not $\texttt{-inf}$.** Filling a score with a large negative kills $\exp(S-\max)$ at that key (measured PAD attention mass $2.87 \to 0$). If every key of a query is PAD, $\max=\texttt{-inf}$ makes $-\infty-(-\infty)=\text{NaN}$; `finfo.min` makes $\max$ finite and softmax falls back to **uniform**. An entity that is PAD at every $\tau$ can occur depending on the batch, so this is a real degeneracy, not a style choice.

**Reuse vs. recache.** One `forward` call builds `pad_mask` once from that call's `token_ids` and reuses it across depth layers: layers rewrite $X$'s values, not which slot held which token. Do **not** cache the mask on the module. Window-sliding generation changes `token_ids` between calls; a previously PAD slot may become a generated token. `build_pad_mask` runs **inside each** `forward` from the `token_ids` of that call.

---


# PSF Rank-4 Model: Attention Output → MLP → Readout → Loss

This document picks up from $X_{xn;\tau q}$ (the completed rank-4 input tensor) and covers the remainder of a single-layer, single-head forward pass through to the training loss: **residual after attention (dropout) → MLP block → residual after MLP (dropout) → readout-boundary LayerNorm → vocabulary readout → softmax → cross-entropy loss**, including the $t$-index shift that connects the attention update to the training targets.

---

## 1. Pipeline Overview

$$
X_{xn;\tau q}
\;\xrightarrow{\text{Attn}}\;
\text{Attn}_{xm;\tau p}
\;\xrightarrow{+\,\mathrm{drop}}\;
X^{\text{attn}}_{xn;\tau q}
\;\xrightarrow{\text{MLP}}\;
z_{xn;\tau q}
\;\xrightarrow{+\,\mathrm{drop}}\;
X_{\text{output},\,xn;\tau q}
\;\xrightarrow{\text{LN}_{nq}}\;
X^{\text{norm}}
\;\xrightarrow{\text{readout}}\;
L_{xv}(t)
\;\xrightarrow{\text{softmax}_v}\;
p_{xv}(t)
\;\xrightarrow{\text{cross-entropy}}\;
\mathcal{L}
$$

Everything up through $\text{Attn}_{xm;\tau p}$ is covered in the QKV/score/softmax/attention-output construction (prior document). This document starts at the residual connection immediately after attention.

---

## 2. Residual After Attention (Step 6)

Following Eq. (8) of the source PDF:

$$
X^{\text{attn}}_{xn;\tau q} = X_{xn;\tau q} + \mathrm{dropout}(\text{Attn}_{xn;\tau q})
$$

Indices align directly — the attention output lives in the same $(x, n, \tau, q)$ space as the input, since $Q, K, V$ were all built by contracting the same $(n,q)$ fibers via $W^Q, W^K, W^V \in \mathbb{R}^{d_x \times d_x \times d_t \times d_t}$.

**§9.** Dropout is on the residual **branch**, not on $X$. The same $p$ is used on the MLP residual (Step 8). $p=0$ is a no-op; `eval()` turns both off. Do not drop out `joint_softmax` (that would fight §7.1), embeddings, `final_norm`, or readout. Recommended $p=0.1$ (raise to $0.2$ if the val gap grows; $0.3$ is steep on an $N\times T=64$ grid). Separate $p$ per branch is not used. This is generalization, not a substitute for §8 logit-scale control, AdamW weight decay, or early-stop.

---

## 3. Position-Wise MLP Block (Step 7)

The MLP acts **independently at each $(x, \tau)$ position**, exactly as in the source PDF's Section 2.1, but generalized: since $(n,q)$ are two separate fibers rather than a single $d$-dimensional space, they are **jointly flattened** into one feature vector of size $d_x \cdot d_t$ before entering the MLP — consistent with the "no separability assumption" principle already used at readout (Section 4 below).

$$
z_{xn;\tau q} = \text{unflatten}\Big( W_2\,\sigma\big(W_1\, \text{flatten}(X^{\text{attn}}_{x\cdot;\tau\cdot}) + b_1\big) + b_2 \Big)
$$

- $\text{flatten}: \mathbb{R}^{d_x \times d_t} \to \mathbb{R}^{d_x d_t}$, and its inverse $\text{unflatten}$, act on the $(n,q)$ slice at fixed $(x,\tau)$.
- $W_1 \in \mathbb{R}^{d_{ff} \times (d_x d_t)}$, $W_2 \in \mathbb{R}^{(d_x d_t) \times d_{ff}}$, $\sigma$ = GELU (or similar).
- **Hidden dimension**: $d_{ff} = 4\, d_x d_t$. The MLP input is $d_{\text{in}} = d_x d_t$ (the jointly flattened $(n,q)$ slice), so this is the standard transformer convention "hidden width $= 4 \times d_{\text{model}}$" with $d_{\text{model}} = d_x d_t$. The earlier geometric-mean rule $d_{ff} = 4\sqrt{d_x d_t}$ does **not** preserve that ratio: $d_{ff}/d_{\text{in}} = 4/d \to 0$ as $d$ grows, and the $d_{\text{in}}\to d_{ff}$ projection would collapse dimensions whenever $d_x d_t > 16$. The $\sqrt{\cdot}$ in the attention scale $1/\sqrt{d_x d_t}$ is a length-$d_x d_t$ inner-product variance normalization, not an "effective fiber dimension," and does not transfer to the MLP width. Only MLP FLOPs align with a standard transformer under this rule; QKV remains $d_x^2 d_t^2$.
- As in the source PDF, the residual connection is preserved as part of this block's *output*, added in the next step — the MLP itself only computes $z$.

---

## 4. Residual After MLP (Step 8)

$$
X_{\text{output},\,xn;\tau q} = X^{\text{attn}}_{xn;\tau q} + \mathrm{dropout}(z_{xn;\tau q})
$$

This is the completed single-layer transformer output (same $p$ as the attention residual). It feeds layer $\ell+1$, or — after §8 `final_norm` — the readout in the last layer. Live code also Pre-LNs the **branch inputs** (Section 4.2) before Attn and MLP; the residual **adds** are still this equation.

---

## 4.1 Readout-boundary LayerNorm (§8)

§8 is still readout-boundary LN only. It does not Pre-LN the layer. Heads may be summed without $\sqrt{H}$, and two residuals are added, so last-layer $X$ RMS can grow. The readout is still the linear map $L = X_{\text{flat}} M$, hence $\mathrm{logit\,std}\propto \|X\|\,\|M\|$. Scaling $X$ at the readout boundary inflates CE (target probability underflows; loss $\approx k(\max z - z_y)$) and collapses the softmax gradient to a one-hot (nonzero grads on a few classes; gradient *norm* stays alive so training does not look "stuck"). In fp16, logits past $\pm 65504$ yield `loss=NaN`.

Intervention is one standard LayerNorm on the jointly flattened $(n,q)$ fiber, **after the last layer and before readout** — an output-boundary scale fix, in the same class as rank-1 Stage A and rank-1 $M$. Lives on `FourTensorModel.final_norm`. Per-layer Pre-LN (Section 4.2) is a separate change: it bounds **QKV scores** inside the stack; §8 still bounds **logits**.

$$
X^{\text{norm}}_{x\cdot;\tau\cdot} = \mathrm{LayerNorm}_{d_x d_t}\big(\mathrm{flatten}(X_{\text{output},\,x\cdot;\tau\cdot})\big)
\quad\text{then unflatten},
\quad \phi(n,q)=n\,d_t+q.
$$

Same $\phi$ as §2 GEMM and §3 MLP flatten. $n$ and $q$ are **not** normalized separately. `reshape`, not `view` (non-contiguous tensors). Affine $\gamma,\beta$ are trained (`nn.LayerNorm(d_x*d_t)`, `eps=1e-5`). LayerNorm is per $(x,\tau)$, so it does not mix slots and is orthogonal to §7.1. PAD logits stay $0$ because $E_{\text{sem}}[\text{pad}]=0$ still forces $M_{:,0}=0$; §7.2 still fires if $W'=0$.

`training_step` and `generate_logits` call `forward`, so both see the same `final_norm` (no signature change). This does **not** bound $\|M\|$ or affine $\gamma$, does not stabilize intra-layer scores (that is Pre-LN), and does not add $\sqrt{H}$ head scaling. §9 dropout is generalization; §8 is logit scale; Pre-LN is branch-input scale — they do not replace each other.

---

## 4.2 Pre-LN (live code; $L\ge 2$)

ROCStories H2L1 trained (window CE $\to 1.35$). H2L2 with only §8 did not: train CE stayed $\sim 6$ and rose, last-slice acc stuck at the unigram rate. That was **not** logit blow-up (`final_norm` already unit-normalizes the readout fiber). It was the second QKV on an unnormalized residual stream: $Q,K\propto X$ so scores $\propto \|X\|^2$, softmax peaks, attention gradients die.

Live `four_tensor_model.py` keeps §8 and adds Pre-LN on each **branch**, residual stream unchanged:

$$
\mathrm{Attn}=\mathrm{Attention}\big(\mathrm{LN}_{d_x d_t}(\mathrm{flatten}(X))\big),\quad
X^{\text{attn}}=X+\mathrm{dropout}(\mathrm{Attn})
$$
$$
z=\mathrm{MLP}\big(\mathrm{LN}_{d_x d_t}(\mathrm{flatten}(X^{\text{attn}}))\big),\quad
X_{\text{output}}=X^{\text{attn}}+\mathrm{dropout}(z)
$$

Same $\phi(n,q)=n\,d_t+q$ as §8. Two `nn.LayerNorm(d_x d_t)` per layer (`ln_attn` on the layer, `ln` on the MLP block). Do not share one LN across Attn and MLP; do not LN after the add; do not drop `final_norm`.

$W^Q,W^K,W^V$ now contract $\mathrm{LN}(X)$, not raw $X_{xn;\tau q}$. Score scale $1/\sqrt{d_x d_t}$ and any QKV decomposition written for an unnormalized residual need that restatement. Backup without Pre-LN: `four_tensor_model0921.py`.

# PSF 4-Tensor Attention Output: Step-6 to Step-8


```python
# ============================================================
# Step 6: Residual connection after attention (§9 dropout on the branch)
# X_out_attn = X + dropout(Attn)
# ============================================================

def residual_after_attention(X, Attn, N, d_x, d_t, T, drop=None):
    """
    X, Attn : stored (B, N, T, d_x, d_t)  [pedagogical loops on one b-slice]
    drop    : one Dropout(p) shared with the MLP residual; each call is an
              independent mask. None or p=0 => identity. eval() disables both.
    """
    X_out_attn = zeros(shape=(N, T, d_x, d_t))
    Attn_d = drop(Attn) if drop is not None else Attn
    for x in range(N):
        for tau in range(T):
            for n in range(d_x):
                for q in range(d_t):
                    X_out_attn[x, tau, n, q] = X[x, tau, n, q] + Attn_d[x, tau, n, q]
    return X_out_attn


# ============================================================
# Step 7: Position-wise MLP block (fully unfolded / flattened)
#   Acts independently at each (x, τ) position.
#   (n, q) jointly flattened into one feature vector of size d_x*d_t.
#
#   d_ff = 4 * d_x * d_t
#   -- d_in = d_x*d_t, so this is the standard 4*d_model convention
#      with d_model = d_x*d_t. NOT 4*sqrt(d_x*d_t): that ratio
#      d_ff/d_in = 4/d -> 0 as d grows, and is not analogous to
#      the attention scale 1/sqrt(d_x*d_t).
#   Layout (§3+§6): X is (B, N, T, d_x, d_t), so
#      reshape(B*N*T, d_x*d_t) is x*T+τ then φ(n,q)=n*d_t+q. No permute.
# ============================================================

def flatten_nq(vec_nq, d_x, d_t):
    """ (d_x, d_t) matrix -> (d_x*d_t,) vector, row-major """
    out = zeros(shape=(d_x * d_t,))
    for n in range(d_x):
        for q in range(d_t):
            out[n * d_t + q] = vec_nq[n, q]
    return out

def unflatten_nq(vec_flat, d_x, d_t):
    """ (d_x*d_t,) vector -> (d_x, d_t) matrix, inverse of flatten_nq """
    out = zeros(shape=(d_x, d_t))
    for n in range(d_x):
        for q in range(d_t):
            out[n, q] = vec_flat[n * d_t + q]
    return out

def compute_dff(d_x, d_t):
    """ d_ff = 4 * d_x * d_t.
    MLP 입력 차원은 d_in = d_x*d_t 이므로, 표준 트랜스포머의
    "은닉폭 = 4*d_model" 관례를 d_model = d_x*d_t 에 그대로 적용한 것.
    """
    return 4 * d_x * d_t

def mlp_block(X_out_attn, W1, b1, W2, b2, N, d_x, d_t, T):
    """
    X_out_attn : shape (N, T, d_x, d_t)
    d_ff = 4 * d_x * d_t
    W1 : shape (d_ff, d_x*d_t)
    b1 : shape (d_ff,)
    W2 : shape (d_x*d_t, d_ff)
    b2 : shape (d_x*d_t,)

    returns:
        Z : shape (N, T, d_x, d_t)  == z_{xn;τq}, MLP output (pre-residual)
    """
    d_ff = compute_dff(d_x, d_t)
    Z = zeros(shape=(N, T, d_x, d_t))

    for x in range(N):
        for tau in range(T):
            slice_nq = X_out_attn[x, tau, :, :]        # (d_x, d_t), last two axes
            v = flatten_nq(slice_nq, d_x, d_t)          # (d_x*d_t,)

            hidden = zeros(shape=(d_ff,))
            for f in range(d_ff):
                s = b1[f]
                for k in range(d_x * d_t):
                    s += W1[f, k] * v[k]
                hidden[f] = gelu(s)

            out = zeros(shape=(d_x * d_t,))
            for k in range(d_x * d_t):
                s = b2[k]
                for f in range(d_ff):
                    s += W2[k, f] * hidden[f]
                out[k] = s

            Z[x, tau, :, :] = unflatten_nq(out, d_x, d_t)

    return Z   # z_{xn;τq}, shape (N, T, d_x, d_t)


# ============================================================
# Step 8: Second residual connection -> layer output
# X_output = X_out_attn + dropout(z)   -- same p as Step 6
# ============================================================

def residual_after_mlp(X_out_attn, Z, N, d_x, d_t, T, drop=None):
    X_output = zeros(shape=(N, T, d_x, d_t))
    Z_d = drop(Z) if drop is not None else Z
    for x in range(N):
        for tau in range(T):
            for n in range(d_x):
                for q in range(d_t):
                    X_output[x, tau, n, q] = X_out_attn[x, tau, n, q] + Z_d[x, tau, n, q]
    return X_output


# ============================================================
# §8: readout-boundary LayerNorm on flatten (n,q)
# ============================================================

def final_norm(X_output, ln, N, d_x, d_t, T):
    """X_output : (B, N, T, d_x, d_t)
    ln : LayerNorm(d_x*d_t), last axis only, same φ as MLP/readout.
    reshape, not view. Does not mix (x,τ)."""
    B, N, T, d_x, d_t = X_output.shape
    flat = X_output.reshape(B, N, T, d_x * d_t)
    return ln(flat).reshape(B, N, T, d_x, d_t)


# ============================================================
# Top-level layer: Attn residual drop -> MLP -> MLP residual drop
# Pedagogical residual algebra (this block). Live code Pre-LNs the
# branch inputs first (Section 4.2):
#   Attn = attention(LN(X));  X = X + drop(Attn)
#   Z    = mlp(LN(X));        X = X + drop(Z)
# final_norm is still NOT here — once, after the last layer (§8)
# ============================================================

def transformer_layer(X, W_Q, W_K, W_V, W1, b1, W2, b2, N, d_x, d_t, T,
                      pad_mask=None, drop=None):
    Q, K, V     = build_qkv(X, W_Q, W_K, W_V, N, d_x, d_t, T)
    Scores      = compute_scores(Q, K, N, d_x, d_t, T)
    A           = joint_softmax(Scores, pad_mask)   # no dropout here (§7.1)
    Attn        = attention_output(A, V, N, d_x, d_t, T)

    X_out_attn  = residual_after_attention(X, Attn, N, d_x, d_t, T, drop)
    Z           = mlp_block(X_out_attn, W1, b1, W2, b2, N, d_x, d_t, T)
    X_output    = residual_after_mlp(X_out_attn, Z, N, d_x, d_t, T, drop)

    return X_output   # MLP is position-wise: no pad_mask


```

---

## 5. Vocabulary Readout: Which Fibers Survive (Step 9)

This section follows the source PDF's argument directly, generalized from a single $d$ to separate $d_x, d_t$.

### 5.1 The Loss Fixes the Readout's Free Indices

The training loss is defined **pointwise over $t$**, and each $S(t)$ is itself indexed by token position $x$:

$$
\mathcal{L} = \sum_{t} \big\| \hat S(t) - S(t) \big\|^2 \quad \text{(embedding-level form, prior to cross-entropy — see Section 7)}
$$

Neither $n$ (semantic fiber) nor $q$ (temporal-context fiber) appears as a free index anywhere in this loss — they are internal to the representation, not part of what is being compared. This fixes, **without any additional assumption**, which fibers of $X_{\text{output},\,xn;\tau q}$ must survive the readout and which must be eliminated:

$$
(x, \tau) \;\; \text{preserved}, \qquad (n, q) \;\; \text{eliminated}
$$

$x$ and $\tau$ must remain because the loss compares $\hat S(t)$ against $S(t)$ separately for each temporal slice and each token position. $n$ and $q$ are contracted away because neither survives independently in $\mathcal{L}$.

### 5.2 Joint Contraction of $(n, q)$ — No Separability Assumed

At fixed $(x, \tau)$, the pair $(n, q)$ spans a $d_x \times d_t$ matrix. The most general linear map from this matrix space to $\mathbb{R}^{|V|}$ is the **joint (bilinear) contraction**:

$$
L_{xv}(\tau) = \sum_{n,q} X_{\text{output},\,xn;\tau q} \; W_{\text{out},\,nq;v}, \qquad W_{\text{out}} \in \mathbb{R}^{d_x \times d_t \times |V|}
$$

Implementation contracts $X^{\text{norm}}$ from §8, not raw $X_{\text{output}}$. The index pattern is the same; only the scale of the $(n,q)$ fiber has been fixed.

This makes **no separability assumption** between $n$ and $q$ — no particular structure, ordering, or factorization is imposed on how the two fibers combine before reaching the vocabulary axis. (Note: shape generalized from the source PDF's $\mathbb{R}^{d \times d \times |V|}$ to $\mathbb{R}^{d_x \times d_t \times |V|}$, consistent with the $d_x \neq d_t$ convention adopted throughout this construction.)

### 5.3 $W_{\text{out}}$ Is a Learned Parameter

$W_{\text{out}}$ is trained **jointly** with $W^Q, W^K, W^V$ and the MLP weights via gradient descent on $\mathcal{L}$, exactly as the other weight tensors are. There is no principled way to fix $W_{\text{out}}$ by hand: the $(n,q)$ feature space is itself learned and emergent through the attention updates, so the correct map from that space into vocabulary space cannot be known in advance and must be optimized jointly with it.

### 5.3.1 Structured Factorization: Vocabulary-Aligned $n$-Fiber, Freshly Learned $q$-Fiber

The dense form of $W_{\text{out}} \in \mathbb{R}^{d_x \times d_t \times |V|}$ (Section 5.2) is the most general choice, but it is not the only one consistent with the joint-contraction requirement — and the two fibers are not symmetric in where their content originates, which motivates a structured alternative.

**Why the two fibers are not symmetric.** The $n$-fiber (semantic, paired with $x$) originates from $E_{\text{sem}} \in \mathbb{R}^{|V| \times d_x}$ (Appendix A) — a table that is **already vocabulary-aligned by construction**. Reusing $E_{\text{sem}}$ at the readout to map the $n$-axis back onto the vocabulary is therefore standard weight tying, not an approximation: the same table that assigns each vocabulary token its semantic coordinates is reused, unmodified, to read those coordinates back out.

The $q$-fiber (temporal-context, paired with $\tau$), by contrast, receives its content from $E_{\text{temp}}$ *and* from $P_{xn}$, $T_{x\tau q}$ — none of which are vocabulary-aligned tables in the readout sense; $P$ and $T$ are positional, not lexical, and even $E_{\text{temp}}$'s role is to give a token a temporal-context signature, not a vocabulary-indexed slot to be read back through. There is consequently no existing vocabulary-aligned table for the $q$-axis to tie against. The asymmetry in the readout is therefore not a simplification imposed on a symmetric structure — it reflects an asymmetry that was already present in how $n$ and $q$ were populated (Appendix A, Stage A/D).

**Factorized form:**

$$
W_{\text{out},\,nq;v} = E_{\text{sem}}[v,n] \cdot W'_{q,v}, \qquad W' \in \mathbb{R}^{d_t \times |V|}
$$

Substituting into the logit (Eq. from Section 5.2):

$$
L_{xv}(\tau) = \sum_{n,q} X_{\text{output},\,xn;\tau q}\, E_{\text{sem}}[v,n]\, W'_{q,v}
$$

The $n$-axis is contracted directly against $E_{\text{sem}}$ — the identical table used to build $S_{xnq}(\tau)$ in Appendix A — contributing **zero new parameters**. Only the $q$-axis passes through a freshly learned projection $W' \in \mathbb{R}^{d_t \times |V|}$. This is a rank-1-in-$n$, full-rank-in-$q$ factorization of the $[d_x, d_t, |V|]$ readout tensor.

**Evaluation order (path B).** The identity above can be grouped two ways. Grouping $n$ first produces a sequence-dependent intermediate $Y_{x\tau qv} = \sum_n X_{\text{output},\,xn;\tau q}\, E_{\text{sem}}[v,n]$ of shape $NT \times d_t \times |V|$ (path A) — at the documented scale, several GB, and vocabulary chunking does not help because autograd retains every chunk. Grouping the factors of $W_{\text{out}}$ first materializes a **sequence-independent** matrix

$$
M_{nq;v} = E_{\text{sem}}[v,n]\, W'_{q,v}, \qquad \phi(n,q) = n\, d_t + q, \qquad M \in \mathbb{R}^{d_x d_t \times |V|},
$$

then $L_{xv}(\tau) = \sum_{n,q} X_{\text{output},\,xn;\tau q}\, M_{\phi(n,q),\,v}$ (equivalently $X_{\text{flat}} M$ after flattening the $(n,q)$ slice). Under the §3 layout $X$ is already $(B, N, T, d_x, d_t)$, so $X_{\text{flat}} = X.\mathrm{reshape}(BNT,\, d_x d_t)$ needs **no permute**. The old unpacking $N, d_x, T, d_t$ plus `permute` that swaps $d_x$ and $T$ must not be restored: if $T=d_x$ the reshape size matches and the values are wrong. $M$ does not grow with $B$, $N$, or $T$. The logits $L$ of size $BNT|V|$ remain a lower bound. Softmax is diagnostic (§6). PAD keys are masked at scores (§7.1); PAD **targets** use `ignore_index`; generation PAD-argmax is a warning (§7.2), not a permanent readout mask. Fiber check: the last two axes of $X^{\text{norm}}$ must be $(d_x, d_t)$ in that order — a swapped $(d_t, d_x)$ layout has the same product $d_x d_t$, so a reshape would silently succeed and produce wrong logits.

| | Path A ($Y$ first) | Path B ($M$ first) |
|---|---|---|
| Intermediate | $BNT \cdot d_t \cdot |V|$ | $d_x d_t \cdot |V|$ ($=M$) |
| Depends on batch/sequence | yes | no |

**Parameter comparison** (illustrative, for a representative $(d, |V|, T)$ configuration; QKV counts shown for both the dense "no-shortcuts" $d^4$ tensor and the structured $6d^2$ decomposition from the parameter-reduction discussion):

| Configuration | QKV params | $W_{\text{out}}$ params | Total | Ratio ($T{=}3$) | Ratio ($T{=}6$) |
|---|---|---|---|---|---|
| No-shortcuts QKV + dense $W_{\text{out}}$ | 30,000 | 158,500 | 188,500 | 3.64 | 5.27 |
| No-shortcuts QKV + **tied-$n$ / learned-$q$** | 30,000 | **15,850** | **45,850** | **0.89** | 1.28 |
| $6d^2$ QKV + dense $W_{\text{out}}$ | 1,800 | 158,500 | 160,300 | 3.09 | 4.49 |
| $6d^2$ QKV + **tied-$n$ / learned-$q$** | 1,800 | **15,850** | **17,650** | **0.34** | **0.49** |

Tying the $n$-axis to $E_{\text{sem}}$ shrinks $W_{\text{out}}$ by roughly a factor of $|V|$'s contribution through $d_x$ (here, $\sim 10\times$: 158,500 → 15,850), which alone brings the $T{=}3$ parameter-to-data ratio from 3.64 down to 0.89 — into the regime where training data exceeds parameter count — **without touching $W^Q, W^K, W^V$ at all**. Combined with the $6d^2$ QKV structuring, the ratio falls to 0.34, comparable to a standard dense-attention baseline (0.31).

**Consistency with "no shortcuts."** This factorization is not applied to the attention mechanism itself — $W^Q, W^K, W^V$ remain full bilinear maps, untouched. The reduction is confined to the readout, and specifically to the one place where an analytic justification exists independent of parameter economy: $n$ already has a vocabulary-aligned source to tie to, while $q$ structurally does not. This is exactly the kind of "analytically necessary" exception the no-shortcuts principle allows, rather than a heuristic simplification applied for convenience. §8 `final_norm` is the matching output-boundary scale fix: it does not enter attention or MLP.

### 5.4 Softmax as a Map onto the Vocabulary Simplex (Step 10)

$L_{xv}(\tau)$ is a logit, **not yet a probability**. A separate softmax over the vocabulary axis is required:

$$
p_{xv}(\tau) = \text{softmax}_v\big(L_{x\cdot}(\tau)\big) = \frac{\exp(L_{xv}(\tau))}{\sum_{v'} \exp(L_{xv'}(\tau))}, \qquad p(\tau) \in \Delta^{|V|-1} \times \cdots \times \Delta^{|V|-1} \;\; (N \text{ copies})
$$

This softmax is applied **independently at each $(x,\tau)$** position — $N \times T$ independent distributions over the vocabulary (and independently for each batch item $b$). Only after this map does $L_{xv}(\tau)$ acquire probabilistic meaning: $p_{xv}(\tau)$ is the model's predicted probability that position $x$ at slice $\tau$ takes on vocabulary token $v$, and it is this $p_{xv}(\tau)$ — not $L_{xv}(\tau)$ — that is used for sampling, argmax decoding, or evaluation against a target distribution.

**§6: not on the training path.** Softmax (`vocab_softmax` / `to_probabilities` / `forward_pass_to_probabilities`) is diagnostic. Training returns logits from `forward` and feeds them to `F.cross_entropy`, which applies log-softmax internally. Step 12 (`one_hot_targets`) is deleted from the code; the boxed one-hot form of $\mathcal{L}$ below is the theory counterpart of integer-target CE.

---

## 6. The $t$-Index Shift: Connecting $X_{\text{output}}$ to Training Targets

This is the piece that links the readout above to an actual trainable loss, following the source PDF's training-data construction.

### 6.1 Training Window Setup

Given a video described sentence-by-sentence, with temporal dimension size $T$ (e.g. $T=3$):

$$
\text{Training input: } X = \{S(0), S(1), S(2)\} \in \mathbb{R}^{N \times d_x \times 3 \times d_t}
\quad\text{(theory)},
\quad\text{stored }(B, N, 3, d_x, d_t)
$$

The attention update (Steps 6–8 above) is interpreted as **estimating the next slice** at each position in the window:

$$
\hat X(\tau) = X(\tau - 1) + \text{Attn}(X(\tau-1)) = \{\hat S(1), \hat S(2), \hat S(3)\}
$$

That is, the model's output at relative slot $\tau$ in the window is trained to approximate $S(\tau+1)$, **not** $S(\tau)$. This is the shift: $\hat X$'s slice at position $\tau$ carries the prediction for time $\tau+1$.

### 6.2 Why $S(t{=}3)$ Must Be in the Training Set

Because the last window slot's prediction target is $S(3)$, and $S(3)$ is not itself part of the $T=3$ input window $\{S(0),S(1),S(2)\}$, the training example must include $S(3)$ as ground truth **without feeding it into the input**. This is what cleanly eliminates circular data leakage: the target for the final predicted slice is drawn from data the model never sees as input in that window.

### 6.3 From Embedding-Level Loss to Cross-Entropy Loss

The source PDF's Eq. (10) originally defines the loss at the **embedding level**:

$$
\mathcal{L}_{\text{embed}} = \big\| \hat X(\tau) - X(\tau) \big\|^2 = \big\| \text{Attn}(X(\tau-1)) - \delta X(\tau) \big\|^2 = \sum_{t=1}^{3} \big\| \hat S(t) - S(t) \big\|^2
$$

This L2 form compares predicted and target **embeddings** directly. Per the current design decision, this is superseded by a **cross-entropy loss over the vocabulary distribution**, using the readout defined in Section 5:

$$
\boxed{\mathcal{L} = -\sum_{\tau=1}^{T} \sum_{x=1}^{N} \sum_{v \in V} y_{xv}(\tau) \,\log p_{xv}(\tau)}
$$

where $y_{xv}(\tau)$ is the one-hot (or smoothed) target distribution for the true token at position $x$, time $\tau$, and $p_{xv}(\tau)$ is the softmax of the logits — the theory form of the loss. **Implementation (§6):** Step 12 does not run. Training holds integer $Y \in \{0,\dots,|V|-1\}^{B \times T \times N}$ and logits $L \in \mathbb{R}^{B \times N \times T \times |V|}$, and computes

$$
\mathcal{L} = \texttt{F.cross\_entropy}\big(\mathrm{reshape}(L,\, BNT \times |V|),\; \mathrm{reshape}(Y.\texttt{permute}(0,2,1),\, BNT)\big)
$$

with `ignore_index` for PAD. Log-softmax lives inside `F.cross_entropy`; `vocab_softmax` is not called. Sum over batch is included. The boxed one-hot sum is recovered (up to the PAD mask) when $y$ is a one-hot of $Y$.

`ignore_index` only omits PAD **targets** from $\mathcal{L}$. It does **not** stop PAD keys from mixing $P$ and $T_{\text{emb}}$ into other positions — that is §7.1, which runs inside `forward` and therefore inside `training_step` with no extra argument. `padding_idx` on $E_{\text{sem}}, E_{\text{temp}}$ only blocks PAD **embedding** gradients; 7.1 is a different path (attention pollution of $W^{Q,K,V}$).

Because logits are stored $(B, N, T, |V|)$ and $Y$ is stored $(B, T, N)$, the permute is mandatory. If $N=T$, dropping it does not change shape and yields a **different** loss; `cross_entropy_loss` therefore asserts `Y.shape == (B, T, N)`.

**Why cross-entropy rather than L2 on embeddings:**
- The final quantity the model is actually judged on is a token-level prediction (argmax / sampling from $p_{xv}(\tau)$), not a raw embedding vector — cross-entropy directly optimizes the distribution that generation uses.
- L2 on embeddings implicitly assumes Euclidean structure in embedding space is the right notion of "closeness" between predicted and true semantic states; cross-entropy makes no such geometric assumption and is the standard choice for next-token-style objectives.
- This also makes the loss consistent with the readout machinery in Section 5, which was derived precisely to produce a proper probability distribution over $V$.

### 6.4 Teacher Forcing: Training Loss Is Strictly 1-Step

Every window used for computing $\mathcal{L}$ consists **entirely of ground-truth slices** — $\{S(0), S(1), S(2)\}$ in, $\{S(1),S(2),S(3)\}$ compared against at the target side. Predicted values $\hat S(\tau)$ are **never** fed back into a window during training. $W^Q, W^K, W^V, W_1, b_1, W_2, b_2, W'$, and `final_norm`'s $\gamma,\beta$ are all optimized against this single, teacher-forced, 1-step objective. Dropout is not a parameter; `eval()` turns it off.

### 6.5 Multi-Step Rollout Loss ($\mathcal{L}_{\text{eval}}$) Is Diagnostic Only

At inference, sliding the window forward reuses previously predicted slices as input (e.g. $X'' = \{S(2), S(3), \hat S(4)\}$), which introduces compounding error (exposure bias) since $\hat S(4)$ enters both as a residual anchor and as a key/value in subsequent attention. The resulting multi-step error

$$
\mathcal{L}_{\text{eval}} = \|\hat S(4) - S(4)\|^2 + \|\hat S(5) - S(5)\|^2 + \|\hat S(6) - S(6)\|^2
$$

is computed **only after training**, with weights held fixed and no gradient flow. It is never used to update any parameters — this keeps the training objective single-valued (only the clean 1-step $\mathcal{L}$ drives gradients) and avoids introducing auxiliary training machinery (e.g. scheduled sampling) to offset exposure bias.

---

## 7. Summary Table: Fiber Roles Through the Full Pipeline

| Stage | Tensor | Shape | Free indices in $\mathcal{L}$? |
|---|---|---|---|
| Input | $X_{xn;\tau q}$ | $(B, N, T, d_x, d_t)$ | — |
| Post-attention residual | $X^{\text{attn}}_{xn;\tau q}$ | $(B, N, T, d_x, d_t)$ | — (dropout on `Attn`, §9) |
| Post-MLP residual | $X_{\text{output},\,xn;\tau q}$ | $(B, N, T, d_x, d_t)$ | — (dropout on $z$, same $p$) |
| Readout-boundary LN | $X^{\text{norm}}_{xn;\tau q}$ | $(B, N, T, d_x, d_t)$ | — (`final_norm`, §8) |
| Logits | $L_{xv}(\tau)$ | $(B, N, T, |V|)$ | $x, \tau$ (and $v$, the output axis) |
| Targets $Y$ | token ids | $(B, T, N)$ | $x, \tau$ — axis order **differs** from logits |
| Probabilities (diagnostic) | $p_{xv}(\tau)$ | $(B, N, T, |V|)$ | $x, \tau$ |
| Loss | $\mathcal{L}$ | scalar | (summed out, including $B$) |

$(n,q)$ exist only up through $X^{\text{norm}}$; they are eliminated at the readout contraction (Section 5.2) and never appear again downstream.

---

## 9. Pseudo Code: Target Construction and Cross-Entropy Loss (Steps 11–13)

This section makes the $+1$ shift of Section 6.1 concrete: $X_{\text{input}} = \{S(0), S(1), S(2)\}$ produces logits at $\tau=0,1,2$, compared against $\{S(1), S(2), S(3)\}$ — **not** against $\{S(0), S(1), S(2)\}$.

§6: every tensor here has a leading $B$. Training consumes **logits** and integer $Y$; it does not call softmax or build a one-hot. Logits are $(B, N, T, |V|)$ and $Y$ is $(B, T, N)$ — `permute(0, 2, 1)` is the bridge.

First, the readout (Step 9) is the §2 path-B contraction on the §3 layout: $M$ then $X_{\text{flat}} M$, with `reshape(B*N*T, d_x*d_t)` and no permute.

```python
# ============================================================
# Step 9 (revised): Vocabulary readout — tied-n / learned-q factorization
#   Path B: M_{nq;v} = E_sem[v,n] * W'_{q,v}, then L = X_flat @ M
#   X_norm   : (B, N, T, d_x, d_t)  after §8 final_norm
#   ->  logits : (B, N, T, |V|)
# ============================================================

def readout_logits_tied(X_output, E_sem, W_prime, N, d_x, d_t, T, vocab_size):
    """
    X_output : (B, N, T, d_x, d_t)
    E_sem    : (|V|, d_x)     -- SAME table as Stage A
    W_prime  : (d_t, |V|)
    returns  : (B, N, T, |V|)
    """
    if X_output.ndim != 5:
        raise ValueError("X_output must be (B, N, T, d_x, d_t)")
    B, N_in, T_in, d_x_in, d_t_in = X_output.shape
    if (d_x_in, d_t_in) != (d_x, d_t):
        raise ValueError("last two axes must be (d_x, d_t); product equality is not a check")
    # no permute — N,T already adjacent (§3)

    d = d_x * d_t
    M = zeros(shape=(d, vocab_size))
    for n in range(d_x):
        for q in range(d_t):
            for v in range(vocab_size):
                M[n * d_t + q, v] = E_sem[v, n] * W_prime[q, v]

    Logits = zeros(shape=(B, N, T, vocab_size))
    for b in range(B):
        for x in range(N):
            for t in range(T):
                for v in range(vocab_size):
                    s = 0.0
                    for n in range(d_x):
                        for q in range(d_t):
                            s += X_output[b, x, t, n, q] * M[n * d_t + q, v]
                    Logits[b, x, t, v] = s
    return Logits


def forward(token_ids, W_Q, W_K, W_V, W1, b1, W2, b2, E_sem, E_temp, W_prime,
            N, d_x, d_t, T, vocab_size, padding_idx=0, drop=None, ln=None):
    """token_ids : (B, T, N)  ->  logits : (B, N, T, |V|)
    pad_mask once per call (§7.1). Residual dropout on Attn and MLP (§9).
    final_norm after the last layer, before readout (§8)."""
    if token_ids.ndim != 3:
        raise ValueError("token_ids must be (B, T, N)")
    pad_mask = build_pad_mask(token_ids, padding_idx)     # (B, N, T)
    X = build_input_tensor(token_ids, N, d_x, d_t, T, E_sem, E_temp)
    X = transformer_layer(
        X, W_Q, W_K, W_V, W1, b1, W2, b2, N, d_x, d_t, T, pad_mask, drop
    )
    # multi-layer: for layer in layers: X = layer(X, pad_mask)  -- same drop p
    # live layer: Attn(LN(X)), MLP(LN(X)); residual adds still on raw X (§4.2)
    if ln is not None:
        X = final_norm(X, ln, N, d_x, d_t, T)
    return readout_logits_tied(X, E_sem, W_prime, N, d_x, d_t, T, vocab_size)


# FourTensorModel holds:
#   self.layers[i]  : Pre-LN Attn/MLP + shared Dropout(p) on both residual branches
#   self.final_norm : nn.LayerNorm(d_x * d_t)   -- once, after the last layer (§8)
# dropout=0.0 is the constructor default (train script); ROCStories shell uses 0.1.
# Pre-LN backup: four_tensor_model0921.py


def forward_pass_to_probabilities(token_ids, W_Q, W_K, W_V, W1, b1, W2, b2, E_sem, E_temp, W_prime,
                                   N, d_x, d_t, T, vocab_size, padding_idx=0):
    """Diagnostic only. Training does not call this."""
    Logits = forward(token_ids, W_Q, W_K, W_V, W1, b1, W2, b2, E_sem, E_temp, W_prime,
                     N, d_x, d_t, T, vocab_size, padding_idx)
    P = vocab_softmax(Logits, N, T, vocab_size)
    return Logits, P
```

**Note on gradients through $E_{\text{sem}}$.** Because $E_{\text{sem}}$ is now used in two places — Stage A of the input pipeline (Appendix A) *and* the readout above — gradient descent on $\mathcal{L}$ accumulates contributions to $E_{\text{sem}}$ from **both** paths (embedding-input path and readout path), exactly as in standard input/output weight tying.

```python
# ============================================================
# Step 11: Target construction — shifted alignment
#   token_ids_full : (B, T+1, N)  ->  Y : (B, T, N)
#   Y[b, τ, x] = token of S(τ+1). Unbatched (T+1, N) is a ValueError.
# ============================================================

def build_targets(token_ids_full):
    """token_ids_full : (B, T+1, N)  ->  Y : (B, T, N)"""
    if token_ids_full.ndim != 3:
        raise ValueError("token_ids_full must be (B, T+1, N)")
    return token_ids_full[:, 1:, :]     # shift by +1 along τ


# ============================================================
# Step 12: one_hot_targets — DELETED from the training path.
#   Theory counterpart of integer-target CE (boxed L in §6.3).
#   Do not materialize y_xv; F.cross_entropy takes class indices.
# ============================================================


# ============================================================
# Step 13: Cross-entropy loss
#   logits : (B, N, T, |V|)
#   Y      : (B, T, N)     -- different axis order
#   permute(0, 2, 1) is the bridge. If N=T, a missing permute
#   still has the right shape and a WRONG loss — hence the guard.
# ============================================================

def cross_entropy_loss(logits, Y, pad_id=0):
    """
    logits : (B, N, T, V)
    Y      : (B, T, N)
    PAD targets excluded by ignore_index. This is *not* a substitute for
    §7.1 key masking: padding_idx blocks PAD *embedding* gradients;
    key masking blocks PAD from polluting other positions' attention.
    """
    if logits.ndim != 4:
        raise ValueError("logits must be (B, N, T, V)")
    if Y.ndim != 3:
        raise ValueError("Y must be (B, T, N)")
    B, N, T, _ = logits.shape
    if Y.shape != (B, T, N):
        raise ValueError(
            f"Y must be (B, T, N)=({B}, {T}, {N}) to match logits (B, N, T, V), "
            f"got {tuple(Y.shape)}"
        )
    tgt = Y.permute(0, 2, 1).reshape(-1)          # (B*N*T,)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt, ignore_index=pad_id,
    )


# ============================================================
# Top-level: token_ids_full (B, T+1, N) -> loss, logits
#   token_ids = {S(0),...,S(T-1)}  ->  logits vs  Y={S(1),...,S(T)}
# ============================================================

def training_step(token_ids_full, model, pad_id=0):
    """
    token_ids_full : (B, T+1, N)
    returns (loss, logits) with logits (B, N, T, |V|)
    DataLoader yields this rank even when B=1.
    """
    if token_ids_full.ndim != 3:
        raise ValueError("token_ids_full must be (B, T+1, N)")
    token_ids = token_ids_full[:, :-1, :]         # (B, T, N)
    logits = model.forward(token_ids)             # (B, N, T, |V|)
    Y = build_targets(token_ids_full)             # (B, T, N)
    loss = cross_entropy_loss(logits, Y, pad_id=pad_id)
    return loss, logits


# ============================================================
# §7.2 generate_logits — diagnostic warning, not a logit mask
#   E_sem[pad]=0 fixes L[..., pad]=0. PAD wins argmax only if every
#   real-class logit is < 0, or if W' collapses to 0 (ties break to
#   index 0). Measured rate in the normal range is ~0%; always-on
#   readout masking is therefore not applied. Warn on PAD-argmax.
# ============================================================

def generate_logits(model, token_ids, temperature=1.0, padding_idx=0):
    """Generation path only. Does not share tensors with training_step.
    Returns last-slot logits (B, N, |V|). Values are not rewritten."""
    with no_grad():
        logits = model.forward(token_ids)[:, :, -1, :] / temperature
        argmax = logits.argmax(dim=-1)
        pad_hit = argmax == padding_idx
        if pad_hit.any():
            w_collapsed = model.readout.W_prime.abs().max() < 1e-6
            warn(
                f"argmax landed on PAD at {pad_hit.sum()}/{pad_hit.numel()} "
                f"positions; W' collapsed={w_collapsed}"
            )
    return logits
```

**Key point**: `build_targets`'s `+1` shift is the confirmed structure — window-relative $\tau \in \{0,1,2\}$ is compared against $S(1), S(2), S(3)$. `token_ids_full` is $(B, T+1, N)$. Softmax and one-hot are not on this path. $Y.\texttt{permute}(0,2,1)$ aligns $(B, T, N)$ with logits $(B, N, T, |V|)$. `training_step` calls `forward`, which already applies §7.1 key masking, §9 residual dropout, per-layer Pre-LN (Section 4.2), and §8 `final_norm`; no extra PAD / dropout / LN argument is required.

**§7.1 vs §7.2.** Attention-key masking is mandatory on every train/eval `forward` (without it, PAD $K,V$ always mix in). Readout PAD-argmax is a rare generation degeneracy ($E_{\text{sem}}[\text{pad}]=0$ fixes $L_{\cdot,\text{pad}}=0$; it wins only if every real logit is $<0$, or $W'=0$ and `argmax` ties to index 0). `generate_logits` warns and does **not** rewrite logits. An optional `readout_bias` could push $b_{\text{pad}}$ negative later; it is not required now.

---

## 10. Open Items Carried Forward

- **MLP joint-flatten design** (Section 3) is a generalization beyond the source PDF (which assumed $d_x = d_t = d$ implicitly); confirmed as the intended "fully unfolded" approach — no separate per-fiber MLPs.
- **$d_{ff} = 4\, d_x d_t$** is the scaling rule (§1): $d_{\text{in}} = d_x d_t$, so this is the standard $4\times d_{\text{model}}$ convention. The earlier $4\sqrt{d_x d_t}$ geometric-mean rule is withdrawn — it does not preserve $d_{ff}/d_{\text{in}}$, and the $\sqrt{\cdot}$ in the attention scale $1/\sqrt{d_x d_t}$ is not an analogy for MLP width.
- **Cross-entropy vs. L2** (Section 6.3): this document reflects the decision to use cross-entropy as the actual training signal, superseding the embedding-level L2 form in the source PDF's Eq. (10) — resolving memory open item 4.
- **Self-consistency check** (re-predicting known deltas at inference, memory open item 3) is referenced structurally in Section 6.5 but not yet formalized as a computable diagnostic here.
- **Tied-$n$ / learned-$q$ readout factorization** (Section 5.3.1, Step 9 pseudocode) reduces $W_{\text{out}}$ from a dense $d_x \times d_t \times |V|$ tensor to a zero-new-parameter tie against $E_{\text{sem}}$ plus a learned $d_t \times |V|$ projection $W'$ — resolving memory open item on parameter reduction paths, specifically for the readout stage (independent of the $6d^2$ / CP-decomposition reduction discussed for QKV). **§2 evaluation order:** this factorization is computed as path B ($M = E_{\text{sem}} \otimes W'$ materialized first, then $X_{\text{flat}} M$), not path A (sequence-dependent $Y_{x\tau qv}$). Vocabulary chunking of $Y$ is rejected: autograd retains every chunk on the training path.
- **§3 / §4 storage.** Theory indices $X_{xn;\tau q}$, $W_{mn,pq}$ are unchanged. Activations are stored $(B, N, T, d_x, d_t)$ and weights $(m, p, n, q)$, so `reshape`/`view` need no permute. Old $W$ checkpoints convert by `permute(0, 2, 1, 3)` on $W^{Q,K,V}$. Verify with $d_x, d_t, T, N$ all distinct.
- **§6 batch + CE.** Public APIs are `(B, ...)` only; there is no `_as_batched`. Training uses integer $Y$ of shape $(B, T, N)$ and logits $(B, N, T, |V|)$ via `F.cross_entropy`; `Y.permute(0, 2, 1)` is mandatory (`N=T` does not catch a missing permute). `one_hot_targets` is deleted. Softmax is diagnostic. An unbatched `(T+1, N)` is a `ValueError`.
- **§7 PAD.** $X_{\text{pad}}=P+T_{\text{emb}}\neq 0$, so standard "PAD embedding is zero" masking does not apply. **7.1** builds `pad_mask` from `token_ids` (shape $(B,N,T)$), fills **key** scores with `finfo.min` (not $\texttt{-inf}$), reuses the mask across depth layers of one `forward`, and recomputes it every call. MLP is unmasked. **7.2** is a generation-path PAD-argmax warning; logits are not rewritten. `ignore_index` and `padding_idx` do not replace 7.1.
- **§8 readout-boundary LayerNorm.** One `nn.LayerNorm(d_x d_t)` after the last layer, before readout, on flatten $\phi(n,q)=n d_t+q$. Logit scale. Orthogonal to §7.1; PAD logit stays $0$. Does not bound $\|M\|$, affine $\gamma$, intra-layer scores, or $\sqrt{H}$ head scale. Still required after Pre-LN because the residual stream is unnormalized. `training_step` / `generate_logits` signatures unchanged.
- **Pre-LN (Section 4.2).** Live code: $X \leftarrow X+\mathrm{dropout}(\mathrm{Attn}(\mathrm{LN}(X)))$, then $X \leftarrow X+\mathrm{dropout}(\mathrm{MLP}(\mathrm{LN}(X)))$. Same $\phi$ as §8. Two LNs per layer; not a substitute for §8 or $\sqrt{H}$. $W^{Q,K,V}$ act on $\mathrm{LN}(X)$. Backup: `four_tensor_model0921.py`.
- **§9 residual dropout.** Same $p$ on both residual branches: $X+\mathrm{dropout}(\text{Attn})$ and $X+\mathrm{dropout}(\text{MLP})$. Not on softmax, embeddings, `final_norm`, Pre-LN, or readout. Recommended $p=0.1$ (train-script default $0$; ROCStories shell $0.1$). Does not replace §8, Pre-LN, AdamW weight decay, or early-stop. Not attention-weight dropout.



# PSF 4-Tensor Attention: Original Step-9 (Old version: $W_{out}$ 3-tensor full training version) to Step-10

Historical dense $W_{\text{out}}$ readout and diagnostic softmax. Training uses the tied readout of Section 9 and does **not** call `vocab_softmax`. Softmax, when used, is `dim=-1` on logits of shape $(B, N, T, |V|)$.

```python
# ============================================================
# Step 9: Vocabulary readout — logits L_xv(t)
#   Joint (bilinear) contraction of (n, q), no separability assumed.
#   Free indices of the loss are (x, t); (n, q) must be eliminated.
# ============================================================

def readout_logits(X_output, W_out, N, d_x, d_t, T, vocab_size):
    """
    X_output : shape (N, T, d_x, d_t)      == X_hat_{xn;tq}, stored [x, τ, n, q]
    W_out    : shape (d_x, d_t, vocab_size) == W_{out, nq; v}
               a LEARNED parameter, trained jointly with W_Q, W_K, W_V, and
               the MLP weights via gradient descent on L. No principled way
               to fix it by hand -- it is optimized jointly with everything
               else, exactly like the other weight tensors.

    returns:
        Logits : shape (N, T, vocab_size)  == L_{xv}(t)
                 NOT yet a probability distribution.
    """
    Logits = zeros(shape=(N, T, vocab_size))

    for x in range(N):
        for t in range(T):
            for v in range(vocab_size):
                s = 0.0
                for n in range(d_x):
                    for q in range(d_t):
                        s += X_output[x, t, n, q] * W_out[n, q, v]
                Logits[x, t, v] = s

    return Logits   # L_{xv}(t), shape (N, T, |V|)


# ============================================================
# Step 10: Softmax over vocabulary axis -> probability distribution
#   Diagnostic only (§6). Training uses F.cross_entropy on logits.
#   softmax_v along dim=-1, independently at each (b, x, t).
#   Logits : (B, N, T, |V|)
# ============================================================

def vocab_softmax(Logits, N, T, vocab_size):
    """
    Logits : shape (N, T, vocab_size)  == L_{xv}(t)

    returns:
        P : shape (N, T, vocab_size)  == p_{xv}(t) ∈ Δ^{|V|-1}
            N*T independent probability distributions over the vocabulary,
            one for each (token position x, temporal slot t).
    """
    P = zeros(shape=(N, T, vocab_size))

    for x in range(N):
        for t in range(T):
            # numerically stable softmax over v
            max_val = -inf
            for v in range(vocab_size):
                if Logits[x, t, v] > max_val:
                    max_val = Logits[x, t, v]

            denom = 0.0
            for v in range(vocab_size):
                denom += exp(Logits[x, t, v] - max_val)

            for v in range(vocab_size):
                P[x, t, v] = exp(Logits[x, t, v] - max_val) / denom

    return P   # p_{xv}(t), shape (N, T, |V|), used for cross-entropy loss
               # against target distribution, sampling, or argmax decoding


# ============================================================
# Top-level: single-layer transformer -> logits -> softmax probabilities
# ============================================================

def forward_pass_to_probabilities(X, W_Q, W_K, W_V, W1, b1, W2, b2, W_out,
                                   N, d_x, d_t, T, vocab_size):
    X_output = transformer_layer(X, W_Q, W_K, W_V, W1, b1, W2, b2, N, d_x, d_t, T)
    Logits   = readout_logits(X_output, W_out, N, d_x, d_t, T, vocab_size)
    P        = vocab_softmax(Logits, N, T, vocab_size)

    return Logits, P   # Logits for cross-entropy loss, P for interpretation/sampling
```




# PSF 4-Tensor Attention: Step-11 to Step-13

Superseded by §6 (Section 9 above). Training uses `build_targets` → integer $Y$ of shape $(B, T, N)$ and `F.cross_entropy` on logits $(B, N, T, |V|)$. `one_hot_targets` is deleted from the training path.

---

# PSF Rank-4 Model: Backward Pass and $W^Q, W^K, W^V$ Update Rules

This document derives the backward pass through the attention block — softmax Jacobian → $\Delta S$ → $\Delta Q, \Delta K, \Delta V$ → weight-tensor gradients — and states the resulting gradient-descent update rule for $W^Q, W^K, W^V \in \mathbb{R}^{d_x \times d_x \times d_t \times d_t}$.

**Notation note**: the derivation below uses $(x,\tau)$ for query position/slot and $(x',\tau')$ for key/value position/slot, matching the $X_{xn;\tau q}$ convention established in prior documents. The scale factor is written as $1/\sqrt{d_x d_t}$ throughout (generalizing the single-$d$ form $1/d$ used when $d_x=d_t=d$), consistent with the score-tensor definition already fixed in the forward-pass documents.

---

## 1. Structural Observation

The backward pass through this rank-4 attention block has the **same structure as standard scaled-dot-product attention**, with $(x,\tau)$ playing the role of the usual single sequence index, and the $(n,q)$/$(m,p)$ fibers playing the role of the usual feature dimension. Nothing about the joint-softmax or the doubled-Frobenius score changes the shape of the derivation — only the index count doubles (each "position" is now a pair $(x,\tau)$, each "feature" a pair $(m,p)$).

---

## 2. Joint Softmax Jacobian

Given $A_{y u,\, y'u'} = \text{softmax}$ of $S_{yu,\,\tilde y \tilde u}$ jointly over $(\tilde y, \tilde u)$:

$$
\frac{\partial A_{yu,\,y'u'}}{\partial S_{xt,\,x't'}} = \delta_{xy}\delta_{tu}\, A_{xt,\,y'u'}\left(\delta_{x'y'}\delta_{t'u'} - A_{xt,\,x't'}\right)
$$

This is exactly the standard softmax Jacobian (diagonal-minus-outer-product form), applied over the joint index pair. The $\delta_{xy}\delta_{tu}$ factor reflects that $A$'s dependence on $S$ is only through the row $S_{xt,\,\cdot}$ that shares the same query index $(x,t)=(y,u)$ — rows of the joint-softmax are independent, exactly as in ordinary softmax.

---

## 3. $\Delta S_{xt,\,x't'} \equiv \partial \mathcal{L}/\partial S_{xt,\,x't'}$

Substituting the Jacobian into the chain rule $\dfrac{\partial \mathcal{L}}{\partial S_{xt,x't'}} = \sum_{yu,y'u'} \dfrac{\partial \mathcal{L}}{\partial A_{yu,y'u'}} \dfrac{\partial A_{yu,y'u'}}{\partial S_{xt,x't'}}$, and using the $\delta_{xy}\delta_{tu}$ to collapse the outer sum over $(y,u)$:

$$
\Delta S_{xt,\,x't'} = A_{xt,\,x't'} \left( \frac{\partial \mathcal{L}}{\partial A_{xt,\,x't'}} - \sum_{y'u'} \frac{\partial \mathcal{L}}{\partial A_{xt,\,y'u'}}\, A_{xt,\,y'u'} \right)
$$

This is the row-wise softmax-gradient identity: for each query $(x,t)$, subtract the $A$-weighted average of the incoming gradient across the key/value grid, then scale by $A$ itself. $\partial \mathcal{L}/\partial A_{xt,x't'}$ itself comes from further downstream (Section 5).

---

## 4. Gradient w.r.t. $Q$, $K$, $V$

### 4.1 Score Tensor Recap

$$
S_{xt,\,x't'} = \frac{1}{\sqrt{d_x d_t}} \sum_{m,p} Q_{xm;tp}\, K_{x'm;t'p}
$$

### 4.2 $\partial \mathcal{L} / \partial Q$

$$
\frac{\partial S_{xt,\,x't'}}{\partial Q_{zm;wp}} = \frac{1}{\sqrt{d_x d_t}}\, \delta_{xz}\delta_{tw}\, K_{x'm;t'p}
$$

$$
\Delta Q_{zm;wp} \equiv \frac{\partial \mathcal{L}}{\partial Q_{zm;wp}} = \sum_{xt,\,x't'} \Delta S_{xt,\,x't'} \cdot \frac{1}{\sqrt{d_x d_t}}\, \delta_{xz}\delta_{tw}\, K_{x'm;t'p} = \frac{1}{\sqrt{d_x d_t}} \sum_{x',t'} \Delta S_{zw,\,x't'}\, K_{x'm;t'p}
$$

In slice-matrix notation (each $(x,\tau)$ position holding a $d_x \times d_t$ matrix):

$$
\boxed{\frac{\partial \mathcal{L}}{\partial Q_{z;w}} = \frac{1}{\sqrt{d_x d_t}} \sum_{x',t'} \Delta S_{zw,\,x't'}\, K_{x';t'}}
$$

### 4.3 $\partial \mathcal{L} / \partial K$

By the symmetric role of $K$ in $S_{xt,x't'} = \frac{1}{\sqrt{d_x d_t}}\sum_{m,p} Q_{xm;tp} K_{x'm;t'p}$ (i.e. $K$ enters through the *second* index pair $(x',t')$ rather than the first):

$$
\frac{\partial S_{xt,\,x't'}}{\partial K_{zm;wp}} = \frac{1}{\sqrt{d_x d_t}}\, \delta_{x'z}\delta_{t'w}\, Q_{xm;tp}
$$

$$
\Delta K_{zm;wp} \equiv \frac{\partial \mathcal{L}}{\partial K_{zm;wp}} = \sum_{xt,\,x't'} \Delta S_{xt,\,x't'} \cdot \frac{1}{\sqrt{d_x d_t}}\, \delta_{x'z}\delta_{t'w}\, Q_{xm;tp} = \frac{1}{\sqrt{d_x d_t}} \sum_{x,t} \Delta S_{xt,\,zw}\, Q_{xm;tp}
$$

$$
\boxed{\frac{\partial \mathcal{L}}{\partial K_{z;w}} = \frac{1}{\sqrt{d_x d_t}} \sum_{x,t} \Delta S_{xt,\,zw}\, Q_{x;t}}
$$

Note the index asymmetry relative to $\Delta Q$: $\Delta Q$ sums $\Delta S$ over its *second* pair with the *first* pair fixed at $(z,w)$; $\Delta K$ sums $\Delta S$ over its *first* pair with the *second* pair fixed at $(z,w)$. This mirrors $Q$/$K$'s asymmetric roles as row/column index of the score matrix.

### 4.4 $\partial \mathcal{L} / \partial V$

$V$ does not appear in $S$ at all — it enters only at the attention-output contraction:

$$
\text{Attn}_{xm;tp} = \sum_{x',t'} A_{xt,\,x't'}\, V_{x'm;t'p}
$$

$$
\frac{\partial \text{Attn}_{xm;tp}}{\partial V_{zm;wp}} = A_{xt,\,zw} \qquad \text{(diagonal in } m,p\text{, since the contraction over } (x',t') \text{ does not mix features)}
$$

Given $\Delta\text{Attn}_{xm;tp} \equiv \partial \mathcal{L}/\partial \text{Attn}_{xm;tp}$ (supplied from downstream — Section 5):

$$
\boxed{\frac{\partial \mathcal{L}}{\partial V_{z;w}} = \sum_{x,t} A_{xt,\,zw}\, \Delta\text{Attn}_{x;t}}
$$

This is the **transpose** of the forward attention-output contraction: whereas the forward pass mixes $V$ into the output using $A$ as weights over $(x',t')$ for each query $(x,t)$, the backward pass mixes $\Delta\text{Attn}$ back into $\Delta V$ using $A$ as weights over $(x,t)$ for each key/value position $(z,w)$ — the same matrix $A$, contracted along the opposite axis.

---

## 5. Where $\Delta\text{Attn}$ and $\partial\mathcal{L}/\partial A$ Come From

$\Delta\text{Attn}_{xm;tp}$ is the gradient arriving at the attention block from downstream — in the single-layer forward pass documented previously, this path is:

$$
\mathcal{L} \;\to\; p_{xv}(\tau) \;\to\; L_{xv}(\tau) \;\to\; X_{\text{output}} \;\to\; z_{xn;\tau q} \text{ (MLP)} \;\to\; X^{\text{attn}}_{xn;\tau q} \;\to\; \text{Attn}_{xn;\tau q}
$$

Because of the **residual connections** at both the post-attention and post-MLP stages ($X^{\text{attn}} = X + \text{Attn}$, $X_{\text{output}} = X^{\text{attn}} + z$), gradient reaches $\Delta\text{Attn}$ by the identity path *and* the MLP path simultaneously — this is the standard residual-network property that gradients skip-connect. Deriving this full chain (cross-entropy → softmax → logits → MLP backward → residual split) is a separate, subsequent step; here $\Delta\text{Attn}_{xm;tp}$ (equivalently $\partial\mathcal{L}/\partial A_{xt,x't'}$, related by the $V$-contraction above) is treated as a **given upstream input** to this block's backward pass, exactly as $K,Q,V$ were treated as given inputs to the forward pass of this block.

---

## 6. Chain Rule to the Weight Tensors

Recall the forward definitions (rank-4 bilinear contraction, no shortcuts):

$$
Q_{xm;tp} = \sum_{n,q} W^Q_{mn,pq}\, X_{xn;tq}, \qquad
K_{xm;tp} = \sum_{n,q} W^K_{mn,pq}\, X_{xn;tq}, \qquad
V_{xm;tp} = \sum_{n,q} W^V_{mn,pq}\, X_{xn;tq}
$$

Each weight tensor's gradient follows by the same pattern — differentiate the bilinear contraction, which simply strips out the $X$ factor:

$$
\frac{\partial Q_{xm;tp}}{\partial W^Q_{\tilde m\tilde n,\tilde p\tilde q}} = \delta_{m\tilde m}\delta_{p\tilde p}\, X_{x\tilde n;t\tilde q}
$$

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W^Q_{mn,pq}} = \sum_{x,t} \Delta Q_{xm;tp}\, X_{xn;tq}}
$$

and identically in structure for $K$ and $V$:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial W^K_{mn,pq}} = \sum_{x,t} \Delta K_{xm;tp}\, X_{xn;tq}} \qquad\qquad
\boxed{\frac{\partial \mathcal{L}}{\partial W^V_{mn,pq}} = \sum_{x,t} \Delta V_{xm;tp}\, X_{xn;tq}}
$$

Each is an **outer product between the upstream gradient (in the $(m,p)$-slice space) and the input $X$ (in the $(n,q)$-slice space), summed over all sequence/window positions $(x,t)$** — exactly the familiar $\partial L/\partial W = \sum (\text{grad-out}) \otimes (\text{input})$ structure of a linear layer, generalized to rank-4.

**Storage (§4).** The boxed formulae are in theory order $W_{mn,pq}$. `W.grad` is stored $(m,p,n,q)$, so a comparison against a hand derivation in $(m,n,p,q)$ order is `W.grad.permute(0, 2, 1, 3)`. Xavier and `F.linear` see the flat $(mp, nq)$ matrix; that flattening is the same $\phi$ as $X$'s last two axes.

---

## 7. Gradient-Descent Update Rule

With learning rate $\eta$:

$$
W^Q_{mn,pq} \;\leftarrow\; W^Q_{mn,pq} - \eta\, \frac{\partial \mathcal{L}}{\partial W^Q_{mn,pq}}
$$
$$
W^K_{mn,pq} \;\leftarrow\; W^K_{mn,pq} - \eta\, \frac{\partial \mathcal{L}}{\partial W^K_{mn,pq}}
$$
$$
W^V_{mn,pq} \;\leftarrow\; W^V_{mn,pq} - \eta\, \frac{\partial \mathcal{L}}{\partial W^V_{mn,pq}}
$$

All three updates are computed from **the same $\Delta S$** (Section 3) via the $\Delta Q, \Delta K$ paths, plus the independently-derived $\Delta V$ path (Section 4.4) — consistent with training being driven by the single, teacher-forced, 1-step cross-entropy loss $\mathcal{L}$ established in the loss-function document; no separate objective touches $W^Q, W^K, W^V$.

---

## 8. Pseudo Code

```python
# ============================================================
# Backward pass: attention block
#   Given: A (joint softmax), Q, K, V, X (forward-pass cache)
#          dL_dAttn : upstream gradient, shape (N, T, d_x, d_t)
#   Produces: dL_dW_Q, dL_dW_K, dL_dW_V
# ============================================================

def attention_backward(A, Q, K, V, X, dL_dAttn, N, d_x, d_t, T):
    """
    A        : shape (N, T, N, T)          == A_{xt,x't'}, cached from forward pass
    Q, K, V  : shape (N, T, d_x, d_t)       cached from forward pass
    X        : shape (N, T, d_x, d_t)       block input, cached from forward pass
    dL_dAttn : shape (N, T, d_x, d_t)       == ΔAttn_{xm;tp}, from downstream (Section 5)

    returns:
        dL_dS   : shape (N, T, N, T)         == ΔS_{xt,x't'}
        dL_dQ   : shape (N, T, d_x, d_t)     == ΔQ_{zm;wp}
        dL_dK   : shape (N, T, d_x, d_t)     == ΔK_{zm;wp}
        dL_dV   : shape (N, T, d_x, d_t)     == ΔV_{zm;wp}
        dL_dWQ, dL_dWK, dL_dWV : shape (d_x, d_t, d_x, d_t)  # stored (m, p, n, q)
    """
    scale = 1.0 / sqrt(d_x * d_t)

    # -------------------------------------------------------
    # Step B1: dL/dA_{xt,x't'}  from  dL/dAttn via V-contraction (transpose)
    #   Attn_{xm;tp} = sum_{x',t'} A_{xt,x't'} V_{x'm;t'p}
    #   => dL/dA_{xt,x't'} = sum_{m,p} dL_dAttn_{xm;tp} * V_{x'm;t'p}
    # -------------------------------------------------------
    dL_dA = zeros(shape=(N, T, N, T))
    for x in range(N):
        for t in range(T):
            for xp in range(N):
                for tp_ in range(T):
                    s = 0.0
                    for m in range(d_x):
                        for p in range(d_t):
                            s += dL_dAttn[x, t, m, p] * V[xp, tp_, m, p]
                    dL_dA[x, t, xp, tp_] = s

    # -------------------------------------------------------
    # Step B2: dL/dV  (transpose of forward V-contraction, weighted by A)
    #   dL/dV_{z;w} = sum_{x,t} A_{xt,zw} * dL_dAttn_{x;t}
    # -------------------------------------------------------
    dL_dV = zeros(shape=(N, T, d_x, d_t))
    for z in range(N):
        for w in range(T):
            for m in range(d_x):
                for p in range(d_t):
                    s = 0.0
                    for x in range(N):
                        for t in range(T):
                            s += A[x, t, z, w] * dL_dAttn[x, t, m, p]
                    dL_dV[z, w, m, p] = s

    # -------------------------------------------------------
    # Step B3: dL/dS  from  dL/dA via joint-softmax Jacobian (Section 2-3)
    #   ΔS_{xt,x't'} = A_{xt,x't'} * ( dL/dA_{xt,x't'}
    #                    - sum_{y'u'} dL/dA_{xt,y'u'} * A_{xt,y'u'} )
    # -------------------------------------------------------
    dL_dS = zeros(shape=(N, T, N, T))
    for x in range(N):
        for t in range(T):
            # row-wise weighted average of incoming gradient
            weighted_avg = 0.0
            for yp in range(N):
                for up in range(T):
                    weighted_avg += dL_dA[x, t, yp, up] * A[x, t, yp, up]

            for xp in range(N):
                for tp_ in range(T):
                    dL_dS[x, t, xp, tp_] = A[x, t, xp, tp_] * (
                        dL_dA[x, t, xp, tp_] - weighted_avg
                    )

    # -------------------------------------------------------
    # Step B4: dL/dQ  (Section 4.2)
    #   ΔQ_{z;w} = scale * sum_{x',t'} ΔS_{zw,x't'} * K_{x';t'}
    # -------------------------------------------------------
    dL_dQ = zeros(shape=(N, T, d_x, d_t))
    for z in range(N):
        for w in range(T):
            for m in range(d_x):
                for p in range(d_t):
                    s = 0.0
                    for xp in range(N):
                        for tp_ in range(T):
                            s += dL_dS[z, w, xp, tp_] * K[xp, tp_, m, p]
                    dL_dQ[z, w, m, p] = scale * s

    # -------------------------------------------------------
    # Step B5: dL/dK  (Section 4.3)
    #   ΔK_{z;w} = scale * sum_{x,t} ΔS_{xt,zw} * Q_{x;t}
    # -------------------------------------------------------
    dL_dK = zeros(shape=(N, T, d_x, d_t))
    for z in range(N):
        for w in range(T):
            for m in range(d_x):
                for p in range(d_t):
                    s = 0.0
                    for x in range(N):
                        for t in range(T):
                            s += dL_dS[x, t, z, w] * Q[x, t, m, p]
                    dL_dK[z, w, m, p] = scale * s

    # -------------------------------------------------------
    # Step B6: dL/dW_Q, dL/dW_K, dL/dW_V  (Section 6)
    #   dL/dW_{mn,pq} = sum_{x,t} ΔX_{xm;tp} * X_{xn;tq}   for X in {Q,K,V}
    # -------------------------------------------------------
    def weight_grad(dL_dXqkv):
        # stored (m, p, n, q). Theory ∂L/∂W_{mn,pq}.
        # To compare against a hand derivation in (m,n,p,q) order:
        #   grad.permute(0, 2, 1, 3)
        grad = zeros(shape=(d_x, d_t, d_x, d_t))
        for m in range(d_x):
            for p in range(d_t):
                for n in range(d_x):
                    for q in range(d_t):
                        s = 0.0
                        for x in range(N):
                            for t in range(T):
                                s += dL_dXqkv[x, t, m, p] * X[x, t, n, q]
                        grad[m, p, n, q] = s
        return grad

    dL_dWQ = weight_grad(dL_dQ)
    dL_dWK = weight_grad(dL_dK)
    dL_dWV = weight_grad(dL_dV)

    return dL_dS, dL_dQ, dL_dK, dL_dV, dL_dWQ, dL_dWK, dL_dWV


# ============================================================
# Gradient-descent update (Section 7)
# ============================================================

def update_weights(W_Q, W_K, W_V, dL_dWQ, dL_dWK, dL_dWV, eta, d_x, d_t):
    for m in range(d_x):
        for p in range(d_t):
            for n in range(d_x):
                for q in range(d_t):
                    W_Q[m, p, n, q] -= eta * dL_dWQ[m, p, n, q]
                    W_K[m, p, n, q] -= eta * dL_dWK[m, p, n, q]
                    W_V[m, p, n, q] -= eta * dL_dWV[m, p, n, q]
    return W_Q, W_K, W_V
```

---

## 9. Open Items Carried Forward

- **$\Delta\text{Attn}$'s upstream derivation** (cross-entropy → softmax → logits → MLP backward → residual split into the attention path) is referenced structurally (Section 5) but not yet derived step-by-step in this document.
- **$\partial \mathcal{L}/\partial X$** (gradient flowing back into the input tensor itself, needed to propagate into $W_{\text{out}}$, MLP weights, and — in the multi-layer case — the previous layer) is not derived here; this single-layer document treats $X$ as the block's fixed input and only derives gradients for $W^Q, W^K, W^V$.
- **Scale generalization**: this document uses $1/\sqrt{d_x d_t}$ throughout, generalizing the $1/d$ form used in the original single-$d$ derivation; reduces identically when $d_x = d_t = d$.
- **$W$ storage vs. theory.** Gradients in this derivation are $W_{mn,pq}$. Stored `W.grad` is $(m,p,n,q)$. Do not compare them axis-wise without `permute(0, 2, 1, 3)`. When $d_x=d_t$, a v1 checkpoint loads without shape error and silently mixes axes.
