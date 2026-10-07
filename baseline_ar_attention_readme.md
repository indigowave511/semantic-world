# Standard AR Attention Baseline — $X(2)\to S(3)$

Companion to `four_tensor_readme.md`. Code: `baseline_ar_attention_model.py`, `baseline_ar_attention_train.py`. Same windows, same vocab. This is not a rank-4 ablation. It is what a **standard QKV model** can do with the same task. AR = autoregressive.

Pedagogical example: $T=3$. The pool may use another $T$; $X(T-1)=S(0)+\cdots+S(T-1)$ predicts $S(T)$ in every case.

---

## 1. Task

$$
X(2) \;=\; S(0)+S(1)+S(2)
\qquad\longrightarrow\qquad
\hat S(3)
$$

$X(2)$ is the three context sentences **as one token sequence**. $S(3)$ is the next sentence. Both models are asked for $\hat S(3)$ given **only** $X(2)$. Gold $S(3)$ is the target, not an extra input.

The 4-tensor model is built to **decompose** that object: token position $x$, sentence index $\tau$, fibers $(n,q)$. A standard attention model has none of those axes. Its tokens are a 1-D list. Position is a 1-D encoding. That is the comparison: same $X(2)\to S(3)$ map, with vs without the decomposition.

---

## 2. Verdict

The claim holds. **The best a standard model can do with $X(2)$ is not to decompose it.** It sees one token sequence and runs ordinary $Q,K,V$ on that sequence. There is no honest standard-QKV move that recovers $(x,\tau)$ or $(n,q)$ without becoming a different architecture.

What remains are packing and **how $S(3)$ is emitted**. Those are still 1-D sequence models. They are listed in Section 4. Default is continuation on the concatenated string (Section 3), **without gold $S(3)$ in the input**.

---

## 2.1 Teacher forcing is not a fair comparison

4-tensor last slice:

$$
\hat S(3)_x \;=\; f\big(X(2)\big)
\qquad\text{for every position $x$}.
$$

No token of $S(3)$ enters $X$. Gold $S(3)$ is used only as the CE **target**.

Causal teacher forcing at step $k$:

$$
\hat S(3)_k \;=\; f\big(X(2),\; S(3)_0,\; \ldots,\; S(3)_{k-1}\big)
$$

with **gold** $S(3)_{<k}$. That is extra **premise** information: true tokens of the target sentence, which the 4-tensor last slice never receives. A lower TF-CE does not mean a better architecture; it means the baseline was allowed to read the answer so far.

Fair premise for any baseline: the given data are $X(2)$ only. Gold $S(3)$ is a target, never an input.

$$
\hat S(3)_k \;=\; f\big(X(2),\; \hat S(3)_0,\; \ldots,\; \hat S(3)_{k-1}\big).
$$

$\hat S(3)_{<k}$ is **not** more premise. It is a function of $X(2)$ and the weights — the model's own previous calibrations, including mistakes. Data-processing: those tokens carry no bits about gold $S(3)$ that were not already determined by $(X(2),\theta)$. The premise stays $X(2)$. What increases is the **number of sequential prediction calibrations** along $S(3)$, not the information given to the task.

The 4-tensor last slice does one parallel calibration of every $x$ from the same premise. Free-running AR does $K$ calibrations, each allowed to use the previous ones. That is a different compute / factorization, not a richer input.

Teacher-forced CE is therefore **not** a reported comparison number against the 4-tensor. The number that sits next to 4-tensor last-slice decode is free-running $\hat S(3)$ from $X(2)$ only. Non-autoregressive decode (Section 4.3) matches the 4-tensor's *one-shot* factorization; it is not required for premise fairness.

---

## 3. Default: one sequence, standard QKV

Strip per-sentence PAD (id $0$). Concatenate in reading order. Periods already sit in the tokenizer, so no extra `SEP` is required.

```
X(2) = tom went to the store . he bought milk . then he walked home .
S(3) = tom drank the milk .
```

Sequence used at comparison time (no gold $S(3)$ on the left):

```
[BOS]  +  X(2)   →  then the model's own  \hat S(3)  +  [EOS]
```

Standard transformer decoder:

$$
H_i = E[\text{id}_i] + \mathrm{PE}(i),
\qquad
Q=HW^Q,\; K=HW^K,\; V=HW^V,
\qquad
A = \mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{d}} + \text{causal mask}\Big).
$$

$W^{Q,K,V}\in\mathbb{R}^{d\times d}$. One embedding table. Residual $+$ dropout on both branches, MLP width $4d$, Pre-LN on Attn and MLP, one LayerNorm before the tied readout $L=H^{\text{norm}}E^\top$ — same *policy* as 4-tensor §8–§9 plus Section 4.2, ordinary matrices.

**Fair path (train and eval).** Condition on `[BOS]+X(2)` only. At each step append the model's own next token (greedy or sample), not the gold token. Stop at `EOS` or $N$ tokens. Compare that $\hat S(3)$ to gold $S(3)$. If a step is wrong, the next step must use that wrong token.

Teacher forcing is the usual way to *train* a causal LM (Section 3.2). It is recorded; it is not the number compared to the 4-tensor.

This is the baseline. $X(2)$ is unstructured context. $S(3)$ is produced from $X(2)$ and, if AR, from previous **guesses**.

---

## 3.1 Why teacher-forced CE leaks — same example

Gold:

```
S(0) = tom went to the store .
S(1) = he bought milk .
S(2) = then he walked home .
S(3) = tom drank the milk .
```

```
 i :  0    1    2     3   4    5      6  7   8      9     10 11   12   13  14     15    16  17   18    19   20    21  22
id : BOS  tom  went  to  the  store   .  he  bought milk  .  then he  walked home  .    tom drank the  milk  .   EOS
     |←———————————————— X(2) ————————————————————————————————→|  |←———— S(3) ———→|
```

Teacher forcing at $i=17$ still inserts gold `drank` even if the model just said `ate`. That term is $-\log p(\texttt{the}\mid X(2),\,\texttt{tom drank})$. The 4-tensor last slice never receives `drank` or `ate`. So this CE is not comparable.

Fair AR at the same points (model said `ate` at the first $S(3)$ step):

| step | input | model may emit | if it emits | next input |
|---|---|---|---|---|
| 1 | `[BOS]+X(2)` | first word of $\hat S(3)$ | `ate` (wrong; gold is `tom`) | $X(2)$ + `ate` |
| 2 | $X(2)$ + `ate` | second word | … | $X(2)$ + `ate` + that word |
| … | own prefix only | … | … | … |

The finished $\hat S(3)$ is compared to gold. Gold $S(3)$ is never on the input side.

Training this way feeds stop-gradient own tokens (free-running / scheduled sampling). Discrete argmax blocks the gradient through the fed id; the loss still hits the logit that chose it. That is less stable than teacher forcing, which is why the method in Section 3.2 exists.

---

## 3.2 Teacher forcing (standard LM train; not the comparison)

Ordinary causal LM training concatenates gold $S(3)$ and predicts the next gold token at every position of that span. One forward pass, no own samples on the left.

| step | input seen (gold) | predicts | gold target | CE term |
|---|---|---|---|---|
| $i=16$ | `BOS … home . tom` | next after that `tom` | `drank` | $-\log p(\texttt{drank}\mid X(2),\,\texttt{tom})$ |
| $i=17$ | `… tom drank` | next | `the` | $-\log p(\texttt{the}\mid X(2),\,\texttt{tom drank})$ |
| $i=18$ | `… tom drank the` | next | `milk` | $-\log p(\texttt{milk}\mid \ldots)$ |
| $i=19$ | `… tom drank the milk` | next | `.` | $-\log p(\texttt{.}\mid \ldots)$ |
| $i=20$ | `… milk .` | next | `EOS` | $-\log p(\texttt{EOS}\mid \ldots)$ |

If at $i=16$ the model’s argmax is `ate`, the next input is still gold `drank`. That is teacher forcing. The five terms are one `F.cross_entropy`. It is stable and cheap.

It adds gold $S(3)_{<k}$ to the **premise** (Section 2.1). Report it if useful as an LM diagnostic. Do not put it next to 4-tensor last-slice CE / decoded $\hat S(3)$. Eval of a TF-trained model for the comparison is still free-running generation from `[BOS]+X(2)` (Section 3).

---

## 4. Other methods (still not a 4-tensor)

None of these decomposes $X$ into $X_{xn;\tau q}$. If a method starts adding a $\tau$-axis, an $x$-grid, or dual fibers, it has left the standard baseline.

### 4.1 Encoder–decoder (seq2seq)

Encoder: $X(2)$ as one sequence (bidirectional). Decoder: causal $S(3)$.

Same tokens, two stacks. The encoder may let every token of $S(2)$ see $S(0)$ (the 4-tensor window is also unmasked). It still has no $(x,\tau)$ structure — only a 1-D encoder sequence and a 1-D decoder sequence.

Use this if the comparison should allow **full context attention** on $X(2)$. It is the other fully standard answer (BART/T5-style). It is not stronger structure; it is a different split of the same 1-D string into “context” vs “target.”

### 4.2 Prefix-LM (one stack, two masks)

One decoder. Bidirectional mask on $X(2)$, causal mask on $S(3)$. Same idea as 4.1, one set of weights. Still 1-D.

### 4.3 Non-autoregressive $S(3)$

Encode $X(2)$, emit all tokens of $S(3)$ in one shot. Same premise $X(2)$; one calibration per position, no sequential feedback of $\hat S(3)_{<k}$. That matches the 4-tensor's **parallel factorization**, not a smaller premise. Own-error AR has the same premise and more calibration steps. NAR is optional if the table should match factorization, not if it should match information.

### 4.4 Hierarchical sentence encoder

Encode each $S(\tau)$ alone, attend over three sentence vectors, then decode $S(3)$.

This **does** split $X(2)$ — but only into “three sentences.” That is ordinary hierarchical NLP, not the 4-tensor hypothesis (spatial slot $x$, dual fibers, rank-4 $W_{mn,pq}$). It also throws away cross-sentence token attention unless a second token-level pass is added, at which point it has converged back to 4.1 with extra machinery.

Not the default. Mention it only to show that “sentence-aware packing” is available and still is not rank-4.

### 4.5 What is not a method

Mean-pooling $X(2)$ into one vector and classifying a sentence. Bag-of-tokens. These are weaker 1-D models, not alternatives.

Anything that stores activations as $(N,T,d_x,d_t)$ or factors $W$ as $W_{mn,pq}$ is the 4-tensor model. It is not a standard baseline.

---

## 5. What to run

| Column | Model | Premise | Calibrations on $S(3)$ |
|---|---|---|---|
| **Default (fair AR)** | decoder-only QKV | $X(2)$ only | $K$ sequential, each uses own $\hat S(3)_{<k}$ |
| Optional | NAR encoder | $X(2)$ only | one shot, parallel (4-tensor-like factorization) |
| Mentioned (not the comparison) | teacher-forced AR | $X(2)$ **+ gold** $S(3)_{<k}$ | standard LM train; extra premise |

Matched width, if a parameter column is wanted: $d=d_x d_t$. That matches the **count** of flattened $W^Q$, not the map.

---

## 6. Pseudo code (default)

```python
def pack_X(token_ids_full, pad_id=0):
    """token_ids_full : (B, T+1, N)
    X(T-1) = concat of sentences 0..T-1, PAD stripped.
    S(T)   = last sentence, PAD stripped.
    """
    ...

def forward(ids, E, PE, layers, ln):
    H = E[ids] + PE
    blocked = causal_or_pad(ids)                 # finfo.min, not -inf
    for layer in layers:
        # live code: Pre-LN on each branch (Section 10); residual add is raw H
        H = H + drop(layer.attn(layer.ln_attn(H), ids))
        H = H + drop(layer.mlp(layer.ln_mlp(H)))
    return ln(H) @ E.T                           # (B, L, |V|), tied  -- §8 analogue

def training_step_fair(token_ids_full, model):
    """Free-running: feed own tokens for S(3), gold is target only."""
    prefix = pack_X(token_ids_full).prefix        # [BOS]+X(2)
    gold   = pack_X(token_ids_full).S3_eos        # gold S(3)+EOS
    ids = prefix
    loss = 0.0
    for y in gold:
        logits = model(ids)[:, -1, :]             # (B, |V|)
        loss = loss + F.cross_entropy(logits, y)
        hat = logits.argmax(dim=-1)               # own token, not y
        ids = concat(ids, hat.detach())           # stop-grad through the fed id
    return loss / len(gold)


def training_step_tf(token_ids_full, model):
    """Teacher forcing: gold S(3) is on the left. Documented only — not in the model file."""
    ids, s3_span = pack_X(token_ids_full).gold_concat   # [BOS]+X(2)+S(3)+[EOS]
    logits = model(ids)
    return F.cross_entropy(logits[:, :-1][s3_span], ids[:, 1:][s3_span])


def generate_S3(model, token_ids_full):
    prefix = pack_X(token_ids_full).prefix         # [BOS]+X(2)  -- no gold S(3)
    return decode_until_eos(model, prefix)         # \hat S(3), own tokens only
```

---

## 7. One sentence

---

## 8. Knobs vs the current 4-tensor run

$d = d_x d_t$. ROCStories shell: $d_x=64$, $d_t=32$ $\Rightarrow$ $d=2048$. $d_{ff}=4d=8192$ follows §1; do not pick a second hidden width.

**Copy (no new choice).** Pool `T,N,|V|`, split, `padding_idx=0`. `n_layers` matches the 4-tensor run. Residual dropout on both branches, Pre-LN on Attn/MLP (Section 10), one final `LayerNorm(d)`, tied $E^\top`. AdamW, `lr=3e-4`, `weight_decay=0.01`, `batch=32`/GPU, `epochs=50`, `seed` from the shell. Eval greedy, `temperature=1.0`, content cap $L_{\max}=N$. Sinusoidal 1-D PE (same formula, one index).

**Must set (architecture has no 4-tensor analogue).**

| Knob | Why it is not a copy | Recommendation |
|---|---|---|
| `n_heads` | 4-tensor heads are *copies* of full $(d_x,d_t)$, not a split of $d$. Standard heads split $d=H d_h$. | **$H=1$** ($d_h=2048$) to match the current run. $H=8$ is a different baseline. $d$ must divide $H$. |
| `BOS` / `EOS` | 4-tensor vocab has only `PAD,UNK,+words`. Generation needs a stop token. | Append two ids at the **end** of the existing vocab. Do not reuse `PAD`. No `SEP` (periods already tokenize). |
| Train path | 4-tensor is one parallel forward on the window. | Comparison: free-running from $X(T-1)$ (Section 3). Teacher forcing (Section 3.2) may be used to *fit* the LM; do not report TF-CE next to 4-tensor last-slice. |

**Not a knob — consequence of $d=d_x d_t$.** One table $E\in\mathbb{R}^{|V|\times 2048}$ vs 4-tensor $E_{\text{sem}}\oplus E_{\text{temp}}$ of size $|V|\times(64+32)$. Attention $W^Q$ count matches ($d^2=d_x^2 d_t^2$); embedding count does not. $W'$ has no counterpart (tying removes it). Do not add a second embedding width unless dropping the $d=d_x d_t$ rule.

**Batch / VRAM.** TF train is one forward of length $\sim L$ (PAD stripped, $L\ll NT$). Fair AR train is $K$ sequential forwards — may need a smaller batch even if TF at $32$ fits. Eval generate is the same $K$-step loop for both train choices.



README와 같은 표기·문체로 맞춰서, 섹션 8 다음에 들어갈 새 섹션을 작성했습니다. 그대로 삽입하시면 됩니다.

---

## 9. OOM fix: last-position-only readout

`training_step_fair` (Section 6) calls `model(ids)[:, -1, :]` inside the loop. The slice happens **after** `model(ids)` returns, i.e. after the tied readout $L = H^{\text{norm}}E^\top$ has already been computed for every position.

At unroll step $k$ (sequence length $\ell_k = |{[\text{BOS}]+X(T-1)}| + k$):

$$
L_k \in \mathbb{R}^{B \times \ell_k \times |V|}, \qquad \text{only } L_k[:, -1, :] \text{ is used.}
$$

The other $\ell_k-1$ rows enter the autograd graph (kept for backward) and are never read. With $d=2048$, $|V|=36{,}752$, this is $\sim\!76$MB per step at $B{=}32$; summed over $K\approx N{+}1$ steps it is the difference between OOM and not, independent of $d_x,d_t$ or batch size — it grows with $\ell_k$ and $K$, not with model width.

### 9.1 Fix

Slice the readout input, not the readout output. $E^\top$ is applied per-position ($L_{i} = H_i^{\text{norm}} E^\top$, no cross-position term), so

$$
\big(H^{\text{norm}}E^\top\big)[:, -1, :] \;=\; H^{\text{norm}}[:, -1:, :]\,E^\top
$$

exactly — this is an identity, not an approximation. $H$ itself (all $\ell_k$ positions, full causal context) is untouched; only the matrix the readout is applied to shrinks from $(B,\ell_k,d)$ to $(B,1,d)$.

```python
def forward_last(ids, E, PE, layers, ln):
    H = E[ids] + PE
    blocked = causal_or_pad(ids)
    for W_Q, W_K, W_V, mlp in layers:
        Q, K, V = H @ W_Q, H @ W_K, H @ W_V           # full sequence, unchanged
        A = softmax((Q @ K.T) / sqrt(d) + blocked)
        H = H + drop(A @ V)
        H = H + drop(mlp(H))
    # live CausalDecoderLayer: Attn(ln_attn(H)), MLP(ln_mlp(H)); this loop is
    # the residual algebra. final ln is still once, after the stack (Section 10).
    H_last = ln(H)[:, -1:, :]                          # slice BEFORE readout
    return (H_last @ E.T).squeeze(1)                    # (B, |V|), not (B, L, |V|)

def training_step_fair(token_ids_full, model):
    prefix = pack_X(token_ids_full).prefix
    gold   = pack_X(token_ids_full).S3_eos
    ids = prefix
    loss = 0.0
    for y in gold:
        logits = model.forward_last(ids)                # (B, |V|) directly
        loss = loss + F.cross_entropy(logits, y)
        hat = logits.argmax(dim=-1)
        ids = concat(ids, hat.detach())
    return loss / len(gold)
```

Do **not** slice `H` before the transformer layers or feed only `H[:, -1:, :]` forward as the next step's input — attention at step $k{+}1$ still needs $K,V$ from every earlier position. Only the readout input is narrowed.

### 9.2 What this does and does not fix

Reduces the per-step readout tensor by a factor of $\ell_k$ (from $\sim\!76$MB to $\sim\!4.5$MB at step 1, growing with $\ell_k$ before the fix, flat after). Does not touch $E$'s parameter/optimizer footprint ($|V|\times d$, $\times 4$ under AdamW) — that scales with $d=d_xd_t$ per Section 8 and is a separate lever. Loss and gradients are numerically identical to the unpatched version; this is a memory fix, not a change to what is being fit (Section 2.1's fair-AR premise is unchanged).

---

## 10. Pre-LN (same change as 4-tensor Section 4.2)

H2L2 four-tensor without per-layer LN did not train (train CE stuck $\sim 6$). The baseline used the **same** block: residual adds, **no** Pre-LN, one `final_norm` before $E^\top$ (`CausalDecoderLayer`: “No per-layer pre-norm”). That was matching 4-tensor §8, not a missing LN unique to the baseline.

Live `baseline_ar_attention_model.py` keeps that readout LN and adds Pre-LN on each branch, same as four-tensor:

$$
H \leftarrow H + \mathrm{dropout}\big(\mathrm{Attn}(\mathrm{LN}_d(H))\big)
$$
$$
H \leftarrow H + \mathrm{dropout}\big(\mathrm{MLP}(\mathrm{LN}_d(H))\big)
$$
$$
H^{\text{norm}} = \mathrm{LN}_d(H)\quad\text{once, after the last layer, then }L=H^{\text{norm}}E^\top
$$

Two `nn.LayerNorm(d)` on `CausalDecoderLayer` (`ln_attn`, `ln_mlp`). Residual stream stays the raw sum. Do not apply Pre-LN on four-tensor only — an $L=2$ bake-off would no longer share a block.

This is not the §9 OOM readout slice. QKV still run on the full prefix; only the **inputs of Attn and MLP** are normalized. Backup without Pre-LN: `baseline_ar_attention_model0921.py`.