"""T=3 window on a five-sentence story, plus the two-step slide.

The layers are FourTensorModel. This file only changes which sentences
enter the loss and how a finished window is advanced.

A stored story is S(0), S(1), S(2), S(3), S(4), shape (B, 5, N).

Training, teacher-forced:
    input  S(0), S(1), S(2)
    target S(1), S(2), S(3)
S(4) is not a training target. S(1) and S(2) are the overlap. S(3) is the
sentence absent from the input.

Evaluation, own tokens only:
    step 1  input S(0), S(1), S(2)           -> hat S(3), scored against S(3)
    step 2  input S(1), S(2), hat S(3)       -> hat S(4), scored against S(4)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from four_tensor_model import FourTensorModel, cross_entropy_loss

T_WINDOW = 3
N_STORY = T_WINDOW + 2  # S(0) .. S(T_WINDOW + 1) = S(4)


def _check_story(states: torch.Tensor, n_tokens: int) -> None:
    if states.dim() != 3:
        raise ValueError(
            f"states must be (B, {N_STORY}, N), got {tuple(states.shape)}"
        )
    if states.size(1) != N_STORY or states.size(2) != n_tokens:
        raise ValueError(
            f"states must be (B, {N_STORY}, {n_tokens})=S(0)..S(4), "
            f"got {tuple(states.shape)}"
        )


def training_story(states: torch.Tensor) -> torch.Tensor:
    """(B, 5, N) -> (B, 4, N) = S(0), S(1), S(2), S(3). S(4) stays out."""
    return states[:, : T_WINDOW + 1, :]


def sentence_token_stats(
    logits_last: torch.Tensor,
    gold: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-padding CE of one predicted sentence.

    logits_last : (B, N, V)   gold : (B, N)
    returns (ce_sum, n_correct, n_tok).
    """
    if logits_last.dim() != 3:
        raise ValueError(
            f"logits_last must be (B, N, V), got {tuple(logits_last.shape)}"
        )
    if gold.shape != logits_last.shape[:2]:
        raise ValueError(
            f"gold (B, N)={tuple(gold.shape)} vs logits {tuple(logits_last.shape[:2])}"
        )
    mask = gold != pad_id
    ce_sum = F.cross_entropy(
        logits_last.reshape(-1, logits_last.size(-1)),
        gold.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )
    n_correct = ((logits_last.argmax(dim=-1) == gold) & mask).sum()
    n_tok = mask.sum()
    return ce_sum, n_correct, n_tok


class FourTensorRolloutModel(FourTensorModel):
    """FourTensorModel built at T=3, with the five-sentence training slice."""

    def __init__(
        self,
        vocab_size: int,
        N: int,
        d_x: int,
        d_t: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        padding_idx: int,
        T: int = T_WINDOW,
    ):
        if T != T_WINDOW:
            raise ValueError(f"rollout model is fixed at T={T_WINDOW}, got {T}")
        super().__init__(
            vocab_size, N, T_WINDOW, d_x, d_t,
            n_layers, n_heads, dropout, padding_idx,
        )

    def training_step(self, states: torch.Tensor):
        """states : (B, 5, N) = S(0)..S(4)  ->  (loss, logits).

        logits : (B, N, 3, V) for input S(0), S(1), S(2).
        Loss is the mean CE of S(1), S(2), S(3). S(4) is not read.
        """
        _check_story(states, self.N)
        held = training_story(states)
        token_ids = held[:, :-1, :]
        logits = self.forward(token_ids)
        Y = held[:, 1:, :]
        loss = cross_entropy_loss(logits, Y, pad_id=self.padding_idx)
        return loss, logits

    @torch.no_grad()
    def rollout(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Two-step slide. states : (B, 5, N). Model is set to eval.

        Step 1 reads gold S(0), S(1), S(2) and scores hat S(3) against S(3).
        Step 2 reads gold S(1), S(2) and hat S(3), and scores hat S(4)
        against S(4). Gold S(3) and S(4) never enter a window.
        """
        _check_story(states, self.N)
        self.eval()
        pad = self.padding_idx

        window = states[:, :T_WINDOW, :]
        logits = self.forward(window)
        last = logits[:, :, -1, :]
        s3_ce, s3_ok, s3_n = sentence_token_stats(last, states[:, 3, :], pad)
        hat_s3 = last.argmax(dim=-1)

        window = torch.stack(
            [states[:, 1, :], states[:, 2, :], hat_s3],
            dim=1,
        )
        logits = self.forward(window)
        last = logits[:, :, -1, :]
        s4_ce, s4_ok, s4_n = sentence_token_stats(last, states[:, 4, :], pad)
        hat_s4 = last.argmax(dim=-1)

        return {
            "hat_s3": hat_s3,
            "hat_s4": hat_s4,
            "s3_ce_sum": s3_ce,
            "s3_n_correct": s3_ok,
            "s3_n_tok": s3_n,
            "s4_ce_sum": s4_ce,
            "s4_n_correct": s4_ok,
            "s4_n_tok": s4_n,
        }


class FourTensorRolloutUpdater:
    """AdamW / SGD on the T=3 teacher-forced loss. S(4) is not in the graph."""

    def __init__(
        self,
        model: FourTensorRolloutModel,
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

    def step(self, states: torch.Tensor):
        """states : (B, 5, N). Forward on S(0), S(1), S(2); loss on S(1), S(2), S(3)."""
        raw = self.model.module if hasattr(self.model, "module") else self.model
        if raw.T != T_WINDOW:
            raise ValueError(f"rollout updater expects T={T_WINDOW}, got {raw.T}")
        _check_story(states, raw.N)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        held = training_story(states)
        token_ids = held[:, :-1]
        runner = self.model
        device_ids = getattr(self.model, "device_ids", None)
        if device_ids is not None and token_ids.size(0) < len(device_ids):
            runner = raw
        logits = runner(token_ids)
        Y = held[:, 1:, :]
        loss = cross_entropy_loss(logits, Y, pad_id=raw.padding_idx)
        loss.backward()
        self.optimizer.step()
        return loss, logits
