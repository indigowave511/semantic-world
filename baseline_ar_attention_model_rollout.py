"""T=3 baseline on a five-sentence story, plus the two-step free-run.

The layers are BaselineARAttention. This file only changes which sentences
enter the loss and how a finished sentence is appended.

A stored story is S(0), S(1), S(2), S(3), S(4), shape (B, 5, N).

Training, free-running. Gold is the target only:
    context [BOS] + non-pad S(0), S(1), S(2)
    target  non-pad S(3) + [EOS]
S(4) is not a training target. This loss includes [EOS], the same training
objective as BaselineARAttention.

Evaluation, own tokens only. Each reported CE drops [EOS] and divides by the
non-padding tokens of that sentence:
    step 1  context S(0), S(1), S(2)        -> hat S(3), scored against S(3)
    step 2  context S(1), S(2), hat S(3)    -> hat S(4), scored against S(4)
Gold S(3) and S(4) never enter a context.
"""

from __future__ import annotations

import torch

from baseline_ar_attention_model import BaselineARAttention, BaselineARUpdater

T_WINDOW = 3
N_STORY = T_WINDOW + 2  # S(0) .. S(4)


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


def slide_window(states: torch.Tensor, hat_s3: torch.Tensor) -> torch.Tensor:
    """Second window (B, 4, N) = S(1), S(2), hat S(3), S(4).

    The parent pack reads the first three rows as context and the last row
    as the S(4) target. Gold S(3) is not in this tensor.
    """
    if hat_s3.shape != states.shape[:1] + states.shape[2:]:
        raise ValueError(
            f"hat S(3) must be (B, N)={tuple(states.shape[:1] + states.shape[2:])}, "
            f"got {tuple(hat_s3.shape)}"
        )
    return torch.stack(
        [states[:, 1, :], states[:, 2, :], hat_s3, states[:, 4, :]],
        dim=1,
    )


class BaselineARRolloutModel(BaselineARAttention):
    """BaselineARAttention built at T=3, with the five-sentence training slice."""

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

    def _content_sentence(self, window: torch.Tensor):
        """Free-run the last row of a (B, 4, N) window.

        Returns (content_ce_sum, n_correct, n_tok, hat).
        content_ce_sum drops [EOS]. hat is (B, N), right-padded, [EOS] not stored.
        n_tok counts non-padding gold tokens.
        """
        if window.shape[1:] != (self.T + 1, self.N):
            raise ValueError(
                f"window must be (B, {self.T + 1}, {self.N}), got {tuple(window.shape)}"
            )
        _loss_sum, _n_all, content_sum, n_content = self._fair_unroll(window)
        hat = self.generate_S3(window)
        gold = self.gold_S_last(window)
        mask = gold != self.padding_idx
        n_correct = ((hat == gold) & mask).sum()
        n_tok = mask.sum()
        if int(n_tok.item()) != n_content:
            raise RuntimeError(
                f"content tokens {n_content} != non-pad gold {int(n_tok.item())}"
            )
        return content_sum, n_correct, n_tok, hat

    def training_step(self, states: torch.Tensor):
        """states : (B, 5, N) = S(0)..S(4)  ->  (loss, None).

        Context is S(0), S(1), S(2). Loss is the free-running CE of S(3)+[EOS].
        S(4) is not read.
        """
        _check_story(states, self.N)
        return super().training_step(training_story(states))

    @torch.no_grad()
    def rollout(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Two-step free-run. states : (B, 5, N). Model is set to eval.

        Step 1 reads gold S(0), S(1), S(2) and scores hat S(3) against S(3).
        Step 2 reads gold S(1), S(2) and hat S(3), and scores hat S(4)
        against S(4). Gold S(3) and S(4) never enter a context.
        """
        _check_story(states, self.N)
        self.eval()

        s3_ce, s3_ok, s3_n, hat_s3 = self._content_sentence(training_story(states))
        s4_ce, s4_ok, s4_n, hat_s4 = self._content_sentence(
            slide_window(states, hat_s3)
        )
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


class BaselineARRolloutUpdater(BaselineARUpdater):
    """AdamW / SGD on the T=3 free-run of S(3)+[EOS]. S(4) is not in the graph."""

    def step(self, states: torch.Tensor):
        """states : (B, 5, N). Context S(0), S(1), S(2); loss on S(3)+[EOS]."""
        raw = self.model.module if hasattr(self.model, "module") else self.model
        if raw.T != T_WINDOW:
            raise ValueError(f"rollout updater expects T={T_WINDOW}, got {raw.T}")
        _check_story(states, raw.N)
        return super().step(training_story(states))
