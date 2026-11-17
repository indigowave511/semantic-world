"""Train BaselineARAttentionQKReinjection with the existing baseline train loop.

baseline_ar_attention_train.py and baseline_ar_attention_model.py stay unchanged.
Checkpoints record both of those files and these two reinjection files. Use an
output_dir that does not contain the published best_*.pt runs.

Optional arguments, removed before the original parser:

    --reinject_from_layer N     default 1. 0 also reinjects into layer 0.
    --position_scale C          default: match mean 4-tensor Pos energy.
"""

from __future__ import annotations

import argparse
import sys

import baseline_ar_attention_train as train
from baseline_ar_attention_model_qk_reinject import BaselineARAttentionQKReinjection


def _parse_extra() -> tuple[int, float | None]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--reinject_from_layer", type=int, default=1)
    pre.add_argument("--position_scale", type=float, default=None)
    extra, remaining = pre.parse_known_args()
    if extra.reinject_from_layer < 0:
        raise SystemExit(
            f"--reinject_from_layer must be >= 0, got {extra.reinject_from_layer}"
        )
    sys.argv = [sys.argv[0], *remaining]
    return extra.reinject_from_layer, extra.position_scale


_original_collect_provenance = train.collect_provenance


def _collect_provenance():
    note = _original_collect_provenance()
    here = train.Path(__file__).resolve()
    note["source_files"]["baseline_ar_attention_model_qk_reinject.py"] = (
        train._source_file_note(here.with_name("baseline_ar_attention_model_qk_reinject.py"))
    )
    note["source_files"]["baseline_ar_attention_train_qk_reinject.py"] = (
        train._source_file_note(here)
    )
    note["qk_reinjection"] = {
        "from_layer": BaselineARAttentionQKReinjection.reinject_from_layer,
        "position_scale": BaselineARAttentionQKReinjection.position_scale,
        "position_scale_mode": (
            "explicit" if BaselineARAttentionQKReinjection.position_scale is not None
            else "mean_four_tensor_pos_energy"
        ),
        "q": True,
        "k": True,
        "v": False,
        "residual": False,
    }
    return note


def main() -> None:
    from_layer, scale = _parse_extra()
    BaselineARAttentionQKReinjection.reinject_from_layer = from_layer
    BaselineARAttentionQKReinjection.position_scale = scale
    train.BaselineARAttention = BaselineARAttentionQKReinjection
    train.collect_provenance = _collect_provenance
    train.train_routine(train.parse_args())


if __name__ == "__main__":
    main()
