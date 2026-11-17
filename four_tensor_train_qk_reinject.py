"""Train FourTensorPositionReinjectionModel with the existing train loop.

four_tensor_train.py and four_tensor_model.py stay unchanged. Checkpoints record
both of those files and these two reinjection files. Use an output_dir that does
not contain the published best_*.pt runs.

Optional argument, removed before the original parser:

    --reinject_from_layer N     default 1. 0 also reinjects into layer 0.
"""

from __future__ import annotations

import argparse
import sys

import four_tensor_train as train
from four_tensor_model_qk_reinject import FourTensorPositionReinjectionModel


def _parse_reinject_flag() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--reinject_from_layer", type=int, default=1)
    extra, remaining = pre.parse_known_args()
    if extra.reinject_from_layer < 0:
        raise SystemExit(
            f"--reinject_from_layer must be >= 0, got {extra.reinject_from_layer}"
        )
    sys.argv = [sys.argv[0], *remaining]
    return extra.reinject_from_layer


_original_collect_provenance = train.collect_provenance


def _collect_provenance():
    note = _original_collect_provenance()
    here = train.Path(__file__).resolve()
    note["source_files"]["four_tensor_model_qk_reinject.py"] = (
        train._source_file_note(here.with_name("four_tensor_model_qk_reinject.py"))
    )
    note["source_files"]["four_tensor_train_qk_reinject.py"] = (
        train._source_file_note(here)
    )
    note["position_reinjection"] = {
        "from_layer": FourTensorPositionReinjectionModel.reinject_from_layer,
        "q": True,
        "k": True,
        "v": False,
        "residual": False,
        "buffers": ["P", "Temb"],
    }
    return note


def main() -> None:
    from_layer = _parse_reinject_flag()
    FourTensorPositionReinjectionModel.reinject_from_layer = from_layer
    train.FourTensorModel = FourTensorPositionReinjectionModel
    train.collect_provenance = _collect_provenance
    train.train_routine(train.parse_args())


if __name__ == "__main__":
    main()
