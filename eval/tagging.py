"""Seen / unseen tagging (docs/evaluation_protocol.md, section 1).

"Unseen" is never assumed from the dataset name. A test fake is:
  seen              - its canonical generator appears among training fakes
  unseen_same_arch  - generator unseen, but its architecture family is in training
  unseen_new_arch   - neither the generator nor the architecture family was trained on
"""
from __future__ import annotations

import pandas as pd


def tag_seen(test: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    train_fakes = train[train.label == 1]
    seen_gen = set(train_fakes.generator)
    seen_arch = set(train_fakes.architecture)

    def tag(row) -> str:
        if row.label == 0:
            return "real"
        if row.generator in seen_gen:
            return "seen"
        if row.architecture in seen_arch:
            return "unseen_same_arch"
        return "unseen_new_arch"

    return test.apply(tag, axis=1)
