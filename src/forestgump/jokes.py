"""Everything the forest says. `--seriously` turns all of it off."""

from __future__ import annotations

TAGLINE = "Life is like a box of baselines."
DEFAULT_BRAG = "trained on 8×H100 for 3 days"  # noqa: RUF001

BANNER = r"""
      /\          /\            /\
     /  \        /  \   /\     /  \
    /    \  /\  /    \ /  \   /    \
   /______\/  \/______\____\ /______\
      ||   /____\  ||    ||     ||
  ~~~~||~~~~~||~~~~||~~~~||~~~~~||~~~~
          RUN, FOREST, RUN!
"""

FOREST_FACTS = (
    "A random forest is a mixture of experts with hard routing. The experts are trees.",
    "Each tree saw a bootstrap resample of your data. With a GPU, that is called data augmentation.",
    "The forest's context window is your entire training set.",
    "A Morgan fingerprint is a message-passing network with zero trainable parameters.",
    "A regression forest cannot predict outside the range of its training labels. It does not hallucinate.",
    "The forest's attention mechanism is called max_features.",
    "The forest was not pretrained. The forest was not fine-tuned. The forest was planted.",
)

MODEL_CARD = """THE FOREST: MODEL CARD
  Architecture      {n_trees} decision trees, standing around
  Leaves            {n_leaves:,} (we counted, on the last fold)
  Pretraining       none
  Fine-tuning       none
  RLHF              the forest does not take feedback
  GPU               none
  Feelings          none
  Carbon footprint  negative"""


def _seconds(t: float) -> str:
    return f"{t:.1f}" if t < 10 else f"{t:,.0f}"


def lose(brag: str, forest_seconds: float) -> str:
    return (
        f"Your model, {brag}, lost to a random forest. "
        f"The forest trained in {_seconds(forest_seconds)} seconds on a CPU. "
        "The forest has no GPU. The forest has no feelings. The forest won."
    )


def tie(p: float) -> str:
    return f"Statistically tied (Holm-adjusted p = {p:.3g}). The forest is unimpressed."


def win(p: float) -> str:
    return (
        f"Your model beat the forest (Holm-adjusted p = {p:.3g}). The forest congratulates you. "
        "The forest has no hands, so it cannot clap."
    )


def refuse(n_cells: int) -> str:
    return f"The forest declines to be judged on n={n_cells}. Come back with at least 3 folds."


LOST_TO_MEAN = "Your model also lost to the mean. The mean is one number."
LOST_TO_KNN = "Your model also lost to the nearest-neighbor lookup. It is a lookup table."


def fact(seed: int) -> str:
    return "Forest fact: " + FOREST_FACTS[seed % len(FOREST_FACTS)]
