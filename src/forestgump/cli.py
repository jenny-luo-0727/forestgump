"""`forestgump run data.csv`: Run, Forest, run."""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from . import __version__, jokes
from .core import INNER_CV_WARNING, run
from .demo import MODELS, demo_frame
from .io import TASKS, InputError

EXIT_FOREST_WON = 1
EXIT_BAD_INPUT = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forestgump", description=f"Did your model beat a random forest? {jokes.TAGLINE}"
    )
    parser.add_argument("--version", action="version", version=f"forestgump {__version__}. {jokes.TAGLINE}")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser(
        "run",
        help="fit the baselines on your folds and judge your model",
        description="One CSV row per (molecule, repeat). The test set of (repeat r, fold f) is its rows; "
        "the training set is every other row of repeat r. --pred is your model's prediction for the row "
        "(for binary targets, the probability of the larger class label).",
    )
    r.add_argument("csv")
    _shared(r)
    r.add_argument("--smiles", default="smiles", help="SMILES column (default: smiles)")
    r.add_argument("--target", default="target", help="label column (default: target)")
    r.add_argument("--fold", default="fold", help="test-fold column (default: fold)")
    r.add_argument("--pred", default="pred", help="your model's out-of-fold prediction column (default: pred)")
    r.add_argument("--repeat", default=None, help="repeat column for repeated CV (default: one repeat)")
    r.add_argument("--task", choices=TASKS, default=None, help="default: binary if the target has 2 values")
    d = sub.add_parser("demo", help="judge a pretend model on 105 synthetic molecules labelled with Crippen logP")
    d.add_argument("--model", choices=MODELS, default="noise", help="noise loses, oracle wins (default: noise)")
    d.add_argument("--repeats", type=int, default=1, help="repeats of 5-fold CV (default: 1)")
    _shared(d)
    return parser


def _shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1, help="sklearn n_jobs (default: all cores)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--brag", default=jokes.DEFAULT_BRAG, help=f'how your model was trained (default: "{jokes.DEFAULT_BRAG}")'
    )
    p.add_argument("--seriously", action="store_true", help="plain output, no jokes")
    p.add_argument("--json", metavar="PATH", help="write every number to a JSON file")
    p.add_argument(
        "--forest-inner-cv",
        action="store_true",
        help="pick the forest's hyperparameters by inner CV instead of out-of-bag score (about 5x slower)",
    )
    p.add_argument(
        "--fail-if-forest-wins",
        action="store_true",
        help=f"exit {EXIT_FOREST_WON} if the forest won, so a CI pipeline fails when your model loses",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.seriously:
        print("Planting the forest...", file=sys.stderr, flush=True)
    if args.command == "demo":
        frame = demo_frame(args.model, n_repeats=args.repeats, seed=args.seed)
        columns = {"repeat": "repeat"}
    else:
        frame = pd.read_csv(args.csv)
        columns = {k: getattr(args, k) for k in ("smiles", "target", "fold", "pred", "repeat", "task")}
    if args.forest_inner_cv:
        print(f"forestgump: {INNER_CV_WARNING}", file=sys.stderr, flush=True)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="forest_selection='inner-cv'")  # already printed above
            report = run(
                frame,
                seed=args.seed,
                n_jobs=args.n_jobs,
                alpha=args.alpha,
                forest_selection="inner-cv" if args.forest_inner_cv else "oob",
                **columns,
            )
    except InputError as exc:
        print(f"forestgump: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    print(report.render(seriously=args.seriously, brag=args.brag))
    if args.json:
        report.to_json(args.json)
    return EXIT_FOREST_WON if args.fail_if_forest_wins and report.verdict == "lose" else 0


if __name__ == "__main__":
    sys.exit(main())
