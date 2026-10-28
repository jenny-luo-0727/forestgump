"""forestgump: did your model beat a random forest? Life is like a box of baselines."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("forestgump")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0+unknown"

from .core import run
from .io import InputError, load
from .report import Report

__all__ = ["InputError", "Report", "__version__", "load", "run"]
