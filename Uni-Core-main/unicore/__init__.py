import os

"""isort:skip_file"""
import sys

try:
    from .version import __version__
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()
__all__ = ["pdb"]
from unicore.distributed import utils as distributed_utils
from unicore.logging import meters, metrics, progress_bar

sys.modules["unicore.distributed_utils"] = distributed_utils
sys.modules["unicore.meters"] = meters
sys.modules["unicore.metrics"] = metrics
sys.modules["unicore.progress_bar"] = progress_bar
import unicore.losses
import unicore.distributed
import unicore.models
import unicore.modules
import unicore.optim
import unicore.optim.lr_scheduler
import unicore.tasks
