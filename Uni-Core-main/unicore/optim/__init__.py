import os

"""isort:skip_file"""
import importlib
from unicore import registry
from unicore.optim.unicore_optimizer import UnicoreOptimizer
from unicore.optim.fp16_optimizer import FP16Optimizer, separate_decay_params

__all__ = ["UnicoreOptimizer", "FP16Optimizer"]
_build_optimizer, register_optimizer, OPTIMIZER_REGISTRY = registry.setup_registry(
    "--optimizer", base_class=UnicoreOptimizer, default="adam"
)


def build_optimizer(args, params, separate=True, *extra_args, **extra_kwargs):
    if separate:
        params = separate_decay_params(args, params)
    return _build_optimizer(args, params, *extra_args, **extra_kwargs)


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("unicore.optim." + file_name)
