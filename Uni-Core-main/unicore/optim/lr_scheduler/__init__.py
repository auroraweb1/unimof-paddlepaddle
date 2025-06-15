import os

"""isort:skip_file"""
import importlib
from unicore import registry
from unicore.optim.lr_scheduler.unicore_lr_scheduler import UnicoreLRScheduler

(
    build_lr_scheduler_,
    register_lr_scheduler,
    LR_SCHEDULER_REGISTRY,
) = registry.setup_registry(
    "--lr-scheduler", base_class=UnicoreLRScheduler, default="fixed"
)


def build_lr_scheduler(args, optimizer, total_train_steps):
    return build_lr_scheduler_(args, optimizer, total_train_steps)


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("unicore.optim.lr_scheduler." + file_name)
