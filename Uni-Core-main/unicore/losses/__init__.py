import os

"""isort:skip_file"""
import importlib
from unicore import registry
from unicore.losses.unicore_loss import UnicoreLoss

build_loss_, register_loss, CRITERION_REGISTRY = registry.setup_registry(
    "--loss", base_class=UnicoreLoss, default="cross_entropy"
)


def build_loss(args, task):
    return build_loss_(args, task)


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("unicore.losses." + file_name)
