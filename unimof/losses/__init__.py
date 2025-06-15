import importlib
from pathlib import Path

for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("unimof.losses." + file.name[:-3])
