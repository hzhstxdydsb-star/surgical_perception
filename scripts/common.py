import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "work/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "work/torch"))

NAMES = ["grasper", "hook", "irrigator", "clipper", "bipolar", "scissors", "snare"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def data_path(path):
    import yaml

    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    data["path"] = str((path.parent / data["path"]).resolve())
    target = ROOT / "work" / (path.parent.name + "-resolved.yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return str(target)
