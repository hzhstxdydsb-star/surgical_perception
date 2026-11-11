import json
import zipfile

from common import ROOT
from prepare_data import convert_annotation

summary = {"checked_frames": 0, "frames_requiring_images": 0, "instances": 0, "minimum_iou": 1.0, "below_0_95": [], "failed_checks": []}
with zipfile.ZipFile(ROOT / "datasets/cholecinstanceseg/raw/cholecinstanceseg.zip") as archive:
    for name in archive.namelist():
        if not name.endswith(".json"):
            continue
        annotation = json.loads(archive.read(name))
        if "imageWidth" not in annotation:
            summary["frames_requiring_images"] += 1
            continue
        lines, details = convert_annotation(annotation, annotation["imageWidth"], annotation["imageHeight"])
        summary["checked_frames"] += 1
        summary["instances"] += len(details)
        for item in details:
            summary["minimum_iou"] = min(summary["minimum_iou"], item["iou"])
            if item["iou"] < 0.95:
                summary["below_0_95"].append({"annotation": name, **item})
            if item["unconnected_iou"] < 0.95 or item["unexplained_pixels"] != 0:
                summary["failed_checks"].append({"annotation": name, **item})
        if summary["checked_frames"] % 2000 == 0:
            print(f"已检查 {summary['checked_frames']} 张标注", flush=True)
(ROOT / "work/geometry-check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in summary.items()}, ensure_ascii=False))
