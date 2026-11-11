import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from common import ROOT, NAMES, require
import cv2
import numpy as np
import yaml
from PIL import Image
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import polygon2mask

output = ROOT / "datasets/cholecinstanceseg/prepared"
with (ROOT / "datasets/cholecinstanceseg/raw/cholecinstanceseg_metadata.csv").open(newline="") as stream:
    metadata_rows = list(csv.DictReader(stream))
metadata = {row["Annotation_Path"].replace("\\", "/"): row for row in metadata_rows}
require(len(metadata) == len(metadata_rows) == 41933, "元数据存在遗漏或重复")
data = yaml.safe_load((output / "data.yaml").read_text())
require(data["names"] == dict(enumerate(NAMES)), "类别配置不一致")
data["path"] = str(output)
data["channels"] = 3
counts = Counter()
classes = defaultdict(Counter)
empty = Counter()
video_splits = defaultdict(set)
ious = []
low = []
total_intersection = total_union = 0
seen = set()
with zipfile.ZipFile(ROOT / "datasets/cholecinstanceseg/raw/cholecinstanceseg.zip") as archive:
    for record_line in (output / "conversion.jsonl").open():
        audit = json.loads(record_line)
        name = audit["annotation"]
        require(name not in seen, f"重复转换记录：{name}")
        seen.add(name)
        _, split, sequence, _, filename = name.split("/")
        require(metadata[name]["Dataset_Split"] == split and metadata[name]["Ann_filename"] == filename, f"元数据与标签路径不一致：{name}")
        require(metadata[name]["Video_ID"] == sequence.split("_")[0], f"视频编号不一致：{name}")
        video_splits[sequence.split("_")[0]].add(split)
        annotation = json.loads(archive.read(name))
        image_name = Path(annotation["imagePath"].replace("\\", "/")).name
        image_path = output / "images" / split / sequence / image_name
        raw_image = ROOT / "datasets/cholecinstanceseg/raw/cholecinstance_seg_reference_image_set" / split / sequence / "img_dir" / image_name
        require(raw_image.is_file() and raw_image.stat().st_size == image_path.stat().st_size, f"图像来源或文件大小不一致：{name}")
        label_path = output / "labels" / split / sequence / (image_path.stem + ".txt")
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
        require((width, height) == (audit["width"], audit["height"]), f"尺寸不一致：{name}")
        groups = defaultdict(list)
        for shape in annotation["shapes"]:
            groups[(shape["label"], shape["group_id"])].append(shape["points"])
        lines = label_path.read_text().splitlines()
        require(len(lines) == len(groups) == len(audit["instances"]), f"实例数量不一致：{name}")
        counts[split] += 1
        empty[split] += not lines
        for line, ((label, identity), polygons), previous in zip(lines, groups.items(), audit["instances"]):
            tokens = line.split()
            require(int(tokens[0]) == NAMES.index(label), f"类别错误：{name}")
            points = np.asarray([float(v) for v in tokens[1:]]).reshape(-1, 2)
            require(len(points) >= 3 and np.isfinite(points).all() and (points >= 0).all() and (points <= 1).all(), f"坐标错误：{name}")
            prediction = polygon2mask((height, width), [(points * [width, height]).reshape(-1)], color=1).astype(bool)
            truth = np.zeros((height, width), dtype=np.uint8)
            for polygon in polygons:
                cv2.fillPoly(truth, [np.asarray(polygon, dtype=np.int32)], 1)
            intersection = int(np.logical_and(truth, prediction).sum())
            union = int(np.logical_or(truth, prediction).sum())
            require(union > 0, f"空实例：{name}")
            iou = intersection / union
            require(abs(iou - previous["iou"]) < 1e-12, f"实际标签与转换记录不一致：{name}")
            require(previous["unconnected_iou"] >= 0.95 and previous["unexplained_pixels"] == 0, f"几何检查不通过：{name}")
            ious.append(iou)
            total_intersection += intersection
            total_union += union
            classes[split][label] += 1
            if iou < 0.95:
                low.append({"annotation": name, "class": label, "instance_id": identity, "iou": iou})
        if len(seen) % 2000 == 0:
            print(f"已独立核查 {len(seen)} 张图像及实际TXT标签", flush=True)
    require(seen == {name for name in archive.namelist() if name.endswith(".json")}, "标注没有完整覆盖")
require(counts == {"train": 26830, "val": 4284, "test": 10819}, f"划分数量错误：{counts}")
require(all(len(splits) == 1 for splits in video_splits.values()), "同一视频跨越不同集合")
official = {}
for split, count in counts.items():
    images = list((output / "images" / split).rglob("*.png"))
    labels = list((output / "labels" / split).rglob("*.txt"))
    require(len(images) == len(labels) == count, f"额外或缺失文件：{split}")
    dataset = YOLODataset(img_path=str(output / "images" / split), data=data, task="segment", imgsz=640,
                          augment=False, hyp=get_cfg(), batch_size=16)
    instances = sum(len(item["cls"]) for item in dataset.labels)
    require(len(dataset) == count and instances == sum(classes[split].values()), f"官方加载器遗漏图像或实例：{split}")
    official[split] = {"images": len(dataset), "instances": instances}
report = {"images": dict(counts), "total_images": len(seen), "instances": len(ious),
          "class_instances": {split: {name: value[name] for name in NAMES} for split, value in classes.items()},
          "missing_classes": {split: [name for name in NAMES if value[name] == 0] for split, value in classes.items()},
          "empty_images": dict(empty),
          "iou_mean": float(np.mean(ious)), "iou_min": min(ious), "iou_median": float(np.median(ious)),
          "iou_area_weighted": total_intersection / total_union, "instances_below_0_95": len(low),
          "no_video_split_overlap": True, "official_loader": official,
          "all_saved_labels_match_conversion_audit": True, "unexplained_pixels": 0}
(output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
(output / "low_iou_instances.json").write_text(json.dumps(low, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
