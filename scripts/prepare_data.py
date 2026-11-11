import argparse
import csv
import json
import runpy
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from common import ROOT, NAMES, require
import cv2
import numpy as np
import yaml
from PIL import Image
from shapely import make_valid, union_all
from shapely.geometry import Polygon, box
from ultralytics.data.converter import merge_multi_segment
from ultralytics.data.utils import polygon2mask


def polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            yield from polygon_parts(item)


def line_parts(geometry):
    if geometry.geom_type == "LineString":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            yield from line_parts(item)


def convert_annotation(annotation, width, height):
    groups = defaultdict(list)
    for shape in annotation["shapes"]:
        require(shape["label"] in NAMES, f"未知类别：{shape['label']}")
        require(isinstance(shape["group_id"], int), "实例编号必须是整数")
        points = np.asarray(shape["points"], dtype=np.float64)
        require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3 and np.isfinite(points).all(), "多边形坐标无效")
        groups[(shape["label"], shape["group_id"])].append(points)
    lines, records = [], []
    for (label, identity), contours in groups.items():
        originals = [Polygon(points) for points in contours]
        geometry = union_all([make_valid(p) for p in originals]).intersection(box(0, 0, width - 1, height - 1))
        parts = list(polygon_parts(geometry))
        require(parts and sum(p.area for p in parts) > 0, f"裁剪后实例没有面积：{label}/{identity}")
        rings = []
        for part in parts:
            rings.append(np.asarray(part.exterior.coords[:-1]))
            rings.extend(np.asarray(ring.coords[:-1]) for ring in part.interiors)
        # make_valid 返回的细线仍对应原标注前景像素，需要保留往返轮廓。
        thin_lines = list(line_parts(geometry))
        for thin_line in thin_lines:
            points = np.asarray(thin_line.coords)
            rings.append(np.concatenate([points, points[-2::-1]], axis=0))
        merged = np.concatenate(merge_multi_segment([p.reshape(-1).tolist() for p in rings]), axis=0) if len(rings) > 1 else rings[0]
        # 坐标放在所属像素内部，避免浮点序列化后向前取整一像素。
        normalized = (np.floor(merged) + 0.25) / np.array([width, height])
        line = str(NAMES.index(label)) + " " + " ".join(f"{x:.8f}" for x in normalized.reshape(-1))
        # 使用实际写入精度还原掩膜，统计裁剪及连接误差。
        restored = np.array([float(x) for x in line.split()[1:]]).reshape(-1, 2) * [width, height]
        prediction = polygon2mask((height, width), [restored.reshape(-1)], color=1, downsample_ratio=1).astype(bool)
        baseline = np.zeros((height, width), dtype=np.uint8)
        for part in parts:
            component = np.zeros_like(baseline)
            boundaries = [np.asarray(part.exterior.coords).astype(np.int32)]
            boundaries.extend(np.asarray(ring.coords).astype(np.int32) for ring in part.interiors)
            cv2.fillPoly(component, boundaries, 1)
            baseline |= component
        for thin_line in thin_lines:
            cv2.polylines(baseline, [np.asarray(thin_line.coords).astype(np.int32)], False, 1, thickness=1)
        boundary = np.zeros_like(baseline)
        cv2.polylines(boundary, [restored.astype(np.int32)], True, 1, thickness=1)
        expected = np.logical_or(baseline, boundary)
        unexplained = int(np.count_nonzero(expected != prediction))
        truth = np.zeros((height, width), dtype=np.uint8)
        for contour in contours:
            cv2.fillPoly(truth, [contour.astype(np.int32)], 1)
        intersection = np.logical_and(truth, prediction).sum()
        union = np.logical_or(truth, prediction).sum()
        require(union > 0, "实例像素面积为零")
        baseline_iou = np.logical_and(truth, baseline).sum() / np.logical_or(truth, baseline).sum()
        lines.append(line)
        records.append({"class": label, "instance_id": identity, "iou": float(intersection / union),
                        "unconnected_iou": float(baseline_iou), "unexplained_pixels": unexplained,
                        "connection_pixels": int(np.count_nonzero(boundary & (1 - baseline))),
                        "source_contours": len(contours), "output_parts": len(parts), "holes": sum(len(p.interiors) for p in parts),
                        "preserved_thin_lines": len(thin_lines),
                        "repaired_invalid_geometry": any(not p.is_valid for p in originals)})
    return lines, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="解压后的官方图像根目录")
    parser.add_argument("--annotations", type=Path, default=ROOT / "datasets/cholecinstanceseg/raw/cholecinstanceseg.zip")
    parser.add_argument("--metadata", type=Path, default=ROOT / "datasets/cholecinstanceseg/raw/cholecinstanceseg_metadata.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/cholecinstanceseg/prepared")
    parser.add_argument("--resume", action="store_true", help="继续已经中断的数据转换")
    args = parser.parse_args()
    require(args.resume or not args.output.exists(), "输出目录已存在，请使用 --resume 继续中断的转换")
    if args.resume:
        require((args.output / "conversion.jsonl").is_file(), "缺少转换记录，无法继续")
        require(not (args.output / "data.yaml").exists(), "转换已经完成，请执行结果验证")
    variables = runpy.run_path(str(ROOT / "sources/cholec_instance_seg-main/utilities/_general_utils/dataset_variables.py"))["CholecInstanceSegVariables"]
    with args.metadata.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    metadata = {r["Annotation_Path"].replace("\\", "/"): r for r in rows}
    require(len(metadata) == len(rows) == 41933, "元数据数量或唯一性错误")
    image_index = {}
    for path in args.images.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            require(path.parent.name == "img_dir", f"图像目录不符合作者结构：{path}")
            key = (path.parent.parent.name, path.name)
            require(key not in image_index, f"重复图像：{key}")
            image_index[key] = path
    args.output.mkdir(parents=True, exist_ok=args.resume)
    counts = Counter()
    paired = set()
    instance_count = 0
    low_iou = 0
    failed_conversion = 0
    completed = {}
    if args.resume:
        for record_line in (args.output / "conversion.jsonl").open():
            record = json.loads(record_line)
            require(record["annotation"] not in completed, "转换记录重复")
            completed[record["annotation"]] = record
        print(f"继续已记录的 {len(completed)} 张图像", flush=True)
    with zipfile.ZipFile(args.annotations) as archive, (args.output / "conversion.jsonl").open("a" if args.resume else "w") as audit:
        annotation_names = [n for n in archive.namelist() if n.endswith(".json")]
        require(len(annotation_names) == len(set(annotation_names)) and set(annotation_names) == set(metadata), "标注清单与元数据不一致")
        require(list(completed) == annotation_names[:len(completed)], "已有转换记录与原始标注顺序不一致")
        for name in annotation_names:
            _, split, sequence, _, filename = name.split("/")
            require(split == metadata[name]["Dataset_Split"] == variables.seq_to_split_dict[sequence], f"划分冲突：{name}")
            annotation = json.loads(archive.read(name))
            key = (sequence, Path(annotation["imagePath"].replace("\\", "/")).name)
            require(key in image_index, f"缺少图像：{key}")
            image_path = image_index[key]
            if name in completed:
                output_image = args.output / "images" / split / sequence / image_path.name
                output_label = args.output / "labels" / split / sequence / (image_path.stem + ".txt")
                require(output_image.is_file() and output_label.is_file(), f"已有转换文件不完整：{name}")
                details = completed[name]["instances"]
                low_iou += sum(d["iou"] < 0.95 for d in details)
                failed_conversion += sum(d["unconnected_iou"] < 0.95 or d["unexplained_pixels"] != 0 for d in details)
                instance_count += len(details)
                counts[split] += 1
                paired.add(key)
                continue
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
            if "imageWidth" in annotation:
                require((width, height) == (annotation["imageWidth"], annotation["imageHeight"]), f"尺寸冲突：{name}")
            lines, details = convert_annotation(annotation, width, height)
            output_image = args.output / "images" / split / sequence / image_path.name
            output_label = args.output / "labels" / split / sequence / (image_path.stem + ".txt")
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_image)
            output_label.write_text("\n".join(lines) + ("\n" if lines else ""))
            audit.write(json.dumps({"annotation": name, "width": width, "height": height, "instances": details}) + "\n")
            low_iou += sum(d["iou"] < 0.95 for d in details)
            failed_conversion += sum(d["unconnected_iou"] < 0.95 or d["unexplained_pixels"] != 0 for d in details)
            instance_count += len(details)
            counts[split] += 1
            paired.add(key)
            if sum(counts.values()) % 1000 == 0:
                audit.flush()
                print(f"已转换 {sum(counts.values())}/41933 张图像", flush=True)
    require(counts == {"train": 26830, "val": 4284, "test": 10819}, f"划分数量错误：{counts}")
    report = {"frames": dict(counts), "instances": instance_count, "unpaired_images": len(set(image_index) - paired), "instances_below_iou_0_95": low_iou, "failed_conversion_checks": failed_conversion}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    require(failed_conversion == 0, "存在几何还原误差或未解释像素差异，请检查 conversion.jsonl")
    (args.output / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(NAMES))}, sort_keys=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
