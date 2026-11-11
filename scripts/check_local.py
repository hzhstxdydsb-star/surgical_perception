import argparse
import json
import shutil
from pathlib import Path

from common import ROOT, NAMES, require
from prepare_data import convert_annotation
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import SegmentationModel

torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ROOT / "work/local-check")
output = parser.parse_args().output.resolve()
require(not output.exists(), "本地检查目录已存在")
(output / "images/val").mkdir(parents=True)
(output / "labels/val").mkdir(parents=True)
source = ROOT / "sources/cholec_instance_seg-main/utilities/_sample_files/VID03_t80_sparse"
details = []
for annotation_path in sorted((source / "ann_dir").glob("*.json")):
    annotation = json.loads(annotation_path.read_text())
    image_path = source / annotation["imagePath"]
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
    lines, records = convert_annotation(annotation, width, height)
    details.extend(records)
    shutil.copy2(image_path, output / "images/val" / image_path.name)
    (output / "labels/val" / (image_path.stem + ".txt")).write_text("\n".join(lines))
require(len(details) == 23, "作者样例实例数量发生变化")
require(min(record["iou"] for record in details) >= 0.95, "样例转换误差过大")
require(all(record["unexplained_pixels"] == 0 for record in details), "样例存在未解释像素差异")
data = {"path": str(output), "train": "images/train", "val": "images/val", "names": dict(enumerate(NAMES)), "channels": 3}
(output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
dataset = YOLODataset(img_path=str(output / "images/val"), data=data, task="segment", imgsz=640,
                      augment=False, hyp=get_cfg(), batch_size=2)
selected = [i for i, label in enumerate(dataset.labels) if len(label["cls"]) > 0][:2]
require(len(selected) == 2, "需要两个真实有标注样例")
batch = dataset.collate_fn([dataset[i] for i in selected])
batch["img"] = batch["img"].float() / 255
pretrained = YOLO(str(ROOT / "weights/yolo26m-seg.pt"))
model = SegmentationModel(pretrained.model.yaml, ch=3, nc=7, verbose=False)
model.load(pretrained.model)
model.args = get_cfg()
model.names = dict(enumerate(NAMES))
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
loss, items = model(batch)
loss = loss.sum()
require(torch.isfinite(loss).item(), "损失存在非有限值")
loss.backward()
gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
require(gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients), "梯度无效")
optimizer.step()
optimizer.zero_grad(set_to_none=True)
pretrained.model = model
pretrained.save(str(output / "diagnostic.pt"))
results = pretrained.predict(source=str(output / "images/val"), device="cpu", imgsz=640, save=False, verbose=False)
require(len(results) == 10, "预测没有覆盖全部真实样例")
report = {"scope": "作者10张真实样例的工程检查；单次参数更新不作为训练效果评估", "images": 10,
          "instances": len(details), "min_conversion_iou": min(record["iou"] for record in details),
          "mean_conversion_iou": float(np.mean([record["iou"] for record in details])),
          "loss": float(loss.detach()), "finite_gradients": True, "prediction_results": len(results)}
(output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
