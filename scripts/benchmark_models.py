import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

from common import ROOT, require
import cv2
import numpy as np
import torch
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=Path, nargs='+', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--samples', type=int, default=50)
args = parser.parse_args()
require(args.samples >= 10, '样本数量至少为10，以完成预热')
require(all(path.is_file() for path in args.weights), '找不到待测权重')
require(torch.cuda.is_available(), '推理性能测量需要CUDA')
torch.cuda.set_device(0)
paths = sorted((ROOT / 'datasets/cholecinstanceseg/prepared/images/val').rglob('*.png'))
selected = random.Random(42).sample(paths, args.samples)
images = [cv2.imread(str(path)) for path in selected]
require(all(image is not None for image in images), '图像解码失败')
reports = []
for weights in args.weights:
    model = YOLO(str(weights))
    for image in images[:10]:
        model.predict(image, device=0, imgsz=640, half=False, conf=0.25, verbose=False, save=False)
    torch.cuda.synchronize()
    require(next(model.model.parameters()).dtype == torch.float32, '基准测试要求FP32')
    torch.cuda.reset_peak_memory_stats(0)
    durations = []
    for image in images:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(image, device=0, imgsz=640, half=False, conf=0.25, verbose=False, save=False)
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    report = {'weights': str(weights), 'images': len(images), 'batch': 1, 'imgsz': 640, 'precision': 'FP32', 'warmup_images': 10, 'mean_ms': statistics.mean(durations), 'median_ms': statistics.median(durations), 'p95_ms': float(np.percentile(durations, 95)), 'fps': 1000 / statistics.mean(durations), 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3, 'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 1024**3, 'durations_ms': durations}
    reports.append(report)
    del result, model
    gc.collect()
    torch.cuda.empty_cache()
output = {'scope': '同一批真实验证图像，CUDA同步计时，包含预处理、模型推理和后处理，排除文件解码、绘制和视频编码', 'seed': 42, 'sources': [str(path.relative_to(ROOT)) for path in selected], 'models': reports}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
print(json.dumps(output, ensure_ascii=False, indent=2))
