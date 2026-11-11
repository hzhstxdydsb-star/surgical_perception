import argparse
import json
import statistics
import time
from pathlib import Path

from common import ROOT, data_path, require
import torch
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "resume", "evaluate", "predict", "profile"])
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/cholecinstanceseg/prepared/data.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/yolo26m-seg.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.yaml")
    parser.add_argument("--name", default="yolo26m-cholec")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--source", type=str)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    require(args.weights.is_file(), f"找不到权重：{args.weights}")
    model = YOLO(str(args.weights))
    if args.action == "resume":
        model.train(resume=True, device=args.device)
        return
    config = yaml.safe_load(args.config.read_text())
    if args.action in {"train", "profile"}:
        config.update(data=data_path(args.data), device=args.device, batch=args.batch, project=str(ROOT / "runs/train"), name=args.name, exist_ok=False)
        require(not (ROOT / "runs/train" / args.name).exists(), "实验目录已存在，请设置新的 --name")
        durations = []
        started = []

        def epoch_start(trainer):
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            started[:] = [time.monotonic()]

        def epoch_end(trainer):
            if not started:
                return
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            durations.append(time.monotonic() - started[0])
            started.clear()

        model.add_callback("on_train_epoch_start", epoch_start)
        model.add_callback("on_fit_epoch_end", epoch_end)
        if args.action == "profile":
            require(args.device.isdigit() and torch.cuda.is_available(), "显存测量需要 CUDA GPU")
            target_epochs = config["epochs"]
            config.update(epochs=3, patience=0, save_period=-1, close_mosaic=0)
            torch.cuda.set_device(int(args.device))
            torch.cuda.reset_peak_memory_stats(int(args.device))
        model.train(**config)
        if args.action == "profile":
            require(len(durations) >= 2, "测量轮数不足")
            seconds = statistics.median(durations[1:])
            result = {"requested_batch": args.batch, "actual_batch": model.trainer.batch_size, "imgsz": config["imgsz"], "epoch_seconds": durations,
                      "representative_epoch_seconds": seconds, "target_epochs": target_epochs,
                      "estimated_training_hours": seconds * target_epochs / 3600,
                      "estimated_training_cost_yuan": seconds * target_epochs / 3600 * 3.41,
                      "peak_reserved_GiB": torch.cuda.max_memory_reserved(int(args.device)) / 1024**3,
                      "scope": "包含每轮训练与验证；排除上传、安装、最终测试和预测；三轮诊断不用于最终指标"}
            (Path(model.trainer.save_dir) / "profile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.action == "evaluate":
        output = ROOT / "runs/evaluate" / args.name
        require(not output.exists(), "评估目录已存在")
        metrics = model.val(data=data_path(args.data), split=args.split, device=args.device, batch=args.batch,
                            imgsz=config["imgsz"], plots=False, project=str(output.parent), name=args.name, exist_ok=True)
        result = {"split": args.split, "overall": {key: float(value) for key, value in metrics.results_dict.items()},
                  "per_class": json.loads(metrics.to_json())}
        (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        require(args.source is not None, "预测需要 --source 图像、目录或视频路径")
        output = ROOT / "runs/predict" / args.name
        require(not output.exists(), "预测目录已存在")
        output.mkdir(parents=True)
        with (output / "predictions.jsonl").open("w") as stream:
            for index, result in enumerate(model.predict(source=args.source, device=args.device, imgsz=config["imgsz"],
                                                        stream=True, save=True, project=str(output.parent), name=args.name, exist_ok=True)):
                stream.write(json.dumps({"index": index, "source": result.path, "objects": json.loads(result.to_json())}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
