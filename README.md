# 手术器械实例分割实验

使用 CholecInstanceSeg 对 YOLO26s-seg、YOLO26m-seg 和 YOLO26l-seg 进行全参数微调。三个模型使用各自对应的官方 COCO 预训练权重，以及相同的数据划分和训练配置。完整对比见[综合实验报告](reports/2026-11-03_yolo26_s_m_l_comparison/EXPERIMENT_REPORT_YOLO26_S_M_L.pdf)。

| 模型 | 完成轮数 | 最佳轮次 | 验证Mask mAP50-95 | 测试Mask mAP50-95 | FP32平均延迟（ms） |
|---|---:|---:|---:|---:|---:|
| YOLO26s-seg | 66 | 36 | 0.71575 | 0.76066 | 3.675 |
| YOLO26m-seg | 62 | 32 | 0.74391 | 0.76502 | 4.360 |
| YOLO26l-seg | 56 | 26 | 0.73870 | 0.75943 | 5.341 |

测试集10,819张图像、15,659个实例。每个模型只使用一个训练种子。GPU计时采用同一RTX 4090、相同50张已解码验证图像、10张预热、Batch=1、FP32和CUDA同步，包含预处理、推理及后处理；三种模型的计时运行时段不同，只有一组测量。测试集没有snare实例。逐类别JSON中无前缀mAP为Box指标，Mask逐类别mAP另见原始终端日志。

## 环境与数据

Python 3.12、PyTorch 2.14.1+cu126、torchvision 0.29.1、Ultralytics 8.4.190。按[数据准备说明](data_info/README.md)下载并解压官方数据。下列命令以 YOLO26l-seg 为例，复现 s 或 m 时替换模型名称和实验名称。

```bash
bash scripts/setup_server.sh
.venv/bin/python scripts/download_weights.py --model yolo26l-seg
.venv/bin/python scripts/prepare_data.py --images datasets/cholecinstanceseg/raw/cholecinstance_seg_reference_image_set
.venv/bin/python scripts/verify_prepared.py
.venv/bin/python scripts/run.py train --config configs/rtx4090.yaml --weights weights/yolo26l-seg.pt --batch 16 --name yolo26l-reproduce
.venv/bin/python scripts/run.py evaluate --config configs/rtx4090.yaml --weights runs/train/yolo26l-reproduce/weights/best.pt --batch 16 --split test --name yolo26l-reproduce-test
```

## 文件

- `scripts/` 和 `configs/` 保存数据转换、核查、训练、测试和速度测量代码及配置。
- `results/` 保存三个模型的实际配置、逐轮指标、测试结果和对比数据。
- `logs/` 保存三个模型的训练事件记录。
- `data_info/` 保存数据准备说明、类别、划分清单和转换核查摘要。
- `reports/` 保存单模型报告、综合报告、训练曲线和预测示例。

## 模型权重

三个 Hugging Face 模型仓库分别保存 `best.pt`、`last.pt`、阶段检查点、类别映射和评估结果。链接需要对应私有仓库的访问权限。

- [YOLO26s-seg](https://huggingface.co/huangzh111/yolo26s-seg-cholecinstanceseg)
- [YOLO26m-seg](https://huggingface.co/huangzh111/yolo26m-seg-cholecinstanceseg)
- [YOLO26l-seg](https://huggingface.co/huangzh111/yolo26l-seg-cholecinstanceseg)
