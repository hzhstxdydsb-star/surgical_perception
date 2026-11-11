import json
import platform
import sys

from common import ROOT, require
import torch
import torchvision
import ultralytics

require(sys.version_info[:2] == (3, 12), "请使用 Python 3.12")
require(torch.cuda.is_available(), "CUDA 不可用，请检查驱动及 PyTorch 安装")
device = torch.device("cuda:0")
value = torch.randn(128, 128, device=device, requires_grad=True)
loss = (value @ value.T).square().mean()
loss.backward()
require(torch.isfinite(loss).item() and torch.isfinite(value.grad).all().item(), "CUDA 前向或反向计算异常")
report = {"python": sys.version, "system": platform.platform(), "torch": torch.__version__,
          "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
          "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
          "vram_GiB": torch.cuda.get_device_properties(0).total_memory / 1024**3,
          "cuda_forward_backward": True}
(ROOT / "work/environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
