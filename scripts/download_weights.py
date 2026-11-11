import argparse
import json

from common import ROOT, require
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['yolo26s-seg', 'yolo26m-seg', 'yolo26l-seg'], default='yolo26m-seg')
args = parser.parse_args()
name = args.model + '.pt'
url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/' + name
destination = ROOT / "weights" / name
require(not destination.exists(), "权重已存在")
destination.parent.mkdir(parents=True, exist_ok=True)
partial = destination.with_suffix(".pt.part")
with requests.get(url, stream=True, timeout=(30, 120)) as stream:
    stream.raise_for_status()
    size = int(stream.headers["Content-Length"])
    with partial.open("wb") as file:
        for chunk in stream.iter_content(1024**2):
            file.write(chunk)
require(partial.stat().st_size == size, "下载大小与官方响应不一致")
partial.rename(destination)
destination.with_suffix('.source.json').write_text(json.dumps({"name": name, "bytes": size, "source": url}, indent=2))
print(f"已下载：{destination}")
