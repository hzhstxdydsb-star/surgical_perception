import json
import shutil
import tarfile
import time
from pathlib import Path

from common import ROOT, require
import yaml
from PIL import Image

output = ROOT / "upload"
output.mkdir(exist_ok=True)
archive_path = output / "yolofinetune-server.tar"
partial = archive_path.with_suffix(".tar.part")
require(not archive_path.exists() and not partial.exists(), "上传包已存在，请先核对已有文件")
selected = [ROOT / name for name in ("README.md", "UPLOAD.md", "VERIFICATION.md", "requirements.txt", "requirements.lock.txt",
            "scripts", "configs", "weights", "sources/cholec_instance_seg-main", "datasets/cholecinstanceseg/prepared")]
files = []
for source in selected:
    require(source.exists(), f"缺少文件或目录：{source}")
    for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
        if path.is_dir():
            continue
        require(not path.is_symlink(), f"上传包不接受符号链接：{path}")
        if "__pycache__" in path.parts or path.name == ".DS_Store" or path.suffix in {".cache", ".pyc"}:
            continue
        files.append(path)
require(len(files) == len(set(files)), "上传清单存在重复文件")
records = [{"path": "yolofinetune/" + path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size} for path in files]
size = sum(record["bytes"] for record in records)
require(shutil.disk_usage(output).free > size + 5 * 1024**3, "打包所需磁盘空间不足")
verification = json.loads((ROOT / "datasets/cholecinstanceseg/prepared/verification.json").read_text())
require(verification["total_images"] == 41933 and verification["instances"] == 64755, "数据验证结果不符合预期")
manifest = {"format": "uncompressed_tar", "files": records, "file_count_excluding_manifest": len(records),
            "payload_bytes": size, "dataset_images": verification["images"], "dataset_instances": verification["instances"]}
manifest_path = output / "upload_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
last_update = time.monotonic()
done = 0
with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
    for path, record in zip(files, records):
        require(path.stat().st_size == record["bytes"], f"文件大小发生变化：{path}")
        info = archive.gettarinfo(str(path), arcname=record["path"])
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        with path.open("rb") as stream:
            archive.addfile(info, stream)
        done += record["bytes"]
        now = time.monotonic()
        if now - last_update >= 20:
            print(f"已打包 {done / 1024**3:.2f}/{size / 1024**3:.2f} GiB", flush=True)
            last_update = now
    archive.add(manifest_path, arcname="yolofinetune/upload_manifest.json", recursive=False)
expected = {record["path"]: record["bytes"] for record in records}
expected["yolofinetune/upload_manifest.json"] = manifest_path.stat().st_size
with tarfile.open(partial, "r:") as archive:
    members = archive.getmembers()
    require(len(members) == len(expected), "包内文件数量不一致")
    require({member.name: member.size for member in members} == expected, "包内文件清单或大小不一致")
    require(all(member.isfile() for member in members), "包内存在非普通文件")
    data = yaml.safe_load(archive.extractfile("yolofinetune/datasets/cholecinstanceseg/prepared/data.yaml"))
    require(data["train"] == "images/train" and data["val"] == "images/val" and data["test"] == "images/test", "数据配置不完整")
    for split in ("train", "val", "test"):
        image_names = [name for name in expected if name.startswith(f"yolofinetune/datasets/cholecinstanceseg/prepared/images/{split}/")]
        label_names = [name for name in expected if name.startswith(f"yolofinetune/datasets/cholecinstanceseg/prepared/labels/{split}/")]
        require(len(image_names) == len(label_names) == verification["images"][split], f"包内图像标签数量错误：{split}")
        with Image.open(archive.extractfile(image_names[0])) as image:
            image.load()
    require(not any(part in {".venv", "work", "dependencies", "raw", "__pycache__", "runs"} for name in expected for part in Path(name).parts), "包内混入非交付目录")
partial.rename(archive_path)
result = {"archive": str(archive_path), "bytes": archive_path.stat().st_size, "GiB": archive_path.stat().st_size / 1024**3,
          "files_in_archive": len(expected), "file_list_and_sizes_verified": True, "sample_images_decoded": 3,
          "dataset_images": verification["images"], "dataset_instances": verification["instances"]}
(output / "package_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
