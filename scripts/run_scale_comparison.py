import argparse
import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, require
import torch
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--weights', type=Path, required=True)
parser.add_argument('--reference', required=True)
parser.add_argument('--profile', required=True)
parser.add_argument('--max-estimated-hours', type=float, default=7.5)
args = parser.parse_args()
os.chdir(ROOT)
work = ROOT / 'work/yolo26s-comparison'
work.mkdir(exist_ok=True)
os.environ['TMPDIR'] = str(work)
os.environ['PYTHONUNBUFFERED'] = '1'
python = ROOT / '.venv/bin/python'
config = ROOT / 'configs/rtx4090.yaml'
state_file = work / 'state.json'


def state(stage, **values):
    result = {'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(), 'name': args.name, **values}
    temporary = state_file.with_suffix('.json.part')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.replace(state_file)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def execute(label, command):
    state(label, log=str((work / (label + '.log')).relative_to(ROOT)))
    with (work / (label + '.log')).open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


profile = json.loads((ROOT / 'runs/train' / args.profile / 'profile.json').read_text())
profile_args = yaml.safe_load((ROOT / 'runs/train' / args.profile / 'args.yaml').read_text())
require(profile['actual_batch'] == 16 and profile['imgsz'] == 640 and profile_args['close_mosaic'] == 0, '测量设置不一致')
require(profile['peak_reserved_GiB'] <= 20, '测量显存余量不足')
require(profile['estimated_training_hours'] <= args.max_estimated_hours, '预计训练时长超过剩余时间预算')
(work / 'approved_profile.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2))
run = ROOT / 'runs/train' / args.name
require(not run.exists(), '正式实验目录已经存在')
execute('training', [str(python), 'scripts/run.py', 'train', '--config', str(config), '--weights', str(args.weights), '--batch', '16', '--name', args.name])

new_args = yaml.safe_load((run / 'args.yaml').read_text())
old_args = yaml.safe_load((ROOT / 'runs/train' / args.reference / 'args.yaml').read_text())
matched = ['data', 'epochs', 'patience', 'batch', 'imgsz', 'nbs', 'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'amp', 'seed', 'deterministic', 'freeze', 'workers', 'close_mosaic', 'mosaic', 'fliplr', 'translate', 'scale', 'mask_ratio', 'overlap_mask']
require(all(new_args[key] == old_args[key] for key in matched), '与参考模型的训练参数不一致')
evaluations = {}
for label, name in [('s', args.name), ('m', args.reference)]:
    evaluate_name = f'{args.name}-{label}-test'
    execute(f'test-{label}', [str(python), 'scripts/run.py', 'evaluate', '--config', str(config), '--weights', str(ROOT / 'runs/train' / name / 'weights/best.pt'), '--batch', '16', '--split', 'test', '--name', evaluate_name])
    evaluations[label] = json.loads((ROOT / 'runs/evaluate' / evaluate_name / 'metrics.json').read_text())
execute('benchmark', [str(python), 'scripts/benchmark_models.py', '--weights', str(ROOT / 'runs/train' / args.reference / 'weights/best.pt'), str(run / 'weights/best.pt'), '--output', str(work / 'benchmark.json')])

with (run / 'results.csv').open() as stream:
    rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
require(rows and all(math.isfinite(value) for row in rows for value in row.values()), '训练指标不完整或非有限')
best_row = max(rows, key=lambda row: row['metrics/mAP50-95(B)'] + row['metrics/mAP50-95(M)'])
best_epoch = int(best_row['epoch'])
training_log = (work / 'training.log').read_text()
early = re.search(r'Best results observed at epoch (\d+)', training_log)
if early:
    require(best_epoch == int(early.group(1)), '最佳轮次与原生日志不一致')
best = torch.load(run / 'weights/best.pt', map_location='cpu', weights_only=False)
require(best['model'].nc == 7 and best['optimizer'] is None, '最佳权重验证失败')
checkpoints = sorted((run / 'weights').glob('epoch*.pt'), key=lambda path: int(path.stem[5:]))
require(bool(checkpoints), '缺少阶段检查点')
checkpoint = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)
require(isinstance(checkpoint['optimizer'], dict), '阶段检查点缺少优化器状态')
summary = {'model': 'YOLO26s-seg', 'dataset': 'CholecInstanceSeg', 'completed_epochs': len(rows), 'best_epoch': best_epoch, 'early_stopping_patience': 30, 'training_hours': rows[-1]['time'] / 3600, 'independent_test_evaluated': True, 'classes': best['model'].names, 'best_checkpoint_loadable': True, 'resume_checkpoint': 'checkpoints/' + checkpoints[-1].name, 'resume_completed_epochs': checkpoint['epoch'] + 1, 'resume_optimizer_state_present': True}
(work / 'training_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
(work / 'best_epoch_validation_metrics.json').write_text(json.dumps({'split': 'val', 'epoch': best_epoch, 'metrics': {key: value for key, value in best_row.items() if key.startswith('metrics/')}}, ensure_ascii=False, indent=2))
comparison = {'training_parameters_matched': matched, 'same_data_split': True, 'test': evaluations, 'benchmark': json.loads((work / 'benchmark.json').read_text()), 's_training': summary}
(work / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2))

records = []


def add(source, destination):
    source = Path(source)
    require(source.is_file(), f'缺少产物：{source}')
    records.append({'source': str(source.relative_to(ROOT)), 'destination': destination, 'bytes': source.stat().st_size})


for path in sorted(run.rglob('*')):
    if not path.is_file():
        continue
    if path.suffix == '.pt':
        add(path, ('checkpoints/' if path.name.startswith('epoch') else 'weights/') + path.name)
    elif path.name == 'args.yaml':
        add(path, 'config/args.yaml')
    elif path.name == 'results.csv':
        add(path, 'metrics/results.csv')
for path in work.iterdir():
    if path.is_file() and path.name not in {'state.json', 'pipeline.log', 'backup_manifest.json'}:
        add(path, ('logs/' if path.suffix == '.log' else 'metrics/') + path.name)
for label in ('s', 'm'):
    add(ROOT / 'runs/evaluate' / f'{args.name}-{label}-test' / 'metrics.json', f'metrics/test-{label}.json')
for name in ['environment.json', 'server-requirements.lock.txt']:
    add(ROOT / 'work' / name, 'config/' + name)
for folder in ['scripts', 'configs']:
    for path in sorted((ROOT / folder).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            add(path, 'code/' + str(path.relative_to(ROOT)))
for name in ['requirements.txt', 'requirements.lock.txt', 'README.md']:
    add(ROOT / name, 'code/' + name)
for name in ['data.yaml', 'verification.json', 'report.json', 'low_iou_instances.json', 'conversion.jsonl']:
    add(ROOT / 'datasets/cholecinstanceseg/prepared' / name, 'data/' + name)
add(args.weights.resolve(), 'pretrained/' + args.weights.name)
add(args.weights.with_suffix('.source.json').resolve(), 'pretrained/source.json')
manifest = {'files': records, 'file_count': len(records), 'bytes': sum(record['bytes'] for record in records)}
(work / 'backup_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
state('completed', summary=summary, manifest='work/yolo26s-comparison/backup_manifest.json')
