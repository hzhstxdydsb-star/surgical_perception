set -euo pipefail
cd "$(dirname "$0")/.."
test "$(uname -s)" = Linux
mkdir -p work
export TMPDIR="$PWD/work"
export PIP_CACHE_DIR="$PWD/work/pip-cache"
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt -c requirements.lock.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pip freeze > work/server-requirements.lock.txt
