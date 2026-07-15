#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-run_metadata}"
mkdir -p "${OUTPUT_DIR}"

git rev-parse HEAD > "${OUTPUT_DIR}/workspace_git_commit.txt"
python3 --version > "${OUTPUT_DIR}/python_version.txt" 2>&1
python3 -m pip freeze > "${OUTPUT_DIR}/pip_freeze.txt"
nvidia-smi -q > "${OUTPUT_DIR}/nvidia_smi.txt"

if [[ -d third_party/Seg-Zero/.git ]]; then
  git -C third_party/Seg-Zero rev-parse HEAD > "${OUTPUT_DIR}/segzero_git_commit.txt"
fi
if [[ -d third_party/verl/.git ]]; then
  git -C third_party/verl rev-parse HEAD > "${OUTPUT_DIR}/verl_git_commit.txt"
fi

echo "Environment metadata written to ${OUTPUT_DIR}"
