#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-${ROOT_DIR}/third_party}"
SEGZERO_DIR="${SEGZERO_DIR:-${THIRD_PARTY_DIR}/Seg-Zero}"
VERL_DIR="${VERL_DIR:-${THIRD_PARTY_DIR}/verl}"
SEGZERO_REF="${SEGZERO_REF:-5507720}"
VERL_REF="${VERL_REF:-v0.6.0}"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${SEGZERO_DIR}/.git" ]]; then
  git clone https://github.com/JIA-Lab-research/Seg-Zero.git "${SEGZERO_DIR}"
fi
git -C "${SEGZERO_DIR}" fetch origin
git -C "${SEGZERO_DIR}" checkout --detach "${SEGZERO_REF}"

if [[ ! -d "${VERL_DIR}/.git" ]]; then
  git clone --branch "${VERL_REF}" --depth 1 https://github.com/verl-project/verl.git "${VERL_DIR}"
fi
git -C "${VERL_DIR}" fetch origin "${VERL_REF}"
git -C "${VERL_DIR}" checkout --detach "${VERL_REF}"

cp -a "${ROOT_DIR}/overlays/verl_v0_6_0/." "${VERL_DIR}/"

echo "Seg-Zero: $(git -C "${SEGZERO_DIR}" rev-parse HEAD)"
echo "veRL:     $(git -C "${VERL_DIR}" rev-parse HEAD) + local propose/refine overlay"
echo "Upstreams are ready under ${THIRD_PARTY_DIR}"
