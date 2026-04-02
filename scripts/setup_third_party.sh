#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WAN_DIR="$ROOT_DIR/third_party/Wan2.1"
PATCH_FILE="$ROOT_DIR/patches/wan2.1-local.patch"

echo "[third_party] Initializing submodules..."
git -C "$ROOT_DIR" submodule update --init --recursive

if [[ ! -d "$WAN_DIR" ]]; then
    echo "[third_party] Wan2.1 submodule not found: $WAN_DIR" >&2
    exit 1
fi

if [[ -f "$PATCH_FILE" ]]; then
    if git -C "$WAN_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
        echo "[third_party] Applying Wan2.1 local patch..."
        git -C "$WAN_DIR" apply "$PATCH_FILE"
    elif git -C "$WAN_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
        echo "[third_party] Wan2.1 patch already applied."
    else
        echo "[third_party] Wan2.1 patch does not apply cleanly. Check submodule commit." >&2
        exit 1
    fi
fi

echo "[third_party] Ready."
