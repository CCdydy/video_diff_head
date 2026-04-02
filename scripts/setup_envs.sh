#!/usr/bin/env bash
# setup_envs.sh — Create conda environments for the video translation pipeline.
#
# Usage:
#   bash scripts/setup_envs.sh              # set up all missing environments
#   bash scripts/setup_envs.sh wan_audio    # set up a single environment
#   bash scripts/setup_envs.sh --list       # show status
#
# Hardware: RTX 5090, CUDA 13.2, PyTorch cu128
# ──────────────────────────────────────────────────────────────

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIP_INDEX="https://download.pytorch.org/whl/cu128"

log()  { echo -e "\033[1;32m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*"; }

env_exists() { conda env list | grep -qE "^$1\s"; }

create_env() {
    local name=$1 python=${2:-3.10}
    if env_exists "$name"; then
        warn "Environment '$name' already exists, skipping."
        return 0
    fi
    log "Creating conda env: $name (python=$python)"
    conda create -n "$name" python="$python" -y
}

# ── wan_audio: FantasyTalking + VACE 共用 ────────────────────

setup_wan_audio() {
    create_env wan_audio 3.10

    local third_party_bootstrap="$ROOT_DIR/scripts/setup_third_party.sh"
    if [[ -f "$third_party_bootstrap" ]]; then
        log "Bootstrapping third-party source trees..."
        bash "$third_party_bootstrap"
    fi

    log "Installing PyTorch (cu128)..."
    conda run -n wan_audio --live-stream pip install \
        torch torchvision torchaudio \
        --index-url "$PIP_INDEX"

    log "Installing diffusers + transformers + accelerate..."
    conda run -n wan_audio --live-stream pip install \
        diffusers transformers accelerate

    local req="$ROOT_DIR/third_party/Wan2.1/requirements.txt"
    if [[ -f "$req" ]]; then
        log "Installing Wan2.1 requirements..."
        conda run -n wan_audio --live-stream pip install -r "$req"
        log "Installing local Wan2.1 package..."
        conda run -n wan_audio --live-stream pip install -e "$ROOT_DIR/third_party/Wan2.1"
    else
        warn "Wan2.1 source tree not ready. After bootstrapping, run:"
        warn "  conda run -n wan_audio pip install -r $req"
    fi

    log ""
    log "wan_audio ready. Download model weights:"
    log "  # Wan2.1-VACE-14B"
    log "  huggingface-cli download Wan-AI/Wan2.1-VACE-14B \\"
    log "    --local-dir $ROOT_DIR/data/models/Wan2.1-VACE-14B"
    log "  # wav2vec2"
    log "  huggingface-cli download facebook/wav2vec2-base-960h \\"
    log "    --local-dir $ROOT_DIR/data/models/wav2vec2-base-960h"
    log "  # FantasyTalking adapter"
    log "  huggingface-cli download acvlab/FantasyTalking fantasytalking_model.ckpt \\"
    log "    --local-dir $ROOT_DIR/data/models"
    log "  # SAM2"
    log "  huggingface-cli download facebook/sam2-hiera-large \\"
    log "    --local-dir $ROOT_DIR/data/models"
}

# ── status ───────────────────────────────────────────────────

ALL_ENVS=(vhap gaussian-avatars cosyvoice funasr sam2 propainter \
          wan_audio codeformer syncnet)

list_envs() {
    printf "%-20s %s\n" "Environment" "Status"
    printf "%-20s %s\n" "───────────" "──────"
    for e in "${ALL_ENVS[@]}"; do
        if env_exists "$e"; then
            printf "%-20s \033[32m✅ Installed\033[0m\n" "$e"
        else
            printf "%-20s \033[33m❌ Missing\033[0m\n" "$e"
        fi
    done
}

# ── main ─────────────────────────────────────────────────────

if [[ "${1:-}" == "--list" ]]; then
    list_envs
    exit 0
fi

if [[ -n "${1:-}" ]]; then
    case "$1" in
        wan_audio) setup_wan_audio ;;
        *)
            echo "Unknown environment: $1"
            echo "Supported: wan_audio"
            exit 1 ;;
    esac
else
    log "Setting up missing environments..."
    env_exists wan_audio || setup_wan_audio
    log "Done. Run 'bash scripts/setup_envs.sh --list' to check."
fi
