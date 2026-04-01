#!/usr/bin/env bash
# setup_envs.sh — Create/update conda environments for the video translation pipeline.
#
# Usage:
#   bash scripts/setup_envs.sh              # set up all missing environments
#   bash scripts/setup_envs.sh latentsync   # set up a single environment
#   bash scripts/setup_envs.sh --list       # show status of all environments
#
# Hardware: NVIDIA RTX 5090 (sm_120), CUDA 13.2, PyTorch cu128
# ──────────────────────────────────────────────────────────────

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIP_INDEX="https://download.pytorch.org/whl/cu128"

# ── helpers ──────────────────────────────────────────────────

log()  { echo -e "\033[1;32m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*"; }
err()  { echo -e "\033[1;31m[error]\033[0m $*" >&2; }

env_exists() { conda env list | grep -qE "^$1\s"; }

create_env() {
    local name=$1 python=${2:-3.10}
    if env_exists "$name"; then
        warn "Environment '$name' already exists, skipping creation."
        return 0
    fi
    log "Creating conda env: $name (python=$python)"
    conda create -n "$name" python="$python" -y
}

# ── environment definitions ──────────────────────────────────

setup_latentsync() {
    create_env latentsync 3.10
    log "Installing LatentSync dependencies..."
    conda run -n latentsync --no-banner pip install \
        torch==2.5.1 torchvision torchaudio \
        --index-url "$PIP_INDEX"

    local req="$ROOT_DIR/module_D_diffusion/D1_latentsync/LatentSync/requirements.txt"
    if [[ -f "$req" ]]; then
        conda run -n latentsync --no-banner pip install -r "$req"
    else
        warn "LatentSync requirements.txt not found. Clone the repo first:"
        warn "  git clone https://github.com/bytedance/LatentSync.git \\"
        warn "    $ROOT_DIR/module_D_diffusion/D1_latentsync/LatentSync"
    fi

    log "LatentSync checkpoints:"
    log "  huggingface-cli download ByteDance/LatentSync-1.6 \\"
    log "    --local-dir $ROOT_DIR/module_D_diffusion/D1_latentsync/LatentSync/checkpoints"
}

setup_echomimic() {
    create_env echomimic 3.10
    log "Installing EchoMimic V2 dependencies..."
    conda run -n echomimic --no-banner pip install \
        torch==2.5.1 torchvision torchaudio xformers==0.0.28.post3 \
        --index-url "$PIP_INDEX"

    local req="$ROOT_DIR/module_D_diffusion/D2_echomimic/echomimic_v2/requirements.txt"
    if [[ -f "$req" ]]; then
        conda run -n echomimic --no-banner pip install -r "$req"
        conda run -n echomimic --no-banner pip install --no-deps facenet_pytorch==2.6.0
    else
        warn "EchoMimic V2 requirements.txt not found. Clone the repo first:"
        warn "  git clone https://github.com/BadToBest/EchoMimicV2.git \\"
        warn "    $ROOT_DIR/module_D_diffusion/D2_echomimic/echomimic_v2"
    fi

    log "EchoMimic V2 checkpoints:"
    log "  huggingface-cli download BadToBest/EchoMimicV2 \\"
    log "    --local-dir $ROOT_DIR/module_D_diffusion/D2_echomimic/echomimic_v2/pretrained_weights"
}

setup_fantasytalking() {
    create_env fantasytalking 3.10
    log "Installing FantasyTalking dependencies..."
    conda run -n fantasytalking --no-banner pip install \
        torch==2.4.0 torchvision torchaudio \
        --index-url "$PIP_INDEX"

    local req="$ROOT_DIR/module_D_diffusion/D3_fantasytalking/fantasy-talking/requirements.txt"
    if [[ -f "$req" ]]; then
        conda run -n fantasytalking --no-banner pip install -r "$req"
    else
        warn "FantasyTalking requirements.txt not found. Clone the repo first:"
        warn "  git clone https://github.com/acvlab/FantasyTalking.git \\"
        warn "    $ROOT_DIR/module_D_diffusion/D3_fantasytalking/fantasy-talking"
    fi

    log "FantasyTalking checkpoints (Wan2.1 14B, ~28 GB):"
    log "  huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \\"
    log "    --local-dir $ROOT_DIR/module_D_diffusion/D3_fantasytalking/fantasy-talking/models/Wan2.1-I2V-14B-720P"
    log "  huggingface-cli download acvlab/FantasyTalking fantasytalking_model.ckpt \\"
    log "    --local-dir $ROOT_DIR/module_D_diffusion/D3_fantasytalking/fantasy-talking/models"
}

# ── status listing ───────────────────────────────────────────

list_envs() {
    local envs=(vhap gaussian-avatars cosyvoice funasr sam2 propainter \
                latentsync echomimic fantasytalking codeformer syncnet)
    printf "%-20s %s\n" "Environment" "Status"
    printf "%-20s %s\n" "───────────" "──────"
    for e in "${envs[@]}"; do
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
    # Set up a single environment
    case "$1" in
        latentsync)     setup_latentsync ;;
        echomimic)      setup_echomimic ;;
        fantasytalking) setup_fantasytalking ;;
        *)
            err "Unknown environment: $1"
            err "Supported: latentsync, echomimic, fantasytalking"
            exit 1
            ;;
    esac
else
    # Set up all missing environments
    log "Setting up missing diffusion environments..."
    env_exists latentsync     || setup_latentsync
    env_exists echomimic      || setup_echomimic
    env_exists fantasytalking || setup_fantasytalking
    log "Done. Run 'bash scripts/setup_envs.sh --list' to verify."
fi
