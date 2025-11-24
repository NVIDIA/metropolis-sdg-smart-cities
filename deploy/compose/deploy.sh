#!/usr/bin/env bash
## SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: Apache-2.0
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
set -euo pipefail

#
# Deployment helper for two scenarios:
# 1) homogeneous: run NIM(s) and Workbench on the same node (default, no args)
# 2) heterogeneous:
#    - nim: run only NIM(s) on this node (prints NIM_HOST to use elsewhere)
#    - workbench: run only Workbench on another node; requires NIM_HOST to be set
# 3) cleanup: stop and remove all containers
#
# Usage:
#   ./deploy.sh                 # homogeneous (NIM + Workbench)
#   ./deploy.sh nim             # NIM-only on this node
#   ./deploy.sh workbench       # Workbench-only; requires NIM_HOST
#   ./deploy.sh cleanup         # Stop and remove all containers
#   DEPLOY_MODE=nim ./deploy.sh # alternative via env var

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-${DEPLOY_MODE:-homogeneous}}"

# Handle cleanup mode early to avoid env sourcing issues
if [ "$MODE" = "cleanup" ] || [ "$MODE" = "down" ] || [ "$MODE" = "stop" ]; then
  echo "[deploy] Mode: cleanup (stopping and removing all containers)"
  ENV_ARG=""
  [ -f env ] && ENV_ARG="--env-file env"
  echo "[deploy] Stopping NIM containers..."
  docker compose -f docker-compose.nim.yml $ENV_ARG down 2>/dev/null || true
  echo "[deploy] Stopping Workbench containers..."
  docker compose -f docker-compose.workbench.yml $ENV_ARG down 2>/dev/null || true
  echo "[deploy] Cleanup complete. All containers have been stopped and removed."
  exit 0
fi

source ./env 
LOCAL_UID=$(id -u)
LOCAL_GID=$(id -g)
export LOCAL_UID
export LOCAL_GID
mkdir -p $LOCAL_NIM_CACHE 
chown $LOCAL_UID:$LOCAL_GID $LOCAL_NIM_CACHE
chmod 755 $LOCAL_NIM_CACHE

usage() {
  echo "Usage: $0 [nim|workbench|cleanup]"
  exit 1
}

get_host_ip() {
  ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}'
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[deploy][error] Required command '$1' not found in PATH. Please install it first." >&2
    exit 1
  fi
}

require_docker_compose() {
  if ! docker compose version >/dev/null 2>&1; then
    echo "[deploy][error] 'docker compose' is not available. Please install a recent Docker Engine with Compose v2." >&2
    exit 1
  fi
}

require_docker_running() {
  if ! docker info >/dev/null 2>&1; then
    echo "[deploy][error] Docker daemon does not appear to be running or is not accessible for this user." >&2
    exit 1
  fi
}

check_gpu_availability() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[deploy][warn] nvidia-smi not found. GPU availability cannot be verified." >&2
    return 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[deploy][error] nvidia-smi failed. NVIDIA drivers may not be installed or GPU is not accessible." >&2
    exit 1
  fi
  local gpu_count
  gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if [ "$gpu_count" -eq 0 ]; then
    echo "[deploy][error] No GPUs detected. This deployment requires NVIDIA GPUs." >&2
    exit 1
  fi
  echo "[deploy] Found $gpu_count GPU(s)"
  return 0
}

check_nvidia_container_toolkit() {
  # Check if nvidia-container-runtime binary exists
  if command -v nvidia-container-runtime >/dev/null 2>&1 || [ -f /usr/bin/nvidia-container-runtime ]; then
    echo "[deploy] NVIDIA Container Toolkit found"
    return 0
  fi
  
  # Check if Docker info shows nvidia as available runtime
  if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "[deploy] NVIDIA Container Toolkit verified (via Docker runtime)"
    return 0
  fi
  
  echo "[deploy][error] NVIDIA Container Toolkit not found or not configured." >&2
  echo "[deploy][error] Please install and configure NVIDIA Container Toolkit:" >&2
  echo "[deploy][error]   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" >&2
  exit 1
}

check_port_availability() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    echo "[deploy][error] 'lsof' command not found. Please install lsof package:" >&2
    echo "[deploy][error]   Ubuntu/Debian: sudo apt-get install lsof" >&2
    echo "[deploy][error]   RHEL/CentOS: sudo yum install lsof" >&2
    exit 1
  fi
  if lsof -i ":$port" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

check_nvcr_login() {
  local docker_config
  docker_config="${DOCKER_CONFIG:-$HOME/.docker}/config.json"
  if [ -f "$docker_config" ]; then
    if grep -q '"nvcr.io"' "$docker_config"; then
      return 0
    fi
  fi
  return 1
}

require_nvcr_login() {
  if ! check_nvcr_login; then
    echo "[deploy][error] Not logged into nvcr.io. Please run:"
    echo "  docker login nvcr.io"
    echo "Use username 'oauthtoken' and your NGC API key as the password."
    echo "Alternatively, set NGC_API_KEY and login before re-running this script."
    exit 1
  fi
}

echo "[deploy] Running prerequisite checks..."
require_cmd docker
require_cmd lsof
require_docker_compose
require_docker_running

# Check GPU availability
check_gpu_availability || echo "[deploy][warn] GPU check skipped or failed"

# Check NVIDIA Container Toolkit
check_nvidia_container_toolkit

# Check port availability (only if env file exists)
if [ -f env ]; then
  set +u  # Temporarily allow unset variables
  source ./env 2>/dev/null || true
  set -u  # Re-enable strict mode
  
  port_conflicts=()
  
  # Determine which ports to check based on mode
  if [ "$MODE" = "homogeneous" ] || [ "$MODE" = "nim" ]; then
    if ! check_port_availability "${VLM_PORT:-8001}"; then port_conflicts+=("${VLM_PORT:-8001}"); fi
    if ! check_port_availability "${LLM_PORT:-8002}"; then port_conflicts+=("${LLM_PORT:-8002}"); fi
    if ! check_port_availability "${TRANSFER_GRADIO_PORT:-8080}"; then port_conflicts+=("${TRANSFER_GRADIO_PORT:-8080}"); fi
  fi
  if [ "$MODE" = "homogeneous" ] || [ "$MODE" = "workbench" ]; then
    if ! check_port_availability "${NOTEBOOK_PORT:-8888}"; then port_conflicts+=("${NOTEBOOK_PORT:-8888}"); fi
    if ! check_port_availability "${CARLA_PORT:-2000}"; then port_conflicts+=("${CARLA_PORT:-2000}"); fi
    if ! check_port_availability "${CARLA_STREAM_PORT_UDP:-2001}"; then port_conflicts+=("${CARLA_STREAM_PORT_UDP:-2001}"); fi
    if ! check_port_availability "${CARLA_STREAM_PORT_TCP:-2002}"; then port_conflicts+=("${CARLA_STREAM_PORT_TCP:-2002}"); fi
  fi
  
  if [ ${#port_conflicts[@]} -gt 0 ]; then
    echo "[deploy][error] Port conflict detected. The following ports are already in use:" >&2
    for port in "${port_conflicts[@]}"; do
      echo "[deploy][error]   Port $port" >&2
    done
    echo "[deploy][error] Please stop the conflicting services or update ports in 'env' file." >&2
    exit 1
  else
    echo "[deploy] Port availability check passed"
  fi
fi

case "$MODE" in
  homogeneous|homo|"")
    echo "[deploy] Mode: homogeneous (NIM + Workbench on this node)"
    require_nvcr_login
    mkdir -p $LOCAL_NIM_CACHE
    docker compose -f docker-compose.nim.yml --env-file env up -d
    export NIM_HOST="$(get_host_ip)"
    export CARLA_HOST="$(get_host_ip)"
    echo "[deploy] NIM_HOST: $NIM_HOST"
    echo "[deploy] CARLA_HOST: $CARLA_HOST"
    docker compose -f docker-compose.workbench.yml --env-file env up -d
    echo "All containers up, please note all the NIMs have additional startup time to initialize the model checkpoints ensure the endpoints are healthy before proceeding"
    ;;

  nim)
    echo "[deploy] Mode: heterogeneous (NIM-only on this node)"
    require_nvcr_login
    mkdir -p $LOCAL_NIM_CACHE
    docker compose -f docker-compose.nim.yml --env-file env up -d
    NIM_IP="$(get_host_ip)"
    echo "[deploy] NIM node ready. Set NIM_HOST=$NIM_IP on the workbench node."
    echo "All containers up, please note all the NIMs have additional startup time to initialize the model checkpoints ensure the endpoints are healthy before proceeding"
    ;;

  workbench)
    echo "[deploy] Mode: heterogeneous (Workbench-only on this node)"
    if ! check_nvcr_login; then
      echo "[deploy][warn] Not logged into nvcr.io. If your workbench Dockerfile uses NVCR base images, please login:"
      echo "  docker login nvcr.io (username: $oauthtoken; password: your NGC API key)"
    fi
    if [ -z "${NIM_HOST:-}" ]; then
      if [ -f env ]; then
        NIM_HOST=$(grep -E '^NIM_HOST=' env | tail -n1 | cut -d'=' -f2- || true)
        export NIM_HOST
      fi
    fi
    if [ -z "${NIM_HOST:-}" ]; then
      echo "[deploy][error] NIM_HOST not set. Set NIM_HOST in the environment or 'env' file before deploying Workbench."
      echo "Example: export NIM_HOST=<NIM_NODE_IP> && ./deploy.sh workbench"
      exit 1
    fi
    echo "[deploy] Using NIM_HOST: $NIM_HOST"
    # Confirm NIM_HOST looks correct to the user before proceeding
    read -r -p "[deploy] Is this the correct NIM host for your NIM node? [y/N]: " _ans
    case "${_ans:-N}" in
      y|Y|yes|YES)
        ;;
      *)
        echo "[deploy][info] Please edit 'deploy/compose/env' and set NIM_HOST to your NIM node IP, then re-run:"
        echo "  cd deploy/compose && vi env    # or your preferred editor"
        echo "  ./deploy.sh workbench"
        exit 1
        ;;
    esac
    docker compose -f docker-compose.workbench.yml --env-file env up -d
    ;;

  *)
    usage
    ;;
esac