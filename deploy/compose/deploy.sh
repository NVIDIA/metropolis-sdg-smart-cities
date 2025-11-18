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
#
# Usage:
#   ./deploy.sh                 # homogeneous (NIM + Workbench)
#   ./deploy.sh nim             # NIM-only on this node
#   ./deploy.sh workbench       # Workbench-only; requires NIM_HOST
#   DEPLOY_MODE=nim ./deploy.sh # alternative via env var

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ./env 
LOCAL_UID=$(id -u)
LOCAL_GID=$(id -g)
export LOCAL_UID
export LOCAL_GID
mkdir -p $LOCAL_NIM_CACHE 
chown $LOCAL_UID:$LOCAL_GID $LOCAL_NIM_CACHE
chmod 755 $LOCAL_NIM_CACHE

MODE="${1:-${DEPLOY_MODE:-homogeneous}}"

usage() {
  echo "Usage: $0 [nim|workbench]"
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
require_docker_compose
require_docker_running

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
      echo "  docker login nvcr.io (username: oauthtoken; password: your NGC API key)"
    fi
    if [ -z "${NIM_HOST:-}" ]; then
      if [ -f env ]; then
        NIM_HOST=$(grep -E '^NIM_HOST=' env | tail -n1 | cut -d'=' -f2- || true)
        export NIM_HOST
      fi
    fi
    if [ -z "${NIM_HOST:-}" ]; then
      echo "[deploy][error] NIM_HOST not set. Set NIM_HOST in the environment or 'env' file before deploying Workbench."
      echo "Example: export NIM_HOST=<NIM_NODE_IP> && ./deploy.sh het-workbench"
      exit 1
    fi
    echo "[deploy] Using NIM_HOST: $NIM_HOST"
    docker compose -f docker-compose.workbench.yml --env-file env up -d
    ;;

  *)
    usage
    ;;
esac