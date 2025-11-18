#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CARLA Ground Truth Generation - Batch Processing Script
# Purpose: Process multiple scenarios with matching configuration, recording, and camera files
# Usage: ./batch_processing.sh [directory]
# Default directory: example_data/test

# Set strict error handling
set -euo pipefail

# Check environment variables
if [ -z "${SCENARIO_DIR:-}" ]; then
    echo "Error: SCENARIO_DIR is not set."
    exit 1
fi
if [ -z "${RUN_ID:-}" ]; then
    echo "Error: RUN_ID is not set."
    exit 1
fi

# Initialize counters
TOTAL_SCENARIOS=0
SUCCESSFUL_SCENARIOS=0
FAILED_SCENARIOS=0

# Print header
echo "================================================================================"
echo "CARLA Ground Truth Generation - Batch Processing"
echo "================================================================================"
echo "Processing Directory: $SCENARIO_DIR"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# Validate directory exists
if [ ! -d "$SCENARIO_DIR" ]; then
    echo "[ERROR] Directory does not exist: $SCENARIO_DIR"
    echo "        Please provide a valid directory containing scenario files."
    exit 1
fi

# Count total JSON files
JSON_COUNT=$(find "$SCENARIO_DIR" -name "*.json" -type f | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo "[WARNING] No JSON configuration files found in: $SCENARIO_DIR"
    echo "          Expected files with pattern: {identifier}.json"
    exit 0
fi

echo "[INFO] Found $JSON_COUNT configuration file(s) to process"
echo ""

for json_file in "$SCENARIO_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        TOTAL_SCENARIOS=$((TOTAL_SCENARIOS + 1))

        # Extract scenario identifier from filename
        filename=$(basename "$json_file" .json)
        log_file="$SCENARIO_DIR/${filename}.log"
        yaml_file="$SCENARIO_DIR/${filename}.yaml"

        echo "[INFO] Running scenario: $filename"

        set +e
        python /workspace/modules/carla-ground-truth-generation/main.py \
            --config "$json_file" \
            --recorder-filename "$log_file" \
            --camera-config "$yaml_file" \
            --wf-config /tmp/wf-config.json \
            --output-dir "/workspace/data/outputs/CARLA/$RUN_ID/scenario_${filename}" \
            --class-filter-config /workspace/data/examples/filter_semantic_classes.yaml \
            --target-fps 30
        exit_code=$?
        set -e

        # Report result
        if [ $exit_code -eq 0 ]; then
            echo "[SUCCESS] Scenario $filename completed successfully"
            echo "[INFO] Output saved to: /workspace/data/outputs/CARLA/$RUN_ID/scenario_${filename}/"
            SUCCESSFUL_SCENARIOS=$((SUCCESSFUL_SCENARIOS + 1))
        else
            echo "[FAILED] Scenario $filename failed with exit code: $exit_code"
            FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
        fi
    fi
done

# Print summary report
echo ""
echo "================================================================================"
echo "BATCH PROCESSING SUMMARY"
echo "================================================================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Total Scenarios:  $TOTAL_SCENARIOS"
echo "Successful:       $SUCCESSFUL_SCENARIOS"
echo "Failed:           $FAILED_SCENARIOS"
echo ""

# Set appropriate exit code
if [ $FAILED_SCENARIOS -gt 0 ]; then
    echo "[RESULT] Batch processing completed with errors"
    exit 1
else
    echo "[RESULT] Batch processing completed successfully"
    exit 0
fi