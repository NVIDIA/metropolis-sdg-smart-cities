#!/bin/bash
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

# Script to process multiple camera configurations for OVDG export
# Usage: ./process_multi_camera.sh <log_file> <output_base_dir> [additional_args]

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <log_file> <output_base_dir> [additional_args]"
    echo "Example: $0 example_data/recording.log multi_camera_output -s 0.0 -d 0.5"
    exit 1
fi

LOG_FILE="$1"
OUTPUT_BASE_DIR="$2"
shift 2  # Remove first two arguments, keep the rest for additional args

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_DIR="${SCRIPT_DIR}/config/scene1"

# Create base output directory
mkdir -p "${OUTPUT_BASE_DIR}"

# Create a summary file
SUMMARY_FILE="${OUTPUT_BASE_DIR}/processing_summary.txt"
echo "Multi-Camera OVDG Export Summary" > "${SUMMARY_FILE}"
echo "================================" >> "${SUMMARY_FILE}"
echo "Log file: ${LOG_FILE}" >> "${SUMMARY_FILE}"
echo "Processing started: $(date)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Process each camera configuration
for cam_config in ${CONFIG_DIR}/cam*.yaml; do
    if [ -f "$cam_config" ]; then
        # Extract camera name (e.g., cam1 from cam1.yaml)
        cam_name=$(basename "$cam_config" .yaml)
        
        echo "Processing ${cam_name}..."
        echo "Camera: ${cam_name}" >> "${SUMMARY_FILE}"
        echo "Config: ${cam_config}" >> "${SUMMARY_FILE}"
        
        # Create output directory for this camera
        cam_output_dir="${OUTPUT_BASE_DIR}/${cam_name}"
        
        # Run main.py for this camera with automatic video generation
        python main.py \
            --camera-config "${cam_config}" \
            -f "${LOG_FILE}" \
            -o "${cam_output_dir}" \
            --generate-videos \
            "$@"  # Pass any additional arguments
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✓ ${cam_name} processed successfully" >> "${SUMMARY_FILE}"
            
            # Count output files
            if [ -d "${cam_output_dir}" ]; then
                num_ovdg=$(find "${cam_output_dir}" -name "ovdg_*.json" | wc -l)
                num_rgb=$(find "${cam_output_dir}/rgb" -name "*.jpg" 2>/dev/null | wc -l)
                echo "  - OVDG files: ${num_ovdg}" >> "${SUMMARY_FILE}"
                echo "  - RGB frames: ${num_rgb}" >> "${SUMMARY_FILE}"
                
                # Check if videos were generated automatically
                if [ -d "${cam_output_dir}/videos" ]; then
                    num_videos=$(find "${cam_output_dir}/videos" -name "*.mp4" | wc -l)
                    echo "  - Videos created automatically: ${num_videos} files" >> "${SUMMARY_FILE}"
                    echo "  - Video location: ${cam_output_dir}/videos/" >> "${SUMMARY_FILE}"
                else
                    echo "  - No videos directory found (may have been skipped or failed)" >> "${SUMMARY_FILE}"
                fi
            fi
        else
            echo "✗ ${cam_name} processing failed" >> "${SUMMARY_FILE}"
        fi
        echo "" >> "${SUMMARY_FILE}"
    fi
done

echo "Processing completed: $(date)" >> "${SUMMARY_FILE}"

# Create a directory structure visualization
echo "" >> "${SUMMARY_FILE}"
echo "Output Directory Structure:" >> "${SUMMARY_FILE}"
echo "===========================" >> "${SUMMARY_FILE}"
tree -L 3 "${OUTPUT_BASE_DIR}" >> "${SUMMARY_FILE}" 2>/dev/null || ls -la "${OUTPUT_BASE_DIR}" >> "${SUMMARY_FILE}"

echo "Multi-camera processing complete!"
echo "Summary saved to: ${SUMMARY_FILE}"
