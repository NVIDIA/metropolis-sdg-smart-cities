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

import carla
import argparse
import numpy as np
import math
import os
import sys
import queue
import cv2 
import json
import random
import logging
import numpy as np
import yaml 
import matplotlib.pyplot as plt
from frames_to_videos import process_scene_directory
import subprocess
import tempfile
import re

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Image quality and format settings
JPEG_QUALITY = 95
FRAME_NUMBER_PADDING = 12  # Zero-padding for frame numbers (e.g., 000000000001)

# Instance colormap generation
COLORMAP_SIZE = 65536
COLORMAP_SEED = 140

# Edge detection parameters
CANNY_THRESHOLD_LOW = 100
CANNY_THRESHOLD_HIGH = 200
POLYGON_EPSILON = 2.0  # Polygon approximation accuracy

# Text overlay settings
LABEL_FONT_SCALE = 0.6
LABEL_FONT_THICKNESS = 2
LABEL_PADDING = 2  # Padding around text background

# OpenDRIVE map generation parameters
OPENDRIVE_VERTEX_DISTANCE = 2.0  # meters
OPENDRIVE_MAX_ROAD_LENGTH = 500.0  # meters
OPENDRIVE_WALL_HEIGHT = 0.0  # meters
OPENDRIVE_EXTRA_WIDTH = 0.6  # meters

# Coordinate conversion
UNREAL_TO_METER_SCALE = 100.0  # Unreal units to meters

# Depth encoding constants (CARLA-specific)
DEPTH_R_WEIGHT = 65536.0
DEPTH_G_WEIGHT = 256.0
DEPTH_B_WEIGHT = 1.0
DEPTH_NORMALIZATION = 16777215.0  # 256^3 - 1

# Progress logging
PROGRESS_LOG_INTERVAL = 100  # Log progress every N frames

# Default values
DEFAULT_TARGET_FPS = 30
DEFAULT_LIMIT_DISTANCE = 100
DEFAULT_AREA_THRESHOLD = 100

# 3D bounding box edges - defines which vertices connect to form the bbox wireframe
EDGES = [
    [0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
    [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]
]

# CARLA semantic segmentation label mappings to human-readable names and visualization colors (RGB)
SEMANTIC_MAP = {
    0: ('unlabelled', (0, 0, 0)),
    1: ('road', (128, 64, 0)),
    2: ('sidewalk', (244, 35, 232)),
    3: ('building', (70, 70, 70)),
    4: ('wall', (102, 102, 156)),
    5: ('fence', (190, 153, 153)),
    6: ('pole', (153, 153, 153)),
    7: ('traffic light', (250, 170, 30)),
    8: ('traffic sign', (220, 220, 0)),
    9: ('vegetation', (107, 142, 35)),
    10: ('terrain', (152, 251, 152)),
    11: ('sky', (70, 130, 180)),
    12: ('pedestrian', (220, 20, 60)),
    13: ('rider', (255, 0, 0)),
    14: ('car', (0, 0, 142)),
    15: ('truck', (0, 0, 70)),
    16: ('bus', (0, 60, 100)),
    17: ('train', (0, 80, 100)),
    18: ('motorcycle', (0, 0, 230)),
    19: ('bicycle', (119, 11, 32)),
    20: ('static', (110, 190, 160)),
    21: ('dynamic', (170, 120, 50)),
    22: ('other', (55, 90, 80)),
    23: ('water', (45, 60, 150)),
    24: ('road line', (157, 234, 50)),
    25: ('ground', (81, 0, 81)),
    26: ('bridge', (150, 100, 100)),
    27: ('rail track', (230, 150, 140)),
    28: ('guard rail', (180, 165, 180))
}

def create_instance_colormap(size=COLORMAP_SIZE, seed=COLORMAP_SEED):
    """Create a shuffled colormap for instance segmentation visualization."""
    np.random.seed(seed)
    base_colors = plt.get_cmap('prism')(np.linspace(0, 1, size))[:, :3]
    indices = np.arange(size)
    shuffled = np.concatenate(([0], np.random.permutation(indices[1:])))
    colormap = (base_colors[shuffled] * 255).astype(np.uint8)
    colormap[0] = [0, 0, 0]  # Ensure background (ID 0) is always black
    return colormap

# Pre-generate consistent colormap for instance segmentation visualization
INSTANCE_COLORMAP = create_instance_colormap()

# Global class filter lists - populated from YAML config at runtime
CLASSES_TO_KEEP_SHADED_SEG = []
CLASSES_TO_KEEP_CANNY = []

def load_class_filter_config(path: str):
    """Load class filter configuration from YAML file."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    global CLASSES_TO_KEEP_SHADED_SEG, CLASSES_TO_KEEP_CANNY
    CLASSES_TO_KEEP_SHADED_SEG = config.get('shaded_segmentation_classes', [])
    CLASSES_TO_KEEP_CANNY = config.get('canny_classes', [])

def run_collision_detection(recorder_filepath, host='127.0.0.1', port=2000, actor_ids=None, debug=False):
    """
    Run collision detection using sdg_collisions.py script.
    Returns a dictionary mapping frame times to collision events.
    
    Args:
        recorder_filepath: Path to the CARLA recording file
        host: CARLA server host
        port: CARLA server port
        actor_ids: List of actor IDs to filter collisions (or single int, or None for all)
        debug: Enable debug logging
    """
    # Locate the sdg_collisions.py script in the project directory structure
    script_path = os.path.join(os.path.dirname(__file__), 'collisions_sdg', 'sdg_collisions.py')
    if not os.path.exists(script_path):
        # Fallback to parent directory structure if not found locally
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'collisions_sdg', 'sdg_collisions.py')
    
    if not os.path.exists(script_path):
        logging.warning(f"Could not find sdg_collisions.py script - collision detection disabled")
        return {}
    
    # Create temporary file to capture collision detection output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_filename = tmp_file.name
    
    try:
        # Build subprocess command with recording file and server connection params
        cmd = [
            sys.executable,  # Ensure we use the same Python interpreter as this script
            script_path,
            '-f', recorder_filepath,
            '--host', host,
            '-p', str(port)
        ]
        
        # Handle actor ID filtering - supports list, single int, or None (all actors)
        if actor_ids is not None:
            # Normalize single int to list for consistent processing
            if isinstance(actor_ids, int):
                actor_ids = [actor_ids]
            # Append each actor ID as a separate command argument
            for actor_id in actor_ids:
                cmd.extend(['--id', str(actor_id)])
        
        if debug:
            cmd.append('-v')
        
        # Execute the collision detection subprocess
        logging.info(f"Running collision detection: {' '.join(cmd)}")
        
        # Switch to temp directory since sdg_collisions.py writes events.json to CWD
        original_dir = os.getcwd()
        temp_dir = os.path.dirname(tmp_filename)
        os.chdir(temp_dir)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        os.chdir(original_dir)
        
        if result.returncode != 0:
            logging.error(f"Collision detection failed: {result.stderr}")
            return {}
        
        # Parse the generated events.json file containing collision data
        events_file = os.path.join(temp_dir, 'events.json')
        collision_events = {}
        
        if os.path.exists(events_file):
            with open(events_file, 'r') as f:
                events_data = json.load(f)
            
            # Index events by start_time for efficient frame-time lookups during replay
            for event in events_data:
                start_time = float(event['start_time'])
                collision_events[start_time] = event
            
            logging.info(f"Found {len(collision_events)} collision events")
            
            # Clean up temporary events file
            os.remove(events_file)
        else:
            logging.info("No collisions detected in the recording")
        
        return collision_events
        
    finally:
        # Clean up temporary file created for subprocess coordination
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

def run_non_collision_sdg(odvg_dir, output_dir, num_closest=1, num_random=1, num_close_colli=1):
    """Run non-collision SDG to generate hard sample events."""
    script_path = os.path.join(os.path.dirname(__file__), 'non_collisions_sdg', 'sdg_non_collisions.py')
    if not os.path.exists(script_path):
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'non_collisions_sdg', 'sdg_non_collisions.py')
    
    if not os.path.exists(script_path):
        logging.warning(f"sdg_non_collisions.py not found - skipping non-collision SDG")
        return
    
    try:
        cmd = [sys.executable, script_path, '--odvg_dir', odvg_dir, '--output_dir', output_dir,
               '--num_closest_event', str(num_closest), '--num_random_event', str(num_random),
               '--num_close_colli_obj_event', str(num_close_colli)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            if result.stdout:
                logging.debug(result.stdout)
            logging.info("Non-collision SDG completed successfully")
        else:
            logging.error(f"Non-collision SDG failed: {result.stderr}")
    except Exception as e:
        logging.error(f"Non-collision SDG error: {e}")

def load_config(config_path: str, args):
    """Load configuration from JSON file and override command-line arguments."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override command-line args with config file values (config file takes precedence)
    # Basic connection parameters
    if 'host' in config:
        args.host = config['host']
    if 'port' in config:
        args.port = config['port']
    if 'timeout' in config:
        args.timeout = config['timeout']
    
    # Camera configuration file path
    if 'camera_config' in config:
        args.camera_config = config['camera_config']
    
    # Recording parameters
    if 'recorder_filename' in config:
        args.recorder_filename = config['recorder_filename']
    if 'start_time' in config:
        args.start = config['start_time']
    if 'duration' in config:
        args.duration = config['duration']
    if 'time_factor' in config:
        args.time_factor = config['time_factor']
    
    # Output parameters
    if 'output_dir' in config:
        args.output_dir = config['output_dir']
    if 'generate_videos' in config:
        args.generate_videos = config['generate_videos']
    
    # Processing parameters
    if 'limit_distance' in config:
        args.limit_distance = config['limit_distance']
    if 'area_threshold' in config:
        args.area_threshold = config['area_threshold']
    if 'class_filter_config' in config:
        args.class_filter_config = config['class_filter_config']
    
    # Map parameters
    if 'xodr_path' in config:
        args.xodr_path = config['xodr_path']
    
    # Replay parameters
    if 'camera_follow_actor' in config:
        args.camera = config['camera_follow_actor']
    if 'ignore_hero' in config:
        args.ignore_hero = config['ignore_hero']
    if 'move_spectator' in config:
        args.move_spectator = config['move_spectator']
    
    # Collision detection parameters
    if 'detect_collisions' in config:
        args.detect_collisions = config['detect_collisions']
    if 'collision_actor_ids' in config:
        args.collision_actor_ids = config['collision_actor_ids']
    
    logging.info(f"Loaded configuration from {config_path}")

def load_workflow_config(wf_config_path: str, args):
    """Load workflow configuration from wf-config.json. CLI arguments take precedence."""
    with open(wf_config_path, 'r') as f:
        wf_config = json.load(f)
    
    # Apply workflow config only for args not explicitly set via CLI (priority: CLI > wf-config > defaults)
    # Process all workflow configuration fields
    for key in wf_config.keys():
        if hasattr(args, key):
            value = extract_value(key, getattr(args, key, None), wf_config)
            setattr(args, key, value)
    logging.info(f"Loaded workflow configuration from {wf_config_path}")

def masked_edges_from_semseg(
    rgb_img: np.ndarray,
    semseg_img: np.ndarray,
    classes: list,
    gaussian_kernel=(5, 5),
    gaussian_sigma=1.0,
    canny_thresh1=CANNY_THRESHOLD_LOW,
    canny_thresh2=CANNY_THRESHOLD_HIGH,
):
    """Generate masked RGB and edge detection from semantic segmentation."""
    blurred_rgb = cv2.GaussianBlur(rgb_img, gaussian_kernel, gaussian_sigma)
    mask = np.zeros(semseg_img.shape[:2], dtype=np.uint8)
    for color in classes:
        lower = np.array(color, dtype=np.uint8)
        upper = np.array(color, dtype=np.uint8)
        mask |= cv2.inRange(semseg_img, lower, upper)
    mask_bool = mask.astype(bool)
    masked_rgb = np.zeros_like(rgb_img)
    masked_rgb[mask_bool] = blurred_rgb[mask_bool]
    gray = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    return masked_rgb, edges

def extract_between(input_string, left_delim, right_delim):
    try:
        start = input_string.index(left_delim) + len(left_delim)
        end = input_string.index(right_delim, start)
        return input_string[start:end]
    except ValueError:
        return None

def parse_frames_duration(info):
    frames = extract_between(info, "Frames: ", "\n")
    duration = extract_between(info, "Duration: ", " seconds")

    if frames and duration:
        return int(frames), float(duration)
    else:
        return -1, -1.0

def draw_bbox2d_on_frame(frame_img, bbox_mask_data, color=(0, 0, 255), thickness=2):
    """
    Draw 2D bounding box on frame image using frame_bboxes format
    
    Args:
        frame_img: OpenCV image (BGR format)
        bbox_mask_data: dictionary with '2d' key containing bbox information
        color: tuple (B, G, R) for box color
        thickness: int, line thickness
    
    Returns:
        frame_img: Image with bounding box drawn
    """
    if bbox_mask_data is None or '2d' not in bbox_mask_data or bbox_mask_data['2d'] is None:
        return frame_img
    
    bbox_mask_data_2d = bbox_mask_data['2d']
    x_min, y_min, x_max, y_max = bbox_mask_data_2d['bbox_2d']
    semantic_label = bbox_mask_data_2d['semantic_label']
    actor_id = bbox_mask_data_2d['actor_id']
    
    # Standardize bbox color to red for consistent visualization
    box_color = (0, 0, 255)  # Red in BGR format
    
    if semantic_label in SEMANTIC_MAP:
        label_name, semantic_color = SEMANTIC_MAP[semantic_label]
    else:
        label_name = f"class_{semantic_label}"
    
    # Draw bounding box rectangle
    cv2.rectangle(frame_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, thickness)
    
    # Prepare actor ID label with font styling
    label = f"{actor_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate text dimensions for proper centering
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
    )
    
    # Calculate bbox center point for label placement
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    text_x = center_x - text_width // 2      # Horizontal centering
    text_y = center_y + text_height // 2     # Vertical centering (baseline alignment)
    
    # Draw solid background for text readability
    cv2.rectangle(frame_img, 
                  (text_x - LABEL_PADDING, text_y - text_height - LABEL_PADDING),
                  (text_x + text_width + LABEL_PADDING, text_y + LABEL_PADDING),
                  box_color, -1)
                  
    # Render white text on colored background
    cv2.putText(frame_img, label, 
                (text_x, text_y), 
                font, LABEL_FONT_SCALE, (255, 255, 255), LABEL_FONT_THICKNESS)
    
    return frame_img

def save_debug_frame_with_bboxes(frame_img, frame_bboxes_masks, frame_num, debug_dir, prefix_filename='bboxs'):
    """
    Save frame with drawn bounding boxes to debug folder using frame_bboxes format
    Args:
        frame_img: OpenCV image (BGR format)
        frame_bboxes_masks: list of bbox and mask dictionaries with '2d'
        frame_num: int, frame number
        debug_dir: string, path to debug directory
    """
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)
    
    # Copy frame to avoid modifying original
    debug_frame = frame_img.copy()

    # Draw bounding boxes for all detections
    for bbox_mask_data in frame_bboxes_masks:
        if bbox_mask_data is not None:
            debug_frame = draw_bbox2d_on_frame(debug_frame, bbox_mask_data)
    
    # Save debug frame
    debug_filename = os.path.join(debug_dir, f"{prefix_filename}_{frame_num:012d}.jpg")
    cv2.imwrite(debug_filename, debug_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    
    return debug_filename

def mask_to_polygon_json(mask):
    """Convert binary mask to JSON-serializable polygon format"""
    if mask is None:
        return []
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return []
    
    polygons = []
    for contour in contours:
        # Simplify the contour to reduce points
        approx = cv2.approxPolyDP(contour, POLYGON_EPSILON, True)
        
        # Convert to flat list format [x1, y1, x2, y2, x3, y3, ...]
        if len(approx) >= 3:  # Need at least 3 points for a polygon
            polygon = approx.reshape(-1, 2).flatten()
            # Convert to regular Python floats (JSON serializable)
            polygon_list = [float(coord) for coord in polygon]
            polygons.append(polygon_list)
    
    return polygons

def draw_polygon_mask(frame_img, polygons, color=(0, 255, 255), alpha=0.3, thickness=2):
    """Draw polygon masks on frame with fill and outline"""
    if not polygons:
        return frame_img
    
    overlay = frame_img.copy()
    
    for polygon in polygons:
        if len(polygon) >= 6:  # At least 3 points (x,y pairs)
            # Reshape flat list to points
            points = np.array(polygon).reshape(-1, 2).astype(np.int32)
            
            # Fill polygon on overlay
            cv2.fillPoly(overlay, [points], color)
            
            # Draw polygon outline on original frame
            cv2.polylines(frame_img, [points], isClosed=True, color=color, thickness=thickness)
    
    # Blend overlay with original frame for transparency
    frame_img = cv2.addWeighted(frame_img, 1-alpha, overlay, alpha, 0)
    return frame_img

def draw_bbox3d_on_frame(frame_img, bbox_3d_data, color=(0, 0, 255), thickness=2):
    """
    Draw 3D bounding box projection on frame image
    
    Args:
        frame_img: OpenCV image (BGR format)
        bbox_3d_data: dictionary with '3d' key containing bbox information including 'projection'
        color: tuple (B, G, R) for box color
        thickness: int, line thickness
    
    Returns:
        frame_img: Image with 3D bounding box drawn
    """
    if bbox_3d_data is None or 'projection' not in bbox_3d_data:
        return frame_img
    
    # Use red color for all bounding boxes
    box_color = (0, 0, 255)  # Red in BGR format
    
    # Draw all projected edges
    for edge in bbox_3d_data['projection']:
        x1, y1, x2, y2 = edge
        cv2.line(frame_img, (x1, y1), (x2, y2), box_color, thickness)
    
    # Add label if we have projection lines
    if bbox_3d_data['projection']:
        # Calculate center point for label placement
        all_points = [(edge[0], edge[1]) for edge in bbox_3d_data['projection']] + \
                     [(edge[2], edge[3]) for edge in bbox_3d_data['projection']]
        if all_points:
            center_x = sum(p[0] for p in all_points) // len(all_points)
            center_y = sum(p[1] for p in all_points) // len(all_points)
            
            # Draw actor ID label
            if 'actor_id' in bbox_3d_data:
                label = f"{bbox_3d_data['actor_id']}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(frame_img, 
                              (center_x - text_width//2 - 2, center_y - text_height//2 - 2),
                              (center_x + text_width//2 + 2, center_y + text_height//2 + 2), 
                              box_color, -1)
                
                # Draw text
                cv2.putText(frame_img, label, 
                            (center_x - text_width//2, center_y + text_height//2), 
                            font, font_scale, (255, 255, 255), font_thickness)
    
    return frame_img

def save_debug_frame_with_3d_bboxes(frame_img, frame_bboxes_masks, frame_num, debug_dir, prefix_filename='bboxs_3d'):
    """
    Save frame with drawn 3D bounding boxes to debug folder
    Args:
        frame_img: OpenCV image (BGR format)
        frame_bboxes_masks: list of bbox dictionaries with '2d' and '3d' keys
        frame_num: int, frame number
        debug_dir: string, path to debug directory
        prefix_filename: prefix for output filename
    """
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)
    
    # Copy frame to avoid modifying original
    debug_frame = frame_img.copy()
    
    # Draw 3D bounding boxes for all detections
    for bbox_mask_data in frame_bboxes_masks:
        if bbox_mask_data is not None and '3d' in bbox_mask_data:
            debug_frame = draw_bbox3d_on_frame(debug_frame, bbox_mask_data['3d'])
    
    # Save debug frame
    debug_filename = os.path.join(debug_dir, f"{prefix_filename}_{frame_num:012d}.jpg")
    cv2.imwrite(debug_filename, debug_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    
    return debug_filename

def save_debug_frame_with_masks(frame_img, frame_bboxes_masks, frame_num, debug_dir, prefix_filename='masks'):
    """
    Save frame with drawn bounding boxes AND masks to debug folder
    Args:
        frame_img: OpenCV image (BGR format)
        frame_bboxes: list of bbox dictionaries with '2d' and '3d' keys
        frame_num: int, frame number
        debug_dir: string, path to debug directory
        prefix_filename: prefix for output filename
    """
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)
    
    # Copy frame to avoid modifying original
    debug_frame = frame_img.copy()
    
    # Draw masks with different colors for each instance
    for i, bbox_mask_data in enumerate(frame_bboxes_masks):
        if bbox_mask_data is not None and '2d' in bbox_mask_data:
            bbox_mask_data_2d = bbox_mask_data['2d']
            
            # Get unique color for each instance based on actor_id
            actor_id = bbox_mask_data_2d.get('actor_id', i)
            # Use the instance colormap to get a unique color for each actor
            instance_color = INSTANCE_COLORMAP[actor_id % len(INSTANCE_COLORMAP)]
            bgr_color = (int(instance_color[2]), int(instance_color[1]), int(instance_color[0]))  # Convert RGB to BGR

            debug_frame = draw_polygon_mask(debug_frame, bbox_mask_data_2d['mask_polygons'], bgr_color, alpha=0.4, thickness=2)
            
            # Add text label on the mask (similar to bbox overlay)
            bbox = bbox_mask_data_2d['bbox_2d']
            semantic_label = bbox_mask_data_2d['semantic_label']
            semantic_name = SEMANTIC_MAP.get(semantic_label, ('Unknown', (255, 255, 255)))[0]
            
            # Calculate mask centroid for text positioning
            mask_polygons = bbox_mask_data_2d['mask_polygons']
            if mask_polygons:
                # Calculate centroid of all polygon points
                # Polygons are stored as flattened lists: [x1, y1, x2, y2, x3, y3, ...]
                all_x_coords = []
                all_y_coords = []
                
                for polygon in mask_polygons:
                    # Extract x and y coordinates from flattened format
                    for i in range(0, len(polygon), 2):
                        if i + 1 < len(polygon):
                            all_x_coords.append(polygon[i])
                            all_y_coords.append(polygon[i + 1])
                
                if all_x_coords and all_y_coords:
                    # Calculate centroid
                    centroid_x = sum(all_x_coords) / len(all_x_coords)
                    centroid_y = sum(all_y_coords) / len(all_y_coords)
                else:
                    # Fallback to bbox center if no polygon points
                    bbox = bbox_mask_data_2d['bbox_2d']
                    centroid_x = (bbox[0] + bbox[2]) / 2
                    centroid_y = (bbox[1] + bbox[3]) / 2
            else:
                # Fallback to bbox center if no mask polygons
                bbox = bbox_mask_data_2d['bbox_2d']
                centroid_x = (bbox[0] + bbox[2]) / 2
                centroid_y = (bbox[1] + bbox[3]) / 2
            
            # Create label text
            label_text = f"{actor_id}"

            # Get text dimensions for centering
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Center text position
            text_x = int(centroid_x - text_width / 2)
            text_y = int(centroid_y + text_height / 2)  # Add half height to center vertically
            
            # Add text background for better visibility
            cv2.rectangle(debug_frame, 
                         (text_x - 2, text_y - text_height - 2), 
                         (text_x + text_width + 2, text_y + 2), 
                         bgr_color, -1)
            
            # Add text
            cv2.putText(debug_frame, label_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
    # Save debug frame
    debug_filename = os.path.join(debug_dir, f"{prefix_filename}_{frame_num:012d}.jpg")
    cv2.imwrite(debug_filename, debug_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    
    return debug_filename

def draw_collisions_on_frame(frame_img, bbox_mask_data, color=(0, 0, 255),thickness=2):
    if bbox_mask_data is None or '2d' not in bbox_mask_data or bbox_mask_data['2d'] is None:
        return frame_img
    
    bbox_mask_data_2d = bbox_mask_data['2d']
    x_min, y_min, x_max, y_max = bbox_mask_data_2d['bbox_2d']
    semantic_label = bbox_mask_data_2d['semantic_label']
    actor_id = bbox_mask_data_2d['actor_id']
    
    # Use red color for all bounding boxes
    box_color = (0, 0, 255)  # Red in BGR format
    
    if semantic_label in SEMANTIC_MAP:
        label_name, semantic_color = SEMANTIC_MAP[semantic_label]
    else:
        label_name = f"class_{semantic_label}"
    
    # Draw rectangle
    cv2.rectangle(frame_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, thickness)
    
    # Draw category label
    label = f"{actor_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
    )
    
    center_x = int((x_min + x_max) / 2)  # Proper center x
    center_y = int((y_min + y_max) / 2)  # Proper center y
    text_x = center_x - text_width // 2      # Center horizontally
    text_y = center_y + text_height // 2     # Center vertically
    
    # Draw background rectangle for text
    cv2.rectangle(frame_img,
                  (text_x - LABEL_PADDING, text_y - text_height - LABEL_PADDING),
                  (text_x + text_width + LABEL_PADDING, text_y + LABEL_PADDING),
                  box_color, -1)
                  
    # Draw text
    cv2.putText(frame_img, label,
                (text_x, text_y),
                font, LABEL_FONT_SCALE, (255, 255, 255), LABEL_FONT_THICKNESS)
                
    return frame_img

def save_debug_frame_collisions(frame_img, frame_bboxes_masks, frame_num, debug_dir, all_actor_id_in_collisions, prefix_filename='collisions'):
    """
    Save frame with drawn collisions to debug folder
    """
    # Create debug directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)
    
    # Copy frame to avoid modifying original
    debug_frame = frame_img.copy()

    # Draw bounding boxes for vechicle have id in collisions
    for bbox_mask_data in frame_bboxes_masks:
        if bbox_mask_data is not None:
            if bbox_mask_data['2d']['actor_id'] in all_actor_id_in_collisions:
                debug_frame = draw_collisions_on_frame(debug_frame, bbox_mask_data)

    # Save debug frame
    debug_filename = os.path.join(debug_dir, f"{prefix_filename}_{frame_num:012d}.png")
    cv2.imwrite(debug_filename, debug_frame)
    
    return debug_filename

# Decode the instance segmentation map into semantic labels and actor IDs
def decode_instance_segmentation(img_rgba: np.ndarray):
    semantic_labels = img_rgba[..., 2]  # R channel
    actor_ids = img_rgba[..., 1].astype(np.uint16) + (img_rgba[..., 0].astype(np.uint16) << 8)
    return semantic_labels, actor_ids

# Generate a 2D bounding box and mask for an actor from the actor ID image
def bbox_2d_and_mask_for_actor(actor, actor_ids: np.ndarray, semantic_labels: np.ndarray, area_threshold: int):
    mask = (actor_ids == actor.id)
    if not np.any(mask):
        return None  # actor not present
    ys, xs = np.where(mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    area = (xmax - xmin) * (ymax - ymin)
    if area < area_threshold:
        return None
    return {'actor_id': actor.id,
            'semantic_label': actor.semantic_tags[0],
            'mask_polygons': mask_to_polygon_json(mask),
            'bbox_2d': (xmin, ymin, xmax, ymax)
            }

# Construct camera intrinsic matrix for 3D-to-2D projection
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0  # Principal point x
    K[1, 2] = h / 2.0  # Principal point y
    return K

# Project a 3D world coordinate onto 2D image plane using camera matrix
def get_image_point(loc, K, w2c):
    
    # Convert CARLA location to homogeneous coordinates for matrix operations
    point = np.array([loc.x, loc.y, loc.z, 1])
    # Transform from world space to camera space
    point_camera = np.dot(w2c, point)

    # Convert from Unreal Engine 4 coordinate system to standard camera coordinates
    # UE4: (x=forward, y=right, z=up) -> Camera: (x=right, y=down, z=forward)
    # Also drop the homogeneous w-component
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # Apply camera intrinsics to project 3D point to 2D image coordinates
    point_img = np.dot(K, point_camera)
    # Normalize by depth (z-coordinate) to get pixel coordinates
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

# Check if 2D point lies within image boundaries
def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False
def bbox_3d_for_actor(actor, camera_bp, camera):
    """
    Calculate 3D bounding box relative to camera instead of ego vehicle
    """
    # Build world-to-camera transformation matrix
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Extract camera intrinsic parameters from blueprint
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Construct projection matrices (regular and inverted for behind-camera handling)
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # Compute vehicle's bounding box center in world coordinates
    vehicle_bbox_loc = actor.get_transform().location + actor.bounding_box.location
    
    # Transform vehicle center from world space to camera space
    vehicle_location_world = np.array([vehicle_bbox_loc.x, vehicle_bbox_loc.y, vehicle_bbox_loc.z, 1])
    vehicle_location_camera = np.dot(world_2_camera, vehicle_location_world)
    
    # Apply coordinate system transformation: UE4 world -> standard camera frame
    # UE4: (x=forward, y=right, z=up) -> Camera: (x=right, y=down, z=forward)
    vehicle_loc_camera_space = {
        'x': vehicle_location_camera[1],    # right axis
        'y': -vehicle_location_camera[2],   # down axis  
        'z': vehicle_location_camera[0]     # forward axis (depth)
    }

    # Extract all 8 vertices of the 3D bounding box in world coordinates
    verts = [v for v in actor.bounding_box.get_world_vertices(actor.get_transform())]

    # Project bbox edges onto 2D image plane (creates wireframe visualization)
    projection = []
    for edge in EDGES:
        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
        p2 = get_image_point(verts[edge[1]], K, world_2_camera)

        p1_in_canvas = point_in_canvas(p1, image_h, image_w)
        p2_in_canvas = point_in_canvas(p2, image_h, image_w)

        # Skip edges where both endpoints are outside the image frame
        if not p1_in_canvas and not p2_in_canvas:
            continue

        ray0 = verts[edge[0]] - camera.get_transform().location
        ray1 = verts[edge[1]] - camera.get_transform().location
        cam_forward_vec = camera.get_transform().get_forward_vector()

        if not (cam_forward_vec.dot(ray0) > 0):
            p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
        if not (cam_forward_vec.dot(ray1) > 0):
            p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)
        
        projection.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))

    # Get actual vehicle transform in world coordinates
    vehicle_transform = actor.get_transform()
    vehicle_rotation = vehicle_transform.rotation
    
    # Vehicle location in world coordinates (CARLA: x=forward, y=right, z=up)
    vehicle_loc_world = {
        'x': float(vehicle_bbox_loc.x),
        'y': float(vehicle_bbox_loc.y),
        'z': float(vehicle_bbox_loc.z)
    }
    
    # Actual vehicle orientation in world coordinates (degrees)
    vehicle_orientation_world = {
        'roll': float(vehicle_rotation.roll),
        'pitch': float(vehicle_rotation.pitch),
        'yaw': float(vehicle_rotation.yaw)
    }

    return {
        'actor_id': actor.id,
        'semantic_label': actor.semantic_tags[0],
        'bbox_3d': {
            'center': vehicle_loc_camera_space,
            'dimensions': {
                'length': actor.bounding_box.extent.x * 2,
                'width': actor.bounding_box.extent.y * 2,
                'height': actor.bounding_box.extent.z * 2
            },
            'rotation': {
                'yaw': math.atan2(vehicle_location_camera[1], vehicle_location_camera[0]),
                'pitch': math.atan2(-vehicle_location_camera[2], 
                           math.sqrt(vehicle_location_camera[0]**2 + vehicle_location_camera[1]**2)),
                'roll': 0
            },
            'distance_to_camera': math.sqrt(
                vehicle_location_camera[0]**2 + 
                vehicle_location_camera[1]**2 + 
                vehicle_location_camera[2]**2
            )
        },
        'bbox_3d_world': {
            'center': vehicle_loc_world,
            'dimensions': {
                'length': actor.bounding_box.extent.x * 2,
                'width': actor.bounding_box.extent.y * 2,
                'height': actor.bounding_box.extent.z * 2
            },
            'rotation': vehicle_orientation_world
        },
        'projection': projection
    }
def extract_value(key, arg_value, config_values):
    """ resolve the config value with priority as follow: cli -> config file"""
    
    # handle bool flags excplicitly
    if isinstance(arg_value, bool):
        return arg_value
    if arg_value is not None:
        return arg_value
    if config_values and key in config_values and config_values[key] is not None:
        return config_values[key]
    raise ValueError("wf config element {} cannot be empty!".format(key))

def main():
    """Main function to parse arguments and start the game loop."""
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON configuration file that sets all parameters')
    argparser.add_argument(
        '--wf-config',
        type=str,
        default=None,
        dest='wf_config',
        help='Path to workflow configuration JSON file')
    argparser.add_argument(
        '--host',
        metavar='H',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080',
        help='window resolution (default: 1920x1080)')
    argparser.add_argument(
        '--camera-config',
        metavar='CONFIG',
        default=None,
        help='Path to camera configuration YAML file')
    argparser.add_argument(
        '--fov',
        default=110.0,
        type=float,
        help='Camera field of view in degrees (default: 110.0, overridden by config file if provided)')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='X position of the camera (overridden by config file if provided)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='Y position of the camera (overridden by config file if provided)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='Z position of the camera (overridden by config file if provided)')
    argparser.add_argument(
        '--pitch',
        default=0.0,
        type=float,
        help='Pitch rotation of the camera (overridden by config file if provided)')
    argparser.add_argument(
        '--yaw',
        default=0.0,
        type=float,
        help='Yaw rotation of the camera (overridden by config file if provided)')
    argparser.add_argument(
        '--roll',
        default=0.0,
        type=float,
        help='Roll rotation of the camera (overridden by config file if provided)')
    argparser.add_argument(
        '-s', '--start',
        metavar='S',
        default=0.0,
        type=float,
        help='starting time (default: 0.0)')
    argparser.add_argument(
        '-d', '--duration',
        metavar='D',
        default=0.0,
        type=float,
        help='duration (default: 0.0)')
    argparser.add_argument(
        '-f', '--recorder-filename',
        metavar='F',
        default=None,
        help='recorder filename (required)')
    argparser.add_argument(
        '-tf', '--time-factor',
        metavar='X',
        type=float,
        help='time factor (default 1.0)')
    argparser.add_argument(
        '-c', '--camera',
        metavar='C',
        default=0,
        type=int,
        help='camera follows an actor (ex: 82)')
    argparser.add_argument(
        '-i', '--ignore-hero',
        action='store_true',
        help='ignore hero vehicles')
    argparser.set_defaults(ignore_hero=None)
    argparser.add_argument(
        '--move-spectator',
        action='store_true',
        help='move spectator camera')
    argparser.set_defaults(move_spectator=None)
    argparser.add_argument(
        '-ld', '--limit-distance',
        metavar='LD',
        type=float,
        help=f'limit distance (default: {DEFAULT_LIMIT_DISTANCE})')
    argparser.add_argument(
        '-at', '--area-threshold',
        metavar='AT',
        type=int,
        help=f'area threshold (default: {DEFAULT_AREA_THRESHOLD})')
    argparser.add_argument(
        '--class-filter-config',
        type=str,
        help='Path to class filter configuration YAML file (default: config/filter_semantic_classes.yaml)')
    argparser.add_argument(
        '-o', '--output-dir',
        metavar='OUTPUT',
        help='output directory (default: /tmp/odvg_output)')
    argparser.add_argument(
        '--xodr-path',
        metavar='XODR',
        default=None,
        help='Path to OpenDRIVE (.xodr) file to load custom map')
    argparser.add_argument(
        '--generate-videos',
        action='store_true',
        help='Automatically generate videos from frames after processing')
    argparser.set_defaults(generate_videos=None)
    argparser.add_argument(
        '--timeout',
        type=float,
        help='Client timeout in seconds (default: 60.0)')
    argparser.add_argument(
        '--detect-collisions',
        action='store_true',
        help='Enable collision detection and include in output')
    argparser.set_defaults(detect_collisions=None)
    argparser.add_argument(
        '--collision-actor-id',
        type=int,
        action='append',
        dest='collision_actor_ids',
        default=None,
        help='Filter collisions for specific actor ID (can be specified multiple times)')
    argparser.add_argument(
        '--target-fps',
        type=int,
        default=DEFAULT_TARGET_FPS,
        help=f'Target FPS for frame extraction (default: {DEFAULT_TARGET_FPS}). If recording FPS > target, frames will be skipped. If recording FPS < target, original FPS is used.')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
    # Set up logging first
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=log_level
    )
    
    # Load configuration file if provided
    if args.config:
        if os.path.exists(args.config):
            load_config(args.config, args)
            # Re-parse resolution in case it was changed by config
            if hasattr(args, 'width') and hasattr(args, 'height'):
                args.res = f"{args.width}x{args.height}"
        else:
            logging.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
    
    # Load workflow configuration if provided via --wf-config
    if args.wf_config:
        # Use explicitly provided wf-config path
        if os.path.exists(args.wf_config):
            load_workflow_config(args.wf_config, args)
            logging.info(f"Loaded workflow config from: {args.wf_config}")
        else:
            logging.error(f"Workflow configuration file not found: {args.wf_config}")
            sys.exit(1)
    logging.info(f"Final config args: {args}")
    # Load class filter configuration
    if args.class_filter_config and os.path.exists(args.class_filter_config):
        load_class_filter_config(args.class_filter_config)
        logging.info(f"Loaded class filter configuration from {args.class_filter_config}")
    else:
        logging.warning(f"Class filter configuration not found at {args.class_filter_config}, using empty filters")

    # Load camera configuration from YAML if provided
    if args.camera_config:
        with open(args.camera_config, 'r') as f:
            camera_configs = yaml.safe_load(f)
            
        # Extract sensors list if it exists
        camera_configs = camera_configs.get('sensors', camera_configs) if isinstance(camera_configs, dict) else camera_configs
            
        # Find RGB camera config
        rgb_config = None
        instance_config = None
        for config in camera_configs:
            if config['sensor'] == 'rgb':
                rgb_config = config
            elif config['sensor'] == 'instance_segmentation':
                instance_config = config
        
        if rgb_config:
            # Override command line arguments with config values
            if 'attributes' in rgb_config:
                attrs = rgb_config['attributes']
                if 'image_size_x' in attrs:
                    args.width = attrs['image_size_x']
                if 'image_size_y' in attrs:
                    args.height = attrs['image_size_y']
                if 'fov' in attrs:
                    args.fov = attrs['fov']
            
            if 'transform' in rgb_config:
                transform = rgb_config['transform']
                if 'location' in transform:
                    loc = transform['location']
                    # Note: YAML values are stored as actual coordinates, not divided by 100
                    # Keep them as-is since they'll be divided by 100 when creating the transform
                    args.x = loc.get('x', args.x)
                    args.y = loc.get('y', args.y)
                    args.z = loc.get('z', args.z)
                if 'rotation' in transform:
                    rot = transform['rotation']
                    args.roll = rot.get('roll', args.roll)
                    args.pitch = rot.get('pitch', args.pitch)
                    args.yaw = rot.get('yaw', args.yaw)
            
            logging.info(f"Loaded camera configuration from {args.camera_config}")
            logging.info(f"Camera position: x={args.x}, y={args.y}, z={args.z}")
            logging.info(f"Camera rotation: roll={args.roll}, pitch={args.pitch}, yaw={args.yaw}")
            logging.info(f"Camera FOV: {args.fov}, Resolution: {args.width}x{args.height}")

    logging.info('listening to server %s:%s', args.host, args.port)

    # Initialize sensors dictionary early to avoid UnboundLocalError
    sensors = {}
    sensor_queues = {}
    
    # Initialize output_dir early to avoid UnboundLocalError in finally block
    output_dir = args.output_dir
    
    try:
        # Connect to CARLA server
        try:
            logging.info(f"Connecting to CARLA server at {args.host}:{args.port}...")
            client = carla.Client(args.host, args.port)
            client.set_timeout(args.timeout)
            
            # Test connection by getting server version
            server_version = client.get_server_version()
            client_version = client.get_client_version()
            logging.info(f"Connected successfully!")
            logging.info(f"  Server version: {server_version}")
            logging.info(f"  Client version: {client_version}")
            
        except RuntimeError as e:
            logging.error(f"Failed to connect to CARLA server at {args.host}:{args.port}")
            logging.error(f"Error: {e}")
            logging.error("Please ensure:")
            logging.error("  1. CARLA server is running")
            logging.error("  2. Host and port are correct")
            logging.error("  3. No firewall is blocking the connection")
            sys.exit(1)
        
        # Keep the .xodr world; ignore any map switches recorded in the log
        if hasattr(client, "set_replayer_ignore_map_changes"):
            client.set_replayer_ignore_map_changes(True)
        
        # Step 1: Load OpenDRIVE map FIRST (if provided)
        # This replaces any placeholder map that might be in the log file
        # The log was recorded with road-only and the xodr contains the full map
        if args.xodr_path is not None:
            if os.path.exists(args.xodr_path):
                with open(args.xodr_path, encoding='utf-8') as od_file:
                    try:
                        data = od_file.read()
                    except OSError:
                        logging.error('File could not be read.')
                        sys.exit()
                logging.info('Loading OpenDRIVE map: %s', os.path.basename(args.xodr_path))
                world = client.generate_opendrive_world(
                    data, carla.OpendriveGenerationParameters(
                        vertex_distance=OPENDRIVE_VERTEX_DISTANCE,
                        max_road_length=OPENDRIVE_MAX_ROAD_LENGTH,
                        wall_height=OPENDRIVE_WALL_HEIGHT,
                        additional_width=OPENDRIVE_EXTRA_WIDTH,
                        smooth_junctions=True,
                        enable_mesh_visibility=True))
            else:
                logging.error(f'OpenDRIVE file not found: {args.xodr_path}')
                sys.exit()
        else:
            # Get world if no xodr was loaded
            world = client.get_world()
        
        # Extract recording metadata (FPS, duration) to configure playback parameters
        recorder_filepath = os.path.abspath(args.recorder_filename)
        info = client.show_recorder_file_info(recorder_filepath, False)
        map_name = None
        match = re.search(r"Map: (\w+)", info)
        if match:
            map_name = match.group(1)
            logging.info(f"Map extracted from log: {map_name}")
            if map_name:
                logging.info(f"Loading map: {map_name}")
                world = client.load_world(map_name)
                current_map = world.get_map()
                logging.info(f"Loaded map: {current_map.name}")
        else:
            logging.info("Map name not found in the log file info! using default map")

        log_frames, log_duration = parse_frames_duration(info)

        log_delta = log_duration / log_frames
        fps = round(1.0 / log_delta)
        logging.info(f"Recorder: {log_frames} frames, {log_duration:.2f}s, fps={fps}")
        
        # Calculate frame skipping based on target FPS
        target_fps = args.target_fps
        frame_skip = 1
        effective_fps = fps
        
        if fps > target_fps:
            # Need to skip frames to reduce FPS
            frame_skip = round(fps / target_fps)
            effective_fps = fps / frame_skip
            logging.info(f"Recording FPS ({fps}) > target FPS ({target_fps})")
            logging.info(f"Frame skipping: will extract 1 out of every {frame_skip} frames")
            logging.info(f"This reduces {fps} FPS → {effective_fps:.1f} FPS in the extracted frames")
            logging.info(f"Output video will play at {target_fps} FPS (real-time speed)")
        else:
            # Use original FPS if it's lower than target
            logging.info(f"Recording FPS ({fps}) <= target FPS ({target_fps}), using original FPS")
            logging.info(f"No frame skipping needed, output video will play at {fps} FPS")
        
        # CONFIGURE SIMULATION TIMESTEP (before spawning sensors)
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = log_delta #0.033  # Approximately 1/30
        world.apply_settings(settings)

        # Step 2: SPAWN CAMERAS/SENSORS (before replay)
        # Get the blueprint for the cameras
        blueprint_library = world.get_blueprint_library()
        
        # Define the camera's location and rotation
        camera_transform = carla.Transform(
            carla.Location(
                x=loc.get('x', 0) / UNREAL_TO_METER_SCALE,
                y=loc.get('y', 0) / UNREAL_TO_METER_SCALE,
                z=loc.get('z', 0) / UNREAL_TO_METER_SCALE
            ),
            carla.Rotation(pitch=args.pitch, yaw=args.yaw, roll=args.roll)
        )
        
        # If camera config is provided, spawn all sensors from config
        if args.camera_config:
            with open(args.camera_config, 'r') as f:
                camera_configs = yaml.safe_load(f)
            
            # Extract sensors list if it exists
            camera_configs = camera_configs.get('sensors', camera_configs) if isinstance(camera_configs, dict) else camera_configs
            
            for config in camera_configs:
                sensor_type = config['sensor']
                bp = blueprint_library.find(f'sensor.camera.{sensor_type}')
                
                # Apply attributes
                for k, v in config.get('attributes', {}).items():
                    bp.set_attribute(k, str(v))
                
                # Apply transform from config
                tf = config.get('transform', {})
                loc = tf.get('location', {})
                rot = tf.get('rotation', {})
                sensor_transform = carla.Transform(
                    carla.Location(
                        x=loc.get('x', 0) / UNREAL_TO_METER_SCALE,
                        y=loc.get('y', 0) / UNREAL_TO_METER_SCALE,
                        z=loc.get('z', 0) / UNREAL_TO_METER_SCALE
                    ),
                    carla.Rotation(roll=rot.get('roll', 0), pitch=rot.get('pitch', 0), yaw=rot.get('yaw', 0))
                )
                
                sensor = world.spawn_actor(bp, sensor_transform)
                sensors[sensor_type] = sensor
                sensor_queues[sensor_type] = queue.Queue()
                logging.debug(f'Created {sensor_type} camera with id {sensor.id}')
        else:
            # Default behavior: spawn RGB and instance segmentation only
            # spawn RGB camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', f'{args.width}')
            camera_bp.set_attribute('image_size_y', f'{args.height}')
            camera_bp.set_attribute('fov', str(args.fov))
            camera = world.spawn_actor(camera_bp, camera_transform)
            sensors['rgb'] = camera
            sensor_queues['rgb'] = queue.Queue()
            logging.debug(f'Created camera "{camera.type_id}" with id {camera.id}')
            
            # spawn instance segmentation camera
            inst_camera_bp = blueprint_library.find('sensor.camera.instance_segmentation')
            inst_camera_bp.set_attribute('image_size_x', f'{args.width}')
            inst_camera_bp.set_attribute('image_size_y', f'{args.height}')
            inst_camera_bp.set_attribute('fov', str(args.fov))
            inst_camera = world.spawn_actor(inst_camera_bp, camera_transform)
            sensors['instance_segmentation'] = inst_camera
            sensor_queues['instance_segmentation'] = queue.Queue()
            logging.debug(f'Created instance segmentation camera "{inst_camera.type_id}" with id {inst_camera.id}')

        #--- IMAGE STREAMING --
        # Set up listeners for all sensors
        for sensor_type, sensor in sensors.items():
            sensor.listen(sensor_queues[sensor_type].put)
        
        # Let sensors activate, then drain any warm-up frames
        world.tick()
        for q in sensor_queues.values():
            while not q.empty():
                q.get()

        # Step 3: START LOG REPLAY (after sensors are spawned)
        #--- REPLAYER ---
        # set the time factor for the replayer
        client.set_replayer_time_factor(args.time_factor)
        # set to ignore the hero vehicles or not
        client.set_replayer_ignore_hero(args.ignore_hero)
        # set to ignore the spectator camera or not
        client.set_replayer_ignore_spectator(not args.move_spectator)
        # replay the session
        logging.info("Starting replay...")
        logging.info(f"Replaying file: {recorder_filepath}")
        try:
            replay_status = client.replay_file(recorder_filepath, args.start, args.duration, args.camera, replay_sensors=False)
            logging.debug(f"Replay status: {replay_status}")
        except RuntimeError as e:
            logging.error(f"Failed to start replay: {e}")
            logging.error("This is likely due to:")
            logging.error("  1. Version mismatch between CARLA client and server")
            logging.error("  2. Incompatible recording file")
            logging.error("  3. Recording file is corrupted or invalid")
            sys.exit(1)

        # Run collision detection if enabled
        collision_events = {}
        if args.detect_collisions:
            logging.info("Running collision detection...")
            collision_events = run_collision_detection(
                recorder_filepath, 
                args.host, 
                args.port, 
                args.collision_actor_ids,
                args.debug
            )
            if collision_events:
                logging.info(f"Found {len(collision_events)} collision events")
                # Save collision events to output directory for reference
                events_file = os.path.join(args.output_dir, 'events_collision.json')
                os.makedirs(args.output_dir, exist_ok=True)
                with open(events_file, 'w') as f:
                    # Convert back to list format for JSON output
                    # Adjust times to be relative to start time
                    events_list = []
                    for event in collision_events.values():
                        event_copy = event.copy()
                        event_copy['start_time'] = float(event['start_time']) - args.start
                        event_copy['end_time'] = float(event['end_time']) - args.start
                        events_list.append(event_copy)
                    json.dump(events_list, f, indent=2)
                logging.info(f"Saved collision events to: {events_file}")
            else:
                logging.info("No collisions detected")

        # MAIN SIMULATION LOOP
        # output_dir already initialized before try block
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for each sensor type
        sensor_dirs = {}
        for sensor_type in sensors.keys():
            sensor_dir = os.path.join(output_dir, sensor_type)
            os.makedirs(sensor_dir, exist_ok=True)
            sensor_dirs[sensor_type] = sensor_dir
        
        # Also create raw data directories for segmentation sensors
        if 'semantic_segmentation' in sensors:
            os.makedirs(os.path.join(output_dir, 'semantic_segmentation_raw'), exist_ok=True)
        if 'instance_segmentation' in sensors:
            os.makedirs(os.path.join(output_dir, 'instance_segmentation_raw'), exist_ok=True)
        
        # Create directories for edge and mask outputs
        if 'rgb' in sensors and 'semantic_segmentation' in sensors:
            edge_dir = os.path.join(output_dir, 'edges')
            mask_dir = os.path.join(output_dir, 'masks')
            os.makedirs(edge_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            sensor_dirs['edges'] = edge_dir
            sensor_dirs['masks'] = mask_dir
        
        # Save semantic labels JSON if we have semantic segmentation
        if 'semantic_segmentation' in sensors:
            semantic_labels = {}
            for label_id, (name, color) in SEMANTIC_MAP.items():
                semantic_labels[str(label_id)] = {
                    "name": name,
                    "color": list(color)  # Convert tuple to list for JSON
                }
            semantic_json_path = os.path.join(sensor_dirs['semantic_segmentation'], 'semantic_labels.json')
            with open(semantic_json_path, 'w') as f:
                json.dump(semantic_labels, f, indent=2)
            logging.debug(f"Saved semantic labels to {semantic_json_path}")
        
        # Create debug subdirectories (always generated)
        debug_bbox_dir = os.path.join(output_dir, 'rgb_with_bboxes')
        debug_mask_dir = os.path.join(output_dir, 'rgb_with_masks')
        debug_bbox3d_dir = os.path.join(output_dir, 'rgb_with_3d_bboxes')
        debug_collisions_dir = os.path.join(output_dir, 'rgb_with_collisions')
        os.makedirs(debug_bbox_dir, exist_ok=True)
        os.makedirs(debug_mask_dir, exist_ok=True)
        os.makedirs(debug_bbox3d_dir, exist_ok=True)
        os.makedirs(debug_collisions_dir, exist_ok=True)
        
        # Create odvg directory for ODVG JSON files
        odvg_dir = os.path.join(output_dir, 'odvg')
        os.makedirs(odvg_dir, exist_ok=True)
    
        frames = []  # To store frames for video
        timestamp = args.start  # Start from the specified start time
        frame_count = 0
        
        # Calculate starting frame number based on start time and fps
        starting_frame_num = int(args.start / log_delta)  # log_delta is the time per frame
        sequential_frame_num = starting_frame_num  # Start numbering from the appropriate frame
        recording_frame_num = starting_frame_num  # Track actual frame number in recording
        output_frame_num = 0  # Track output frame number (for sequential naming)
        
        # Calculate actual end time based on start and duration arguments
        end_time = args.start + args.duration if args.duration > 0 else log_duration
        
        # Log the frame range that will be processed
        if args.duration > 0:
            ending_frame_num = int(end_time / log_delta)
            logging.info(f"Processing frames {starting_frame_num} to {ending_frame_num} (time: {args.start:.2f}s to {end_time:.2f}s)")
        else:
            logging.info(f"Processing from frame {starting_frame_num} (time: {args.start:.2f}s) to end of recording")
        
        # Log frame skipping info
        if frame_skip > 1:
            total_frames = int((end_time - args.start) / log_delta)
            output_frames = total_frames // frame_skip
            logging.info(f"Frame skipping enabled: processing every {frame_skip} frames")
            logging.info(f"Total input frames: {total_frames}, Output frames: ~{output_frames}")
        
        # Extract actor IDs involved in collisions
        all_actor_id_in_collisions = set()
        if collision_events:
            for event in collision_events.values():
                all_actor_id_in_collisions.update(event.get('objects', []))
        all_actor_id_in_collisions = [int(aid) for aid in all_actor_id_in_collisions]

        instance_data = {
            'instances': {}
        }

        object_data = {
            'frames':{}
        }
        
        while timestamp < end_time:
            frame_idx = world.tick()
            
            # Skip frames based on frame_skip value
            if recording_frame_num % frame_skip != 0:
                # Skip this frame - but we need to drain the sensor queues
                # to keep them in sync with the simulation
                try:
                    for sensor_type, sensor_queue in sensor_queues.items():
                        _ = sensor_queue.get(timeout=1.0)  # Discard the data
                except queue.Empty:
                    logging.warning(f"No data received from sensors at skipped frame {recording_frame_num}")
                
                recording_frame_num += 1
                timestamp += log_delta
                continue
            
            # Use output_frame_num for file naming to maintain sequential numbering
            json_frame_data = {
                'frame_id': f"frame_{output_frame_num:012d}",
                "depth_file_name": f"depth/depth_{output_frame_num:012d}.png",
                "instance_segmentation_file_name": f"instance_segmentation/seg_{output_frame_num:012d}.png",
                "semantic_segmentation_file_name": f"semantic_segmentation/seg_{output_frame_num:012d}.png",
                "edge_file_name": f"edge/edge_{output_frame_num:012d}.png",
                'timestamp': timestamp,
                'width': args.width,
                'height': args.height,
                'detections': {
                    'instances': [] 
                }
            }
            object_data['frames'][f"frame_{output_frame_num:012d}"] = {
                                    'format':'jpg',
                                    'frame_id': f"frame_{output_frame_num:012d}",
                                    'instances':[]
                                }
            
            # Add collision events if any occur at this timestamp
            if args.detect_collisions and collision_events:
                # Check if there's a collision at this frame time (with some tolerance)
                for collision_time, collision_event in collision_events.items():
                    # Allow for small timing differences (within one frame)
                    if abs(timestamp - collision_time) < log_delta:
                        if 'events' not in json_frame_data:
                            json_frame_data['events'] = []
                        
                        # Add collision event to frame data
                        json_frame_data['events'].append({
                            'event_id': collision_event['event_id'],
                            'category': collision_event['category'],
                            'sub_category': collision_event['event_sub_category'],
                            'collision_actors': collision_event['objects'],
                            'event_caption': collision_event['event_caption']
                        })
                        logging.info(f"Collision detected at frame {output_frame_num}: actors {collision_event['objects']}")
            
            # Collect data from all sensors
            sensor_data = {}
            try:
                for sensor_type, sensor_queue in sensor_queues.items():
                    data = sensor_queue.get(timeout=1.0)
                    # Apply CityScapesPalette conversion for semantic segmentation
                    if sensor_type == 'semantic_segmentation':
                        data.convert(carla.ColorConverter.CityScapesPalette)
                    sensor_data[sensor_type] = data
            except queue.Empty:
                logging.warning(f"No data received from sensors at frame {frame_idx}")
                continue
            
            # Start frame logging (verbose only)
            logging.debug(f"\n=== Frame {output_frame_num:012d} ===")
            
            # Process and save images from all sensors
            processed_images = {}
            
            for sensor_type, sensor_image in sensor_data.items():
                img_array = np.frombuffer(sensor_image.raw_data, dtype=np.uint8)
                img_array = img_array.reshape((sensor_image.height, sensor_image.width, 4))
                
                # Save different sensor types appropriately
                if sensor_type == 'rgb':
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                    processed_images[sensor_type] = img_bgr
                    # Save RGB as JPEG
                    image_filename = f"frame_{output_frame_num:0{FRAME_NUMBER_PADDING}d}.jpg"
                    image_path = os.path.join(sensor_dirs[sensor_type], image_filename)
                    cv2.imwrite(image_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    logging.debug(f"  Saved RGB to: {image_path}")
                    
                elif sensor_type == 'depth':
                    # Convert depth to millimeters and save as 16-bit PNG
                    depth_array = img_array[:, :, :3].astype(np.float32)
                    # Convert from RGB encoding to depth value (CARLA-specific encoding)
                    normalized_depth = np.dot(depth_array, [DEPTH_R_WEIGHT, DEPTH_G_WEIGHT, DEPTH_B_WEIGHT])
                    normalized_depth /= DEPTH_NORMALIZATION
                    depth_meters = normalized_depth * 1000.0  # Convert to meters
                    depth_mm = (depth_meters * 1000).astype(np.uint16)  # Convert to millimeters
                    processed_images[sensor_type] = depth_mm
                    # Save as 16-bit PNG
                    image_filename = f"depth_{output_frame_num:012d}.png"
                    image_path = os.path.join(sensor_dirs[sensor_type], image_filename)
                    cv2.imwrite(image_path, depth_mm)
                    logging.debug(f"  Saved depth to: {image_path}")
                    
                elif sensor_type == 'semantic_segmentation':
                    # Semantic segmentation - save both colored and raw versions
                    # CARLA semantic segmentation camera outputs colored images with semantic information
                    processed_images[sensor_type] = img_array
                    
                    # Convert BGRA to BGR for colored visualization
                    semantic_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                    
                    # Save colored visualization (this is what CARLA provides with proper colors)
                    image_filename = f"seg_{output_frame_num:012d}.png"
                    image_path = os.path.join(sensor_dirs[sensor_type], image_filename)
                    cv2.imwrite(image_path, semantic_bgr)
                    logging.debug(f"  Saved semantic seg to: {image_path}")
                    
                    # Extract and save raw semantic labels
                    # After CityScapesPalette conversion, we need to map colors back to class IDs
                    semantic_labels = np.zeros(semantic_bgr.shape[:2], dtype=np.uint8)
                    
                    # Map each color to its semantic class ID
                    for class_id, (class_name, color_rgb) in SEMANTIC_MAP.items():
                        # Convert RGB to BGR for comparison
                        color_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]])
                        # Find pixels matching this color
                        mask = np.all(semantic_bgr == color_bgr, axis=2)
                        semantic_labels[mask] = class_id
                    
                    raw_dir = os.path.join(output_dir, 'semantic_segmentation_raw')
                    os.makedirs(raw_dir, exist_ok=True)
                    raw_path = os.path.join(raw_dir, image_filename)
                    cv2.imwrite(raw_path, semantic_labels)  # Save single channel with class IDs
                    logging.debug(f"  Saved semantic raw to: {raw_path}")
                    
                elif sensor_type == 'instance_segmentation':
                    # Instance segmentation - save both colored and raw versions
                    processed_images[sensor_type] = img_array
                    
                    # Extract instance IDs from the raw data
                    # In CARLA, instance IDs are encoded in the G and B channels
                    instance_ids = (img_array[:, :, 2].astype(np.uint16) << 8) | img_array[:, :, 1].astype(np.uint16)
                    
                    # Apply colormap for visualization (like carla_cosmos does)
                    colored_instances = INSTANCE_COLORMAP[instance_ids]
                    
                    # Save colored visualization
                    image_filename = f"seg_{output_frame_num:012d}.png"
                    image_path = os.path.join(sensor_dirs[sensor_type], image_filename)
                    cv2.imwrite(image_path, colored_instances)
                    logging.debug(f"  Saved instance seg to: {image_path}")
                    
                    # Save raw instance segmentation data (16-bit)
                    raw_dir = os.path.join(output_dir, 'instance_segmentation_raw')
                    os.makedirs(raw_dir, exist_ok=True)
                    raw_path = os.path.join(raw_dir, image_filename)
                    cv2.imwrite(raw_path, instance_ids)  # Save as 16-bit PNG
                    logging.debug(f"  Saved instance raw to: {raw_path}")
                    
                elif sensor_type == 'normals':
                    # Surface normals
                    normals_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                    processed_images[sensor_type] = normals_bgr
                    # Save as PNG
                    image_filename = f"frame_{output_frame_num:012d}.png"
                    image_path = os.path.join(sensor_dirs[sensor_type], image_filename)
                    cv2.imwrite(image_path, normals_bgr)
                    logging.debug(f"  Saved normals to: {image_path}")
                else:
                    # Generic handling for other sensor types
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                    processed_images[sensor_type] = img_bgr
                    image_filename = f"frame_{output_frame_num:012d}.png"
                    image_path = os.path.join(sensor_dirs[sensor_type], image_filename)
                    cv2.imwrite(image_path, img_bgr)

            # Update json with RGB image filename
            if 'rgb' in processed_images:
                json_frame_data['image_filename'] = f"frame_{output_frame_num:012d}.jpg"

            # Generate edge and mask from semantic segmentation
            if 'rgb' in processed_images and 'semantic_segmentation' in processed_images:
                rgb_img = processed_images['rgb']
                semantic_img = processed_images['semantic_segmentation']
                
                # Convert semantic segmentation BGRA to BGR for processing
                semantic_bgr = cv2.cvtColor(semantic_img, cv2.COLOR_BGRA2BGR)
                
                # Generate masked RGB and edges
                masked_rgb, edges = masked_edges_from_semseg(
                    rgb_img, 
                    semantic_bgr, 
                    CLASSES_TO_KEEP_CANNY
                )
                
                # Save masked RGB image
                mask_filename = f"frame_{output_frame_num:012d}.png"
                mask_path = os.path.join(sensor_dirs['masks'], mask_filename)
                cv2.imwrite(mask_path, masked_rgb)
                logging.debug(f"  Saved masks to: {mask_path}")
                
                # Save edge image (convert grayscale to BGR for consistency)
                edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                edge_filename = f"edge_{output_frame_num:012d}.png"
                edge_path = os.path.join(sensor_dirs['edges'], edge_filename)
                cv2.imwrite(edge_path, edge_bgr)
                logging.debug(f"  Saved edges to: {edge_path}")
                
                # Store processed images for potential later use
                processed_images['masks'] = masked_rgb
                processed_images['edges'] = edge_bgr

            # Empty list to collect bounding boxes for this frame
            frame_bboxes_masks = []
            
            # Process instance segmentation for bounding boxes if available
            if 'instance_segmentation' in processed_images and 'rgb' in sensors:
                inst_seg = processed_images['instance_segmentation']
                rgb_camera = sensors['rgb']
                
                # Get camera blueprint for FOV info
                camera_bp = blueprint_library.find('sensor.camera.rgb')
                camera_bp.set_attribute('image_size_x', str(sensor_data['rgb'].width))
                camera_bp.set_attribute('image_size_y', str(sensor_data['rgb'].height))
                camera_bp.set_attribute('fov', str(args.fov))
                
                # get bounding box and 2d projected bbox all vechicles
                vehicles = world.get_actors().filter('*vehicle*')
                for vehicle in vehicles:
                    dist = vehicle.get_transform().location.distance(rgb_camera.get_transform().location)
                    # Filter for the vehicles within limit_distance
                    if dist < args.limit_distance:
                        # Limit to vehicles in front of the camera
                        forward_vec = rgb_camera.get_transform().get_forward_vector()
                        inter_vehicle_vec = vehicle.get_transform().location - rgb_camera.get_transform().location
                        if forward_vec.dot(inter_vehicle_vec) > 0:
                            # Generate 2D and 3D bounding boxes for each actor
                            semantic_labels, actor_ids = decode_instance_segmentation(inst_seg)
                            bbox_mask_data = bbox_2d_and_mask_for_actor(vehicle, actor_ids, semantic_labels, args.area_threshold)
                            bbox_3d_data = bbox_3d_for_actor(vehicle, camera_bp, rgb_camera)
                            
                            if bbox_mask_data is not None:
                                # Always collect bbox/mask data for debug images
                                frame_bboxes_masks.append({'2d': bbox_mask_data, '3d': bbox_3d_data})
                                mask_polygons = bbox_mask_data['mask_polygons']
                                json_frame_data['detections']['instances'].append({
                                    'object-id': vehicle.id,
                                    'category': SEMANTIC_MAP[vehicle.semantic_tags[0]][0],
                                    'blueprint_id': vehicle.type_id,
                                    'bbox_2d': (int(bbox_mask_data['bbox_2d'][0]),int(bbox_mask_data['bbox_2d'][1]),int(bbox_mask_data['bbox_2d'][2]),int(bbox_mask_data['bbox_2d'][3])),
                                    'mask': mask_polygons,
                                    'road_lane_id': None,
                                    'lane_position': None, 
                                    'vehicle_color': [int(x.strip()) for x in vehicle.attributes['color'].split(',')] if 'color' in vehicle.attributes else None,
                                    'bbox_3d': bbox_3d_data['bbox_3d'],
                                    'bbox_3d_world': bbox_3d_data['bbox_3d_world']
                                })
                                instance_data['instances'][f"object_{vehicle.id}"] = {
                                    "object_type": SEMANTIC_MAP[vehicle.semantic_tags[0]][0],
                                    "instance_id": vehicle.id,
                                    "color": [int(x.strip()) for x in vehicle.attributes['color'].split(',')] if 'color' in vehicle.attributes else None,
                                    "caption": None
                                }
                                object_data['frames'][f"frame_{output_frame_num:012d}"]['instances'].append({
                                    'id': vehicle.id,
                                    'category': SEMANTIC_MAP[vehicle.semantic_tags[0]][0],
                                    'blueprint_id': vehicle.type_id,
                                    'bounding_box_2d_tight': (int(bbox_mask_data['bbox_2d'][0]),int(bbox_mask_data['bbox_2d'][1]),int(bbox_mask_data['bbox_2d'][2]),int(bbox_mask_data['bbox_2d'][3])),
                                    'mask': mask_polygons,
                                    'road_lane_id': None,
                                    'lane_position': None, 
                                    'vehicle_color': [int(x.strip()) for x in vehicle.attributes['color'].split(',')] if 'color' in vehicle.attributes else None,
                                    'bbox_3d': bbox_3d_data['bbox_3d']
                                })
                                

            if 'rgb' in processed_images:
                # Save debug frame with bounding boxes (always generated)
                rgb_img = processed_images['rgb']
                
                debug_mask_filename = save_debug_frame_with_masks(rgb_img, frame_bboxes_masks, output_frame_num, debug_mask_dir, prefix_filename='frame')
                logging.debug(f"  Saved masks to: {debug_mask_filename}")
                
                debug_bbox_filename = save_debug_frame_with_bboxes(rgb_img, frame_bboxes_masks, output_frame_num, debug_bbox_dir, prefix_filename='frame')
                logging.debug(f"  Saved 2D bboxes to: {debug_bbox_filename}")
                
                debug_bbox3d_filename = save_debug_frame_with_3d_bboxes(rgb_img, frame_bboxes_masks, output_frame_num, debug_bbox3d_dir, prefix_filename='frame')
                logging.debug(f"  Saved 3D bboxes to: {debug_bbox3d_filename}")

                debug_collisions_filename = save_debug_frame_collisions(rgb_img, frame_bboxes_masks, output_frame_num, debug_collisions_dir,all_actor_id_in_collisions, prefix_filename='frame')
                logging.debug(f"  Saved collisions to: {debug_collisions_filename}")

            dataset_path = os.path.join(odvg_dir, f'odvg_{output_frame_num:012d}.json')

            with open(dataset_path, 'w') as f:
                json.dump(json_frame_data, f, indent=3)
            logging.debug(f"  Saved ODVG to: {dataset_path}")

            # update frame count and timestamp
            frame_count += 1
            output_frame_num += 1  # Increment output frame number
            recording_frame_num += 1  # Increment recording frame number
            if frame_count % PROGRESS_LOG_INTERVAL == 0:
                logging.info(f"Generated {frame_count} frames, timestamp={timestamp:.3f}, idx={frame_idx}")
            timestamp += log_delta
        
        instance_path = os.path.join(output_dir, 'instances.json')
        with open(instance_path, 'w') as f:
            json.dump(instance_data, f, indent=3)
        logging.info(f"Saved instance to: {instance_path}")

        object_path = os.path.join(output_dir, 'objects.json')
        with open(object_path, 'w') as f:
            json.dump(object_data, f, indent=3)
        logging.info(f"Saved object to: {object_path}")

        logging.info("="*50)
        logging.info("Data generation completed!")
        logging.info(f"  Output directory: {output_dir}")
        logging.info(f"  Total frames processed: {frame_count}")
        logging.info("="*50)

    finally:
        logging.info('Cleaning up actors...')
        # Clean up all sensors
        for sensor_type, sensor in sensors.items():
            sensor.stop()
            sensor.destroy()
            logging.debug(f'Destroyed {sensor_type} sensor')
        
        client.stop_replayer(keep_actors=False)
        
        # Generate videos if requested
        if args.generate_videos:
            logging.info("="*50)
            logging.info("Generating videos from frames...")
            logging.info("="*50)
            
            try:
                process_scene_directory(
                    scene_dir=output_dir,
                    output_base=output_dir,
                    fps=target_fps,  # Use target FPS, not effective FPS
                    use_local_videos_dir=True
                )
                logging.info("Videos generated successfully!")
                logging.info(f"  Video directory: {os.path.join(output_dir, 'videos')}")
            except Exception as e:
                logging.error(f"Video generation failed: {e}")
                logging.info("You can manually generate videos later using:")
                logging.info(f"  python frames_to_videos.py -i {output_dir} -o {os.path.join(output_dir, 'videos')}")
        
        # Generate non-collision SDG events
        odvg_dir = os.path.join(output_dir, 'odvg')
        if os.path.exists(odvg_dir) and any(f.startswith('odvg_') and f.endswith('.json') for f in os.listdir(odvg_dir)):
            logging.info("Generating non-collision SDG events...")
            run_non_collision_sdg(odvg_dir, output_dir, num_closest=1, num_random=1, num_close_colli=1)
        
        logging.info('All done!')

if __name__ == '__main__':
    main()
