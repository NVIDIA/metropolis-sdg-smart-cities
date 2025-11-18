#!/usr/bin/env python3

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

"""Overlay ODVG annotations onto video files."""

import argparse
import json
import logging
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from main import (
    INSTANCE_COLORMAP,
    SEMANTIC_MAP,
    draw_bbox2d_on_frame,
    draw_bbox3d_on_frame,
    draw_polygon_mask,
)


def load_odvg_data(odvg_dir: str) -> Dict:
    """Load ODVG JSON files and organize by frame_id."""
    json_files = sorted(glob(f"{odvg_dir}/odvg_*.json"))
    if not json_files:
        logging.warning(f"No ODVG files found in {odvg_dir}")
        return {}
    
    logging.info(f"Found {len(json_files)} ODVG files")
    frame_data = {}
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            if frame_id := data.get('frame_id'):
                frame_data[frame_id] = data
    return frame_data


def get_semantic_label_id(label_name: str) -> int:
    """Map semantic label name to ID."""
    return next((lid for lid, (name, _) in SEMANTIC_MAP.items() 
                 if name.lower() == label_name.lower()), 0)


def overlay_2d_bboxes(frame: np.ndarray, detections: List[Dict], thickness: int = 2, 
                      collision_actor_ids: List[int] = None, area_threshold: int = 0) -> tuple:
    """Overlay 2D bounding boxes using main.py style.
    
    Returns:
        tuple: (modified_frame, num_overlays_drawn)
    """
    overlays_drawn = 0
    for det in detections:
        if 'bbox_2d' not in det:
            continue
        
        actor_id = det.get('object-id', 0)
        bbox_2d = det['bbox_2d']
        
        # Skip small bboxes below area threshold (reduces clutter from distant objects)
        if area_threshold > 0:
            area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
            if area < area_threshold:
                continue
        
        # Apply actor ID filter if collision-specific visualization is requested
        if collision_actor_ids is not None and actor_id not in collision_actor_ids:
            continue
        
        bbox_data = {
            '2d': {
                'bbox_2d': bbox_2d,
                'actor_id': actor_id,
                'semantic_label': get_semantic_label_id(det.get('category', 'unknown')),
                'mask_polygons': det.get('mask', [])
            }
        }
        frame = draw_bbox2d_on_frame(frame, bbox_data, thickness=thickness)
        overlays_drawn += 1
    return frame, overlays_drawn


def overlay_3d_bboxes(frame: np.ndarray, detections: List[Dict], thickness: int = 2,
                      collision_actor_ids: List[int] = None, area_threshold: int = 0) -> tuple:
    """Overlay 3D bounding boxes (requires projection data in ODVG).
    
    Returns:
        tuple: (modified_frame, num_overlays_drawn)
    """
    overlays_drawn = 0
    for det in detections:
        actor_id = det.get('object-id', 0)
        
        # Skip small bboxes below area threshold (reduces clutter from distant objects)
        if area_threshold > 0 and 'bbox_2d' in det:
            bbox_2d = det['bbox_2d']
            area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
            if area < area_threshold:
                continue
        
        # Apply actor ID filter if collision-specific visualization is requested
        if collision_actor_ids is not None and actor_id not in collision_actor_ids:
            continue
        
        if 'bbox_3d' in det and 'bbox_2d' in det:
            logging.debug(f"3D bbox for actor {actor_id} needs projection data")
            overlays_drawn += 1
    return frame, overlays_drawn


def calculate_mask_centroid(mask_polygons: List[List[float]], bbox_2d: tuple) -> tuple:
    """Calculate centroid from mask polygons or fallback to bbox center."""
    coords = [(poly[i], poly[i+1]) 
              for poly in mask_polygons for i in range(0, len(poly)-1, 2)]
    
    if coords:
        x_coords, y_coords = zip(*coords)
        return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)
    return (bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2


def overlay_masks(frame: np.ndarray, detections: List[Dict], 
                  alpha: float = 0.4, thickness: int = 2,
                  collision_actor_ids: List[int] = None, area_threshold: int = 0) -> tuple:
    """Overlay instance masks with labels using main.py style.
    
    Returns:
        tuple: (modified_frame, num_overlays_drawn)
    """
    overlays_drawn = 0
    for det in detections:
        if not (mask_polygons := det.get('mask', [])):
            continue
        
        actor_id = det.get('object-id', 0)
        
        # Skip small masks below area threshold (reduces clutter from distant objects)
        if area_threshold > 0 and 'bbox_2d' in det:
            bbox_2d = det['bbox_2d']
            area = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
            if area < area_threshold:
                continue
        
        # Apply actor ID filter if collision-specific visualization is requested
        if collision_actor_ids is not None and actor_id not in collision_actor_ids:
            continue
        
        instance_color = INSTANCE_COLORMAP[actor_id % len(INSTANCE_COLORMAP)]
        bgr_color = tuple(int(instance_color[2-i]) for i in range(3))
        
        frame = draw_polygon_mask(frame, mask_polygons, bgr_color, alpha, thickness)
        overlays_drawn += 1
        
        if bbox_2d := det.get('bbox_2d'):
            cx, cy = calculate_mask_centroid(mask_polygons, bbox_2d)
            label = f"{actor_id}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx, ty = int(cx - tw/2), int(cy + th/2)
            
            cv2.rectangle(frame, (tx-2, ty-th-2), (tx+tw+2, ty+2), bgr_color, -1)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame, overlays_drawn


def process_video(video_path: str, odvg_dir: str, output_path: str,
                  overlay_type: str = '2d', bbox_thickness: int = 2,
                  mask_alpha: float = 0.4, show_masks: bool = False,
                  collision_actor_ids: List[int] = None, area_threshold: int = 0) -> None:
    """Process video and overlay ODVG annotations."""
    frame_data = load_odvg_data(odvg_dir)
    if not frame_data:
        logging.error("No frame data loaded")
        return
    
    if collision_actor_ids:
        logging.info(f"Filtering overlays for collision actor IDs: {collision_actor_ids}")
    
    if area_threshold > 0:
        logging.info(f"Filtering overlays by bbox area threshold: {area_threshold} pixels")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Processing {width}x{height} @ {fps}fps, {total_frames} frames")
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        logging.error(f"Failed to create output: {output_path}")
        cap.release()
        return
    
    frame_idx = processed_count = total_overlays = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id = f"frame_{frame_idx:012d}"
            frame_overlays = 0
            if frame_id in frame_data:
                detections = frame_data[frame_id].get('detections', {}).get('instances', [])
                if detections:
                    if overlay_type in ['2d', 'all']:
                        frame, count = overlay_2d_bboxes(frame, detections, bbox_thickness, collision_actor_ids, area_threshold)
                        frame_overlays += count
                    if overlay_type in ['3d', 'all']:
                        frame, count = overlay_3d_bboxes(frame, detections, bbox_thickness, collision_actor_ids, area_threshold)
                        frame_overlays += count
                    if overlay_type in ['masks', 'all'] or show_masks:
                        frame, count = overlay_masks(frame, detections, mask_alpha, bbox_thickness, collision_actor_ids, area_threshold)
                        frame_overlays += count
                    if frame_overlays > 0:
                        processed_count += 1
                        total_overlays += frame_overlays
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logging.debug(f"Processed {frame_idx}/{total_frames} ({processed_count} frames with {total_overlays} overlays)")
    
    finally:
        cap.release()
        out.release()
        logging.info(f"Processed {frame_idx} frames total")
    
    # Verify sufficient overlays were drawn to create meaningful visualization
    if total_overlays <= 1:
        # Remove output video if insufficient overlays (prevents empty/useless files)
        output_file = Path(output_path)
        if output_file.exists():
            output_file.unlink()
        if total_overlays == 0:
            logging.warning(f"  No overlays passed the filters - video not generated")
        else:
            logging.warning(f"  Only 1 overlay passed the filters - video not generated (minimum: 2)")
        logging.warning(f"  Bboxes filtered by area_threshold={area_threshold} or collision_actor_ids={collision_actor_ids}")
        return
    
    logging.info(f"Completed: {frame_idx} frames, {processed_count} with overlays ({total_overlays} total) -> {output_path}")


def main():
    """CLI entry point for ODVG video overlay tool."""
    parser = argparse.ArgumentParser(description='Overlay ODVG annotations onto videos')
    parser.add_argument('-i', '--input-video', required=True, help='Input video file')
    parser.add_argument('-d', '--odvg-dir', required=True, help='ODVG directory')
    parser.add_argument('-o', '--output-video', required=True, help='Output video file')
    parser.add_argument('-t', '--overlay-type', choices=['2d', '3d', 'masks', 'all'], 
                       default='2d', help='Overlay type (default: 2d)')
    parser.add_argument('--bbox-thickness', type=int, default=2, help='Bbox line thickness')
    parser.add_argument('--mask-alpha', type=float, default=0.4, help='Mask transparency (0-1)')
    parser.add_argument('--show-masks', action='store_true', help='Add masks to bbox overlays')
    parser.add_argument('--collision-actor-ids', type=int, nargs='+', 
                       dest='collision_actor_ids', default=None,
                       help='Filter overlays for specific actor IDs (space-separated list, e.g., --collision-actor-ids 441 430)')
    parser.add_argument('-at', '--area-threshold', type=int, default=0,
                       dest='area_threshold',
                       help='Filter bboxes smaller than this area in pixels (default: 0, no filtering)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    # Validate required input paths exist
    if not Path(args.input_video).exists():
        logging.error(f"Video not found: {args.input_video}")
        sys.exit(1)
    if not Path(args.odvg_dir).is_dir():
        logging.error(f"Directory not found: {args.odvg_dir}")
        sys.exit(1)
    
    # Ensure output directory structure exists
    Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
    
    process_video(args.input_video, args.odvg_dir, args.output_video,
                  args.overlay_type, args.bbox_thickness, args.mask_alpha, args.show_masks,
                  args.collision_actor_ids, args.area_threshold)


if __name__ == '__main__':
    main()