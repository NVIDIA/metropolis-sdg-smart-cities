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

"""
Script to convert frame sequences to videos using OpenCV.
Processes all frame types (rgb, depth, segmentation, etc.) and saves videos to videos/ directory.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import glob
from typing import List, Tuple


def get_frame_numbers(frame_files: List[str]) -> List[int]:
    """Extract and sort frame numbers from filenames."""
    frame_nums = []
    for f in frame_files:
        basename = os.path.basename(f)
        # Extract number from filename like 'frame_000001.jpg' or 'frame_000001.png'
        try:
            num = int(basename.split('_')[1].split('.')[0])
            frame_nums.append(num)
        except:
            continue
    return sorted(frame_nums)


def create_video_from_frames(input_dir: str, output_path: str, frame_type: str, fps: int = 30, codec: str = 'mp4v', start_frame: int = 0, duration: int = 0) -> bool:
    """
    Create a video from a sequence of image frames.
    
    Args:
        input_dir: Directory containing frame images
        output_path: Path for the output video file
        fps: Frames per second for the output video
        codec: Video codec to use (default: mp4v)
        start_frame: Starting frame index (0-based)
        duration: Number of frames to process (0 = all frames from start)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Find all image files (supports both jpg and png)
    string_jpg_pattern = "frame_*.jpg"
    string_png_pattern = "frame_*.png"

    if frame_type == "depth":
        string_png_pattern = "depth_*.png"
        string_jpg_pattern = "depth_*.jpg"
    if frame_type == "edges":
        string_png_pattern = "edge_*.png"
        string_jpg_pattern = "edge_*.jpg"
    if frame_type == "semantic_segmentation" or frame_type == "semantic_segmentation_raw" or frame_type == "instance_segmentation" or frame_type == "instance_segmentation_raw":
        string_png_pattern = "seg_*.png"
        string_jpg_pattern = "seg_*.jpg"
    
    pattern_jpg = os.path.join(input_dir, string_jpg_pattern)
    pattern_png = os.path.join(input_dir, string_png_pattern)
    
    frame_files = sorted(glob.glob(pattern_jpg) + glob.glob(pattern_png))
    
    if not frame_files:
        print(f"No frames found in {input_dir}")
        return False
    
    # Get frame numbers to ensure proper ordering
    frame_nums = get_frame_numbers(frame_files)
    if not frame_nums:
        print(f"Could not extract frame numbers from {input_dir}")
        return False
    
    # Apply start frame and duration filters
    if start_frame > 0:
        frame_nums = [num for num in frame_nums if num >= start_frame]
        if not frame_nums:
            print(f"No frames found starting from frame {start_frame}")
            return False
    
    if duration > 0:
        # Calculate end frame based on duration
        end_frame = start_frame + duration
        frame_nums = [num for num in frame_nums if num < end_frame]
        if not frame_nums:
            print(f"No frames found in range {start_frame} to {end_frame}")
            return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Could not read first frame from {frame_files[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Could not open video writer for {output_path}")
        return False
    
    print(f"Creating video {output_path} ({width}x{height}) from {len(frame_nums)} frames (frames {frame_nums[0]}-{frame_nums[-1]})...")
    
    # Process frames in order
    processed = 0
    for num in frame_nums:
        # Find the frame file for this number
        frame_file = None
        for f in frame_files:
            frame_format = f"frame_{num:012d}."
            if frame_type == "depth":
                frame_format = f"depth_{num:012d}."
            if frame_type == "edges":
                frame_format = f"edge_{num:012d}."
            if frame_type == "semantic_segmentation" or frame_type == "semantic_segmentation_raw" or frame_type == "instance_segmentation" or frame_type == "instance_segmentation_raw":
                frame_format = f"seg_{num:012d}."
                
            if frame_format in f:
                frame_file = f
                break
        
        if frame_file is None:
            print(f"Warning: Missing frame {num:012d}")
            continue
        
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
        
        # Handle special cases for different frame types
        if 'depth' in input_dir:
            # For depth images, we might want to normalize and colorize
            # Assuming depth is stored as 16-bit PNG
            if frame.dtype == np.uint16:
                # Normalize to 8-bit for visualization
                frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # Apply colormap for better visualization
                frame = cv2.applyColorMap(frame_norm, cv2.COLORMAP_JET)
            elif len(frame.shape) == 2 or frame.shape[2] == 1:
                # Grayscale depth - apply colormap
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        
        elif 'segmentation' in input_dir and 'raw' not in input_dir:
            # Segmentation images might need special handling
            # They're usually already colored, so we just ensure they're in the right format
            if len(frame.shape) == 2:
                # If grayscale, convert to color
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        elif 'normals' in input_dir:
            # Normal maps are typically stored as RGB where R,G,B represent X,Y,Z components
            # They should display correctly as-is
            pass
        
        # Write frame to video
        out.write(frame)
        processed += 1
        
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(frame_files)} frames...")
    
    # Release everything
    out.release()
    print(f"Successfully created video with {processed} frames: {output_path}")
    return True


def process_scene_directory(scene_dir: str, output_base: str, fps: int = 30, use_local_videos_dir: bool = False, start_frame: int = 0, duration: int = 0):
    """
    Process all frame directories in a scene and create corresponding videos.
    
    Args:
        scene_dir: Directory containing frame subdirectories (e.g., output/harbor_road)
        output_base: Base directory for output videos
        fps: Frames per second for videos
        use_local_videos_dir: If True, save videos in scene_dir/videos instead of output_base
        start_frame: Starting frame index (0-based)
        duration: Number of frames to process (0 = all frames from start)
    """
    if use_local_videos_dir:
        # Save videos in a 'videos' subdirectory within the scene directory
        output_dir = os.path.join(scene_dir, 'videos')
    else:
        # Handle multi-level paths for camera directories
        path_parts = scene_dir.rstrip('/').split('/')
        if 'cam' in path_parts[-1] and len(path_parts) >= 2:
            # For multi-camera scenes, use scene_camera format
            scene_name = '_'.join(path_parts[-2:])
        else:
            scene_name = os.path.basename(scene_dir)
        
        output_dir = os.path.join(output_base, scene_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Frame types to process
    frame_types = [
        'rgb',
        'rgb_with_bboxes',
        'rgb_with_3d_bboxes',
        'rgb_with_masks',
        'depth',
        'semantic_segmentation',
        'semantic_segmentation_raw',
        'instance_segmentation',
        'instance_segmentation_raw',
        'normals',
        'edges',
        'masks',
        'rgb_with_collisions'
    ]
    
    # Process each frame type
    for frame_type in frame_types:
        input_dir = os.path.join(scene_dir, frame_type)
        
        if not os.path.exists(input_dir):
            print(f"Skipping {frame_type} - directory not found")
            continue
        
        output_path = os.path.join(output_dir, f"{frame_type}.mp4")
        success = create_video_from_frames(input_dir, output_path,frame_type, fps, 'mp4v', start_frame, duration)
        
        if not success:
            print(f"Failed to create video for {frame_type}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert CARLA ground truth frames to videos"
    )
    parser.add_argument(
        "--input", "-i",
        default="output",
        help="Input directory containing scene folders (default: output)"
    )
    parser.add_argument(
        "--output", "-o",
        default="videos",
        help="Output directory for videos (default: videos)"
    )
    parser.add_argument(
        "--scene", "-s",
        help="Process only a specific scene (e.g., harbor_road)"
    )
    parser.add_argument(
        "--camera-dir", "-c",
        help="Process a specific camera directory directly (e.g., output/multi_camera_scene1/cam1)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output videos (default: 30)"
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="Video codec to use (default: mp4v)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index (0-based, default: 0)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=0,
        help="Number of frames to process (0 = all frames from start, default: 0)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.camera_dir:
        # Process specific camera directory directly
        camera_dir = args.camera_dir
        if not os.path.exists(camera_dir):
            print(f"Error: Camera directory not found: {camera_dir}")
            return
        
        # Extract a meaningful name from the path
        path_parts = camera_dir.rstrip('/').split('/')
        if len(path_parts) >= 2:
            # For paths like output/multi_camera_scene1/cam1, use multi_camera_scene1_cam1
            output_name = '_'.join(path_parts[-2:])
        else:
            output_name = path_parts[-1]
        
        print(f"Processing camera directory: {camera_dir}")
        # For camera directories, save videos locally
        process_scene_directory(camera_dir, args.output, args.fps, use_local_videos_dir=True, 
                              start_frame=args.start_frame, duration=args.duration)
        print(f"\nVideos saved to: {camera_dir}/videos/")
        
    elif args.scene:
        # Process specific scene
        scene_dir = os.path.join(args.input, args.scene)
        if not os.path.exists(scene_dir):
            print(f"Error: Scene directory not found: {scene_dir}")
            return
        print(f"Processing scene: {args.scene}")
        process_scene_directory(scene_dir, args.output, args.fps, start_frame=args.start_frame, duration=args.duration)
    else:
        # Process all scenes
        for scene_name in os.listdir(args.input):
            scene_dir = os.path.join(args.input, scene_name)
            if os.path.isdir(scene_dir):
                print(f"\nProcessing scene: {scene_name}")
                print("="*50)
                process_scene_directory(scene_dir, args.output, args.fps, start_frame=args.start_frame, duration=args.duration)


if __name__ == "__main__":
    main()
