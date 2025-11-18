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
Integrated hard sample generation pipeline for CARLA ODVG data.
Combines playback creation, closest trajectory pair finding, and event splitting into one file.
"""

import os
import argparse
import json
import logging
import numpy as np
import random
from datetime import datetime, timedelta, timezone

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Bbox filtering thresholds (pixels)
MIN_BBOX_HEIGHT = 28
MIN_BBOX_WIDTH = 28
MIN_BBOX_AREA = 36 * 36  # 1296 pixels

# ============================================================================
# UTILITY FUNCTIONS - Timestamp parsing and data loading helpers
# ============================================================================

def parse_iso8601_timestamp(timestamp_str: str) -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware datetime."""
    if timestamp_str.endswith("Z"):
        iso_str = timestamp_str[:-1] + "+00:00"
    else:
        iso_str = timestamp_str
    try:
        return datetime.fromisoformat(iso_str)
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def load_all_object(event_file_path: str) -> set:
    """Load all object IDs from an event file."""
    with open(event_file_path, "r") as f:
        events = json.load(f)
    objects = set()
    for event in events:
        object_list = event["objects"]
        for object_id in object_list:
            objects.add(object_id)
    return objects


def load_all_pair(event_file_path: str) -> set:
    """Load all object pairs from an event file."""
    with open(event_file_path, "r") as f:
        events = json.load(f)
    object_pairs = set()
    for event in events:
        object_list = event["objects"]
        for i in range(len(object_list)):
            for j in range(len(object_list)):
                if i == j:
                    continue
                object_pairs.add((object_list[i], object_list[j]))
    return object_pairs


# ============================================================================
# STEP 1: CREATE PLAYBACK FROM ODVG - Convert ODVG frames to playback format
# ============================================================================

# Vehicle types to include in event generation (filters out pedestrians, bikes, etc.)
ALLOWED_OBJECT_TYPES = ["truck", "car", "bus"]


def filter_object_by_avg_bbox(
    bboxes: list, 
    height_threshold: int, 
    width_threshold: int,
    area_threshold: int
) -> bool:
    """
    Filter objects based on average bbox dimensions across frames.
    Returns True if object meets minimum size requirements (should be kept).
    """
    height_avg = 0
    width_avg = 0
    for bbox in bboxes:
        height_avg += bbox[3] - bbox[1]
        width_avg += bbox[2] - bbox[0]
    height_avg = int(height_avg / len(bboxes))
    width_avg = int(width_avg / len(bboxes))
    area_avg = int(height_avg * width_avg)
    return height_avg > height_threshold and width_avg > width_threshold and area_avg > area_threshold


def filter_playback(frames_data: list, collision_objects_per_video: dict) -> list:
    """
    Filter out small/distant objects from playback data while preserving collision participants.
    Builds object tracking history across frames to compute average bbox sizes.
    """
    objects_set = {}
    # Collect all bboxes for each object across all frames
    for frame_data in frames_data:
        sensor_id = frame_data["sensorId"]
        if sensor_id not in objects_set:
            objects_set[sensor_id] = {}
        for object_data in frame_data["objects"]:
            object_id, x1, y1, x2, y2, vehicle_type = object_data.split("#")[0].split("|")[:-1]
            if object_id not in objects_set[sensor_id]:
                objects_set[sensor_id][object_id] = []
            objects_set[sensor_id][object_id].append((float(x1), float(y1), float(x2), float(y2)))
    
    filtered_objects = {}
    for sensor_id in objects_set.keys():
        filtered_objects[sensor_id] = set()
        collision_objects = collision_objects_per_video[sensor_id]
        for object_id in objects_set[sensor_id].keys():
            if not filter_object_by_avg_bbox(
                objects_set[sensor_id][object_id], 
                MIN_BBOX_HEIGHT, 
                MIN_BBOX_WIDTH, 
                MIN_BBOX_AREA
            ) and object_id not in collision_objects:
                filtered_objects[sensor_id].add(object_id)
                logging.debug(f"{sensor_id} Filtered object {object_id}")
    
    for sensor_id in filtered_objects.keys():
        logging.debug(f"{sensor_id} Total filtered objects: {len(filtered_objects[sensor_id])}")
    
    filtered_frames_data = []
    for frame_data in frames_data:
        sensor_id = frame_data["sensorId"]
        new_frame_data = {
            "version": frame_data["version"],
            "id": frame_data["id"],
            "@timestamp": frame_data["@timestamp"],
            "sensorId": sensor_id,
            "objects": []
        }
        for object_data in frame_data["objects"]:
            object_id = object_data.split("#")[0].split("|")[:-1][0]
            if object_id not in filtered_objects[sensor_id]:
                new_frame_data["objects"].append(object_data)
        filtered_frames_data.append(new_frame_data)
    
    return filtered_frames_data


def get_frame_data(data: dict, video_name: str, file_id: str, timestamp: str) -> dict:
    """Extract frame data from ODVG JSON."""
    frame_data = {
        "version": "4.0",
        "id": file_id,
        "@timestamp": timestamp,
        "sensorId": video_name,
        "objects": []
    }

    detection_data = data["detections"]["instances"]
    
    for object_data in detection_data:
        if object_data["category"] in ALLOWED_OBJECT_TYPES:
            object_id = object_data["object-id"]
            bbox_data = object_data["bbox_2d"]
            x1 = int(bbox_data[0])
            y1 = int(bbox_data[1])
            x2 = int(bbox_data[2])
            y2 = int(bbox_data[3])
            vehicle_type = "Vehicle"
            object_str = str(object_id) + "|" + str(x1) + "|" + str(y1) + "|" + str(x2) + "|" + str(y2) + "|" + vehicle_type + "|#" + "|"*7 + "0"
            frame_data["objects"].append(object_str)
    
    return frame_data


def create_playback_from_odvg(odvg_dir: str) -> tuple:
    """Create playback data from ODVG directory."""
    logging.info("="*80)
    logging.info("STEP 1: Creating playback from ODVG")
    logging.info("="*80)
    
    start_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    frames_data = []
    collision_objects_per_video = {}
    collision_pairs_per_video = {}
    first_timestamp_per_video = {}
    
    # Treat the folder as a single scene with flat structure
    video_name = os.path.basename(odvg_dir.rstrip('/'))
    logging.info(f"Processing flat folder structure: {odvg_dir} as scene '{video_name}'")
    
    collision_objects_per_video[video_name] = set()
    collision_pairs_per_video[video_name] = set()
    frame_files = []
    first_timestamp_per_video[video_name] = None
    
    # Look for collision events in parent directory (output_dir)
    parent_dir = os.path.dirname(odvg_dir.rstrip('/'))
    collision_file_path = os.path.join(parent_dir, "events_collision.json")
    if os.path.exists(collision_file_path):
        collision_objects_per_video[video_name].update(load_all_object(collision_file_path))
        collision_pairs_per_video[video_name].update(load_all_pair(collision_file_path))
        logging.info(f"Loaded {len(collision_objects_per_video[video_name])} collision objects and {len(collision_pairs_per_video[video_name])} collision pairs from: {collision_file_path}")
    
    # Collect all odvg_*.json files
    files_in_dir = os.listdir(odvg_dir)
    for file in files_in_dir:
        if file.endswith(".json") and file.startswith("odvg_"):
            frame_files.append(os.path.join(odvg_dir, file))
    
    frame_files = sorted(frame_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    logging.info(f"Found {len(frame_files)} ODVG frames")
    
    first_timestamp_data = None
    
    for file in frame_files:
        file_id = file.split("_")[-1].split(".")[0]
        with open(file, "r") as f:
            data = json.load(f)
            timestamp_data = data["timestamp"]
            if first_timestamp_data is None:
                first_timestamp_data = timestamp_data
                first_timestamp_per_video[video_name] = start_timestamp
            time_delta = timestamp_data - first_timestamp_data  # seconds
            timestamp = datetime.strptime(start_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(milliseconds=1000*time_delta)
            timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            frame_data = get_frame_data(data, video_name, file_id, timestamp)
            frames_data.append(frame_data)
    
    total_objects = 0
    for frame_data in frames_data:
        total_objects += len(frame_data["objects"])
    
    frames_data = filter_playback(frames_data, collision_objects_per_video)
    logging.info(f"Before filtering: {total_objects} objects")
    total_objects = 0
    for frame_data in frames_data:
        total_objects += len(frame_data["objects"])
    logging.info(f"After filtering: {total_objects} objects")
    
    return frames_data, first_timestamp_per_video, collision_objects_per_video, collision_pairs_per_video


# ============================================================================
# STEP 2: FIND CLOSEST TRAJECTORY PAIRS - Generate hard negative samples
# ============================================================================

def calculate_euclidean_distance_vectorized(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Calculate the Euclidean distances between two sets of points."""
    differences = points1 - points2
    squared_differences = differences**2
    sum_squared_differences = np.sum(squared_differences, axis=1)
    return np.sqrt(sum_squared_differences)


def parse_frame(frame: dict) -> list:
    """Parse frame objects into structured format."""
    objects = []
    for object_data in frame["objects"]:
        object_id, x1, y1, x2, y2, vehicle_type = object_data.split("#")[0].split("|")[:-1]
        objects.append((object_id, float(x1), float(y1), float(x2), float(y2), vehicle_type))
    return objects


def create_pairs(objects: list) -> tuple:
    """Create all possible pairs of objects in a frame."""
    points1 = []
    points2 = []
    pair_ids = []
    min_dim_sums = []
    area_pairs = []
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            object_id, x1, y1, x2, y2, vehicle_type = objects[i]
            object_id2, x12, y12, x22, y22, vehicle_type2 = objects[j]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            x_center2 = (x12 + x22) / 2
            y_center2 = (y12 + y22) / 2
            points1.append((x_center, y_center))
            points2.append((x_center2, y_center2))
            pair_ids.append((object_id, object_id2))
            # sum of min(width, height) for both boxes
            width1 = max(0.0, x2 - x1)
            height1 = max(0.0, y2 - y1)
            width2 = max(0.0, x22 - x12)
            height2 = max(0.0, y22 - y12)
            min_dim_sums.append((min(width1, height1) + min(width2, height2))/2)
            area_pairs.append((width1, height1, width2, height2))
    return points1, points2, pair_ids, min_dim_sums, area_pairs


def gen_event(event_index: int, objects_set: dict, pair_id: tuple) -> list:
    """Generate a single event for a pair of objects."""
    events = []
    event_id = f"event_{event_index:06d}"
    object_id1, object_id2 = pair_id
    start_time = min(objects_set[object_id1]["start_time"], objects_set[object_id2]["start_time"])
    end_time = max(objects_set[object_id1]["end_time"], objects_set[object_id2]["end_time"])
    event = {
        "event_id": event_id,
        "start_time": start_time,
        "end_time": end_time,
        "category": "normal",
        "sub_category": ["close by"],
        "objects": [object_id1, object_id2],
        "event_caption": ""
    }
    events.append(event)
    return events


def gen_random_event(
    event_index: int,
    objects_set: dict[str, dict[str, dict[str, float]]],
    pair_ids: list[tuple[str, str]],
    abandon_objects: set[str],
    abandon_pairs: set[tuple[str, str]],
    collision_pairs: set[tuple[str, str]]
) -> list[dict]:
    """Generate a random event from available pairs."""
    events = []
    event_id = f"event_{event_index:06d}"
    available_pair_ids = []
    n_pair_ids = len(pair_ids)
    for i in range(n_pair_ids):
        object_id1, object_id2 = pair_ids[i]
        if object_id1 in abandon_objects or object_id2 in abandon_objects:
            continue
        pair_id = pair_ids[i]
        dup_pair_id = (pair_id[1], pair_id[0])
        # Skip if this specific pair had a collision (check both orderings)
        if pair_id in collision_pairs or dup_pair_id in collision_pairs:
            continue
        if pair_id not in abandon_pairs:
            available_pair_ids.append(pair_id)
    if len(available_pair_ids) == 0:
        return events
    candidate_pair_id = random.choice(available_pair_ids)
    object_id1, object_id2 = candidate_pair_id
    start_time = min(objects_set[object_id1]["start_time"], objects_set[object_id2]["start_time"])
    end_time = max(objects_set[object_id1]["end_time"], objects_set[object_id2]["end_time"])
    event = {
        "event_id": event_id,
        "start_time": start_time,
        "end_time": end_time,
        "category": "normal",
        "sub_category": ["random"],
        "objects": [object_id1, object_id2],
        "event_caption": ""
    }
    events.append(event)
    return events


def gen_closest_events(
    event_index: int,
    objects_set: dict,
    pair_ids: list,
    distances: list,
    area_pairs: list,
    num_closest_event: int,
    exist_pairs: set,
    collision_objects: set,
    collision_pairs: set,
    sensor_id: str
) -> tuple:
    """Generate closest trajectory pair events."""
    top_k_events = []
    top_k_areas = []
    sorted_indices = np.argsort(distances)
    unique_pair_ids = set()
    top_k = min(num_closest_event * 2, len(pair_ids))
    # Generate closest events
    if len(exist_pairs) >= 2*num_closest_event:
        return top_k_events, event_index

    for index in sorted_indices:
        if len(unique_pair_ids) >= 2*top_k:
            break
        if len(unique_pair_ids) >= 2*num_closest_event and distances[index] >= 1.0:
            break
        pair_id = pair_ids[index]
        dup_pair_id = (pair_id[1], pair_id[0])
        object_id1, object_id2 = pair_id
        # Skip if this specific pair had a collision (check both orderings)
        if pair_id in collision_pairs or dup_pair_id in collision_pairs:
            continue
        if pair_id not in unique_pair_ids and pair_id not in exist_pairs:
            area_1 = area_pairs[index][0] * area_pairs[index][1]
            area_2 = area_pairs[index][2] * area_pairs[index][3]
            area_sum = area_1 + area_2
            top_k_areas.append(area_sum)
            unique_pair_ids.add(pair_id)
            unique_pair_ids.add(dup_pair_id)
            logging.debug(f"{sensor_id} pair {pair_id} distance: {distances[index]:.2f}")
            top_k_events.extend(gen_event(event_index, objects_set[sensor_id], pair_id))
            event_index += 1
    # Sort top_k_events by area_sum
    sorted_indices = np.argsort(top_k_areas)[::-1]
    events = []
    n_event = 0
    for index in sorted_indices:
        event = top_k_events[index]
        events.append(event)
        n_event += 1
        if n_event >= num_closest_event:
            break
    return events, event_index


def gen_random_events(
    event_index: int,
    objects_set: dict,
    pair_ids: list,
    distances: list,
    collision_objects: set,
    exist_pairs: set,
    collision_pairs: set,
    num_random_event: int,
    sensor_id: str
) -> tuple:
    """Generate random events."""
    events = []
    sorted_indices = np.argsort(distances)
    unique_pair_ids = set()
    unique_list = []
    for index in sorted_indices:
        pair_id = pair_ids[index]
        if pair_id not in unique_pair_ids:
            unique_pair_ids.add(pair_id)
            unique_list.append(pair_id)
    unique_list = unique_list[:len(unique_list)//2]
    for _ in range(num_random_event):
        events.extend(gen_random_event(event_index, objects_set[sensor_id], unique_list, \
            collision_objects, exist_pairs, collision_pairs))
        event_index += 1
    return events, event_index


def gen_close_colli_obj_events(
    event_index: int,
    objects_set: dict,
    pair_ids: list,
    distances: list,
    collision_objects: set,
    exist_pairs: set,
    collision_pairs: set,
    num_close_colli_obj_event: int,
    sensor_id: str
) -> tuple:
    """Generate close collision object events."""
    events = []
    close_colli_distances = []
    close_colli_pair_ids = []
    for index, pair_id in enumerate(pair_ids):
        object_id1, object_id2 = pair_id
        dup_pair_id = (pair_id[1], pair_id[0])
        # Skip if this specific pair had a collision (check both orderings)
        if pair_id in collision_pairs or dup_pair_id in collision_pairs:
            continue
        if (object_id1 in collision_objects or object_id2 in collision_objects) \
            and not (object_id1 in collision_objects and object_id2 in collision_objects):
            close_colli_distances.append(distances[index])
            close_colli_pair_ids.append(pair_id)
    
    sorted_indices = np.argsort(close_colli_distances)
    unique_pair_ids = set()
    for index in sorted_indices:
        if len(unique_pair_ids) >= num_close_colli_obj_event:
            break
        pair_id = close_colli_pair_ids[index]
        if pair_id not in unique_pair_ids and pair_id not in exist_pairs:
            unique_pair_ids.add(pair_id)
            events.extend(gen_event(event_index, objects_set[sensor_id], pair_id))
            event_index += 1
    return events, event_index


def extend_exist_pairs(exist_pairs: set, events: list) -> set:
    """Extend existing pairs with new events."""
    for event in events:
        for i in range(len(event["objects"])):
            for j in range(len(event["objects"])):
                if i == j:
                    continue
                exist_pairs.add((event["objects"][i], event["objects"][j]))
    return exist_pairs


def find_closest_trajectory_pair(
    frames_data: list,
    first_timestamp_per_video: dict,
    collision_objects_per_video: dict,
    collision_pairs_per_video: dict,
    num_closest_event: int,
    num_random_event: int,
    num_close_colli_obj_event: int,
    odvg_dir: str
) -> dict:
    """Find closest trajectory pairs and generate events."""
    logging.info("="*80)
    logging.info("STEP 2: Finding closest trajectory pairs")
    logging.info("="*80)
    
    points1_per_sensor = {}
    points2_per_sensor = {}
    pair_ids_per_sensor = {}
    area_pairs_per_sensor = {}
    denom_per_sensor = {}
    objects_set = {}

    for frame_data in frames_data:
        sensor_id = frame_data["sensorId"]
        timestamp_str = frame_data["@timestamp"]
        objects = parse_frame(frame_data)
        points1, points2, pair_ids, min_dim_sums, area_pairs = create_pairs(objects)
        
        if sensor_id not in points1_per_sensor.keys():
            points1_per_sensor[sensor_id] = []
            points2_per_sensor[sensor_id] = []
            pair_ids_per_sensor[sensor_id] = []
            denom_per_sensor[sensor_id] = []
            area_pairs_per_sensor[sensor_id] = []
        
        points1_per_sensor[sensor_id].extend(points1)
        points2_per_sensor[sensor_id].extend(points2)
        pair_ids_per_sensor[sensor_id].extend(pair_ids)
        denom_per_sensor[sensor_id].extend(min_dim_sums)
        area_pairs_per_sensor[sensor_id].extend(area_pairs)
        
        if sensor_id not in objects_set.keys():
            objects_set[sensor_id] = {}
        
        for object_data in objects:
            object_id, x1, y1, x2, y2, vehicle_type = object_data
            timestamp_dt = parse_iso8601_timestamp(timestamp_str)
            first_timestamp_dt = parse_iso8601_timestamp(first_timestamp_per_video[sensor_id])
            time_in_seconds = (timestamp_dt - first_timestamp_dt).total_seconds()
            if object_id not in objects_set[sensor_id].keys():
                objects_set[sensor_id][object_id] = {
                    "start_time": time_in_seconds,
                    "end_time": time_in_seconds,
                }
            objects_set[sensor_id][object_id]["end_time"] = time_in_seconds

    all_events = {}
    for sensor_id in points1_per_sensor.keys():
        collision_objects = collision_objects_per_video.get(sensor_id, set())
        collision_pairs = collision_pairs_per_video.get(sensor_id, set())
        # Don't add collision_pairs to exist_pairs initially, since that breaks the counting logic
        # Instead, we'll check collision_pairs separately in each generation function
        exist_pairs = set()

        events = []
        points1 = points1_per_sensor[sensor_id]
        points2 = points2_per_sensor[sensor_id]
        pair_ids = pair_ids_per_sensor[sensor_id]
        distances = calculate_euclidean_distance_vectorized(np.array(points1), np.array(points2))
        area_pairs = area_pairs_per_sensor[sensor_id]
        
        # Normalize by denominator = (min(w1,h1) + min(w2,h2))
        denom = np.array(denom_per_sensor[sensor_id], dtype=float)
        eps = 1e-6
        distances = distances / (denom + eps)
        
        event_index = 1
        
        # Generate closest events
        if len(collision_objects) > 0:
            generated_events, event_index = gen_closest_events(event_index, objects_set, pair_ids, distances, area_pairs,
                num_closest_event, exist_pairs, collision_objects, collision_pairs, sensor_id)
        else:
            generated_events, event_index = gen_closest_events(event_index, objects_set, pair_ids, distances, area_pairs,
                num_closest_event + 1, exist_pairs, collision_objects, collision_pairs, sensor_id)
        events.extend(generated_events)
        exist_pairs = extend_exist_pairs(exist_pairs, generated_events)
        logging.info(f"Generated {len(generated_events)} closest events for {sensor_id}")
            
        # Generate random events
        generated_events, event_index = gen_random_events(event_index, objects_set, pair_ids, distances,
            collision_objects, exist_pairs, collision_pairs, num_random_event, sensor_id)
        events.extend(generated_events)
        exist_pairs = extend_exist_pairs(exist_pairs, generated_events)
        logging.info(f"Generated {len(generated_events)} random events for {sensor_id}")

        # Generate close collision object events
        generated_events, event_index = gen_close_colli_obj_events(event_index, objects_set, pair_ids, distances,
            collision_objects, exist_pairs, collision_pairs, num_close_colli_obj_event, sensor_id)
        events.extend(generated_events)
        exist_pairs = extend_exist_pairs(exist_pairs, generated_events)
        logging.info(f"Generated {len(generated_events)} close collision object events for {sensor_id}")

        all_events[sensor_id] = events
        logging.info(f"Total events for {sensor_id}: {len(events)}")
    
    return all_events


# ============================================================================
# STEP 3: SPLIT AND MOVE EVENTS - Export individual event JSON files
# ============================================================================

def move_events(all_events: dict, output_dir_noncollision: str) -> None:
    """Split events into individual files."""
    logging.info("="*80)
    logging.info("STEP 3: Splitting events into individual files")
    logging.info("="*80)
    
    os.makedirs(output_dir_noncollision, exist_ok=True)
    event_counter = 1
    for video_name, events in all_events.items():
        for event in events:
            dst_event_file_path = os.path.join(output_dir_noncollision, f"events_noncollision_{event_counter}.json")
            with open(dst_event_file_path, "w", encoding="utf-8") as f:
                json.dump([event], f, indent=2, ensure_ascii=False)
                f.write("\n")
            logging.debug(f"Written: {dst_event_file_path}")
            event_counter += 1
    logging.info(f"Total event files written: {event_counter - 1}")


# ============================================================================
# MAIN PIPELINE - Orchestrates all steps for hard sample generation
# ============================================================================

def run_pipeline(
    odvg_dir: str,
    output_dir_noncollision: str,
    num_closest_event: int,
    num_random_event: int,
    num_close_colli_obj_event: int
):
    """Run the complete hard sample generation pipeline."""
    logging.info("="*80)
    logging.info("HARD SAMPLE GENERATION PIPELINE")
    logging.info("="*80)
    logging.info(f"ODVG Directory: {odvg_dir}")
    logging.info(f"Output Directory: {output_dir_noncollision}")
    logging.info(f"Closest Events: {num_closest_event}")
    logging.info(f"Random Events: {num_random_event}")
    logging.info(f"Close Collision Object Events: {num_close_colli_obj_event}")
    
    # Step 1: Create playback from ODVG
    frames_data, first_timestamp_per_video, collision_objects_per_video, collision_pairs_per_video = create_playback_from_odvg(odvg_dir)
    logging.debug("Playback data kept in memory only, not saved to disk")
    
    # Step 2: Find closest trajectory pairs
    all_events = find_closest_trajectory_pair(
        frames_data,
        first_timestamp_per_video,
        collision_objects_per_video,
        collision_pairs_per_video,
        num_closest_event,
        num_random_event,
        num_close_colli_obj_event,
        odvg_dir
    )
    logging.debug("Combined events kept in memory only, not saved to disk")
    
    # Step 3: Split events into individual files (only output saved to disk)
    move_events(all_events, output_dir_noncollision)
    
    logging.info("="*80)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated hard sample generation pipeline")
    parser.add_argument("--odvg_dir", type=str, required=True, 
                       help="Input ODVG folder with odvg_*.json files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for individual event files (events_noncollision_*.json)")
    parser.add_argument("--num_closest_event", type=int, default=1,
                       help="Number of closest trajectory pairs per scene")
    parser.add_argument("--num_random_event", type=int, default=1,
                       help="Number of random events per scene")
    parser.add_argument("--num_close_colli_obj_event", type=int, default=1,
                       help="Number of close-collision-object events per scene")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging with production-ready format
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=log_level
    )
    
    run_pipeline(
        odvg_dir=args.odvg_dir,
        output_dir_noncollision=args.output_dir,
        num_closest_event=args.num_closest_event,
        num_random_event=args.num_random_event,
        num_close_colli_obj_event=args.num_close_colli_obj_event
    )

