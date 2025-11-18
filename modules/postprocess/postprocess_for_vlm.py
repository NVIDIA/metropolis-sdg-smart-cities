# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 
import json
import glob
import os
import argparse
import random

QUESTION_TEMPLATE = (
    "Is there a car collision between vehicles with numeric IDs {ID_1} and {ID_2}? "
    "Your final answer should be either Yes or No."
)

def prepare_single_annotation(item_id, video_path, id_list, gt_answer):
    question = QUESTION_TEMPLATE.format(ID_1=id_list[0], ID_2=id_list[1])
    item = {
        "id": item_id,
        "video": video_path,
        "conversations": [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": gt_answer
            }
        ]
    }
    return item


def prepare_cosmos_annotations(carla_folder, cosmos_folder, run_id):
    annotations = []

    carla_scenarios = glob.glob(os.path.join(carla_folder, run_id, "scenario*"), recursive=True)
    for scenario in carla_scenarios:
        scenario_name = os.path.basename(scenario)
        run_name = os.path.basename(os.path.dirname(scenario))

        collision_event_file = os.path.join(scenario, "events_collision.json")
        if not os.path.exists(collision_event_file):
            continue
        with open(collision_event_file, "r") as f:
            collision_event = json.load(f)[0]['objects']

        non_collision_lst = []
        
        non_collision_index = 1
        while non_collision_index:
            non_collision_event_file = os.path.join(scenario, f"events_noncollision_{non_collision_index}.json")
            if not os.path.exists(non_collision_event_file):
                break

            with open(non_collision_event_file, "r") as f:
                non_collision_events = json.load(f)
                non_collision_lst.append(non_collision_events[0]['objects'])
            non_collision_index += 1
        
        cosmos_scenario = os.path.join(cosmos_folder, run_name, scenario_name)
        for augmentation_folder in os.listdir(cosmos_scenario):
            if not os.path.isdir(os.path.join(cosmos_scenario, augmentation_folder)):
                continue
            augmentation_name = os.path.basename(augmentation_folder)
            augmentation_path = os.path.join(cosmos_scenario, augmentation_folder)
            
            collision_video_path = os.path.join(augmentation_path, "events_collision_som.mp4")
            if not os.path.exists(collision_video_path):
                print(f"Collision video path not found: {collision_video_path}")
                continue
            annotations.append(prepare_single_annotation(collision_video_path, collision_video_path, collision_event, "Yes"))

            for i, non_collision_obj in enumerate(non_collision_lst):
                video_path = os.path.join(augmentation_path, f"events_noncollision_{i+1}_som.mp4")
                if not os.path.exists(video_path):
                    continue
                annotations.append(prepare_single_annotation(video_path, video_path, non_collision_obj, "No"))
    return annotations

def prepare_carla_annotations(carla_folder, run_id):
    annotations = []

    carla_scenarios = glob.glob(os.path.join(carla_folder, run_id, "scenario*"), recursive=True)

    for scenario in carla_scenarios:
        scenario_name = os.path.basename(scenario)
        run_name = os.path.basename(os.path.dirname(scenario))

        collision_event_file = os.path.join(scenario, "events_collision.json")
        if not os.path.exists(collision_event_file):
            continue
        with open(collision_event_file, "r") as f:
            collision_event = json.load(f)[0]['objects']

        non_collision_lst = []
        
        non_collision_index = 1
        while non_collision_index:
            non_collision_event_file = os.path.join(scenario, f"events_noncollision_{non_collision_index}.json")
            if not os.path.exists(non_collision_event_file):
                break

            with open(non_collision_event_file, "r") as f:
                non_collision_events = json.load(f)
                non_collision_lst.append(non_collision_events[0]['objects'])
            non_collision_index += 1
        
        carla_scenario = os.path.join(carla_folder, run_name, scenario_name)
        
        collision_video_path = os.path.join(carla_scenario, "events_collision_rgb_som.mp4")
        annotations.append(prepare_single_annotation(collision_video_path, collision_video_path, collision_event, "Yes"))

        for i, non_collision_obj in enumerate(non_collision_lst):
            video_path = os.path.join(carla_scenario, f"events_noncollision_{i+1}_rgb_som.mp4")
            if not os.path.exists(video_path):
                continue
            annotations.append(prepare_single_annotation(video_path, video_path, non_collision_obj, "No"))
    return annotations
        
def balance_yesno(data_path, output_folder, yes_ratio=0.5):
    with open(data_path, "r") as f:
        data = json.load(f)

    yes_data = [item for item in data if item["conversations"][1]["value"] == "Yes"]
    no_data = [item for item in data if item["conversations"][1]["value"] == "No"]

    len_yes = len(yes_data)
    len_no = len(no_data)
    
    expected_no_len = len_yes / yes_ratio * (1 - yes_ratio)
    no_ratio = expected_no_len / len_no
    if no_ratio > 1:
        print(f"no_ratio is greater than 1, {no_ratio}, keeping all no data")
        balanced_data = data
    else:
        no_balanced_data = [item for item in no_data if random.random() < no_ratio]
        balanced_data = yes_data + no_balanced_data
        print(f"original {data_path} has {len_yes} yes and {len_no} no, balanced to {len_yes} yes and {len(no_balanced_data)} no")

    random.shuffle(balanced_data)
    with open(os.path.join(output_folder, "carla_cosmos_annotations_balanced.json"), "w") as f:
        json.dump(balanced_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla_folder", type=str, default="CARLA")
    parser.add_argument("--cosmos_folder", type=str, default="Cosmos")
    parser.add_argument("--output_folder", type=str, default="annotations")
    parser.add_argument("--run_id", type=str, default="default_run")
    parser.add_argument("--yes_ratio", type=float, default=0.5)
    args = parser.parse_args()

    carla_annotations = prepare_carla_annotations(args.carla_folder, args.run_id)
    cosmos_annotations = prepare_cosmos_annotations(args.carla_folder, args.cosmos_folder, args.run_id)

    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "carla_annotations.json"), "w") as f:
        json.dump(carla_annotations, f, indent=4)
    with open(os.path.join(args.output_folder, "cosmos_annotations.json"), "w") as f:
        json.dump(cosmos_annotations, f, indent=4)
    
    carla_cosmos_annotations = carla_annotations + cosmos_annotations
    with open(os.path.join(args.output_folder, "carla_cosmos_annotations.json"), "w") as f:
        json.dump(carla_cosmos_annotations, f, indent=4)
        
    if args.yes_ratio is not None:
        balance_yesno(os.path.join(args.output_folder, "carla_cosmos_annotations.json"), args.output_folder, args.yes_ratio)