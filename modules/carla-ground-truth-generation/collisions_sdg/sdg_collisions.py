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
import logging
import random
import numpy as np
import re
import json

# Constants
CARLA_CLIENT_TIMEOUT = 20.0  # seconds


def report_collisions(args):
    """
    Extract and process collision events from CARLA recording file.
    
    Args:
        args: Parsed command-line arguments containing server info and filters
    """
    try:
        # Establish connection to CARLA server
        client = carla.Client(args.host, args.port)
        client.set_timeout(CARLA_CLIENT_TIMEOUT)

        # Query collision data from recording file (vehicle-to-vehicle collisions)
        recorder_text = client.show_recorder_collisions(args.recorder_filename, "v", "v")
        if args.debug:
            logging.debug(recorder_text)

        # Parse collision records: timestamp, actor1_id, actor1_type, actor2_id, actor2_type
        pattern = re.compile(
            r'\s*(\d+)\s+v v\s+(\d+)\s+([\w\.\-_]+)\s+(\d+)\s+([\w\.\-_]+)'
        )

        matches = pattern.findall(recorder_text)

        seen_pairs = set()
        data = []
        seq_counter = 1

        for m in matches:
            time = int(m[0])
            id1 = int(m[1])
            id2 = int(m[3])

            # Apply actor ID filtering if specified (requires BOTH actors in filter list)
            if args.ids:
                # Only keep collisions where both participants match the filter
                if id1 not in args.ids or id2 not in args.ids:
                    continue

            # Normalize pair order to prevent duplicate entries (e.g., (A,B) == (B,A))
            pair = tuple(sorted((id1, id2)))

            if pair not in seen_pairs:
                seen_pairs.add(pair)
                seq_str = f"event_{seq_counter:06d}"
                data.append({
                    "event_id": seq_str,
                    "start_time": str(time),
                    "end_time": str(time),
                    "category": "collision",
                    "event_sub_category": "",
                    "objects": [str(id1), str(id2)],
                    "event_caption": ""
                })
                seq_counter += 1

        # Export collision events to JSON file (only if collisions were detected)
        if seq_counter > 1:
            with open("events.json", "w") as f:
                json.dump(data, f, indent=4)
            logging.info("Data written to events.json")
        else:
            logging.info("No collisions were detected")

    finally:
        pass

def main():
    """Main function to parse arguments and start the camera."""
    argparser = argparse.ArgumentParser(
        description='CARLA Camera SDG Client')
    argparser.add_argument(
        '-f','--recorder-filename',
        type=str,
        required=True,
        help='Carla log file')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--id',
        type=int,
        action='append',
        dest='ids',
        help='ID of vehicle to filter collisions (can be specified multiple times)')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=log_level
    )

    logging.info('Connecting to CARLA server %s:%s', args.host, args.port)

    report_collisions(args)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        pass
