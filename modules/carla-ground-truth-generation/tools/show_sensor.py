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
import pygame

def game_loop(args):
    """
    Main game loop for the CARLA camera viewer.

    Args:
        args: An object containing the script's arguments.
    """
    pygame.init()
    actor_list = []

    try:
        # Connect to the CARLA server
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()

        # Get the blueprint for the RGB camera
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.find('sensor.camera.rgb')

        # Set camera attributes
        bp.set_attribute('image_size_x', f'{args.width}')
        bp.set_attribute('image_size_y', f'{args.height}')
        bp.set_attribute('fov', str(args.fov))

        # Define the camera's location and rotation from user arguments
        transform = carla.Transform(
            carla.Location(x=args.x/100.0, y=args.y/100.0, z=args.z/100.0),
            carla.Rotation(pitch=args.pitch, yaw=args.yaw, roll=args.roll)
        )

        # Spawn the camera
        camera = world.spawn_actor(bp, transform)
        actor_list.append(camera)
        print(f'Created camera "{camera.type_id}" with id {camera.id}')

        # Set up the Pygame display
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Camera Feed")

        # This function will be called each time the camera produces a new image
        def process_image(image):
            """Processes the camera image and displays it on the Pygame screen."""
            raw_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            raw_image = np.reshape(raw_image, (image.height, image.width, 4))
            # The image from CARLA is in BGRA format, so we need to swizzle the channels for Pygame
            bgr_image = raw_image[:, :, :3]
            # Convert to RGB for display
            rgb_image = bgr_image[:, :, ::-1]
            surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))
            display.blit(surface, (0, 0))

        # Start listening for image data
        camera.listen(lambda image: process_image(image))

        # Main loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip()

    finally:
        print('Cleaning up actors...')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('Done.')

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
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--fov',
        default=110.0,
        type=float,
        help='Camera field of view in degrees (default: 110.0)')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='X position of the camera')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='Y position of the camera')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='Z position of the camera')
    argparser.add_argument(
        '--pitch',
        default=0.0,
        type=float,
        help='Pitch rotation of the camera')
    argparser.add_argument(
        '--yaw',
        default=0.0,
        type=float,
        help='Yaw rotation of the camera')
    argparser.add_argument(
        '--roll',
        default=0.0,
        type=float,
        help='Roll rotation of the camera')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()
