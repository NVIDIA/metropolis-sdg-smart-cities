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

import logging

import tempfile
import os
import re
import base64

from openai import OpenAI

import omni.client
from utils.common import validate_and_cast_config_params

omni.client.initialize()

# Add the parent directory to the path for any potential imports
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Discarding the old base class for now to be more easier to integrate with storage services and the e2e
# Dropping support for generate camera descriptions to be agnostic to simulation backend
class VLMSceneCaptioning:
    # Define expected parameter types for validation
    _PARAM_TYPES = {
        'retry': int,
        'temperature': float,
        'top_p': float,
        'frequency_penalty': float,
        'max_tokens': int,
        'stream': bool,
        'system_prompt': str,
        'user_prompt': str,
        'endpoint': str,
        'model': str
    }
    
    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        retry: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        max_tokens: int,
        stream: bool,
        endpoint: str,
        model: str,
        logger: logging.Logger,
    ):
        # Validate and cast parameters to ensure correct types
        params = {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'retry': retry,
            'temperature': temperature,
            'top_p': top_p,
            'frequency_penalty': frequency_penalty,
            'max_tokens': max_tokens,
            'stream': stream,
            'endpoint': endpoint,
            'model': model
        }
        
        validated_params = validate_and_cast_config_params(params, self._PARAM_TYPES, logger)
        
        self.system_prompt = validated_params['system_prompt']
        self.user_prompt = validated_params['user_prompt']

        self.retry = validated_params['retry']
        self.temperature = validated_params['temperature']
        self.top_p = validated_params['top_p']
        self.frequency_penalty = validated_params['frequency_penalty']
        self.max_tokens = validated_params['max_tokens']
        self.stream = validated_params['stream']

        # Ensure endpoint URL is properly formatted (remove trailing slash to avoid double slashes)
        self.endpoint = validated_params['endpoint'].rstrip('/')
        self.model = validated_params['model']

        self.logger = logger
    
    
    @classmethod
    def from_config(cls, config_params: dict, endpoint: str, model: str, logger: logging.Logger):
        """
        Create VLMSceneCaptioning instance from configuration dictionary with type validation.
        
        Args:
            config_params: Dictionary containing VLM configuration parameters
            endpoint: VLM endpoint URL
            model: VLM model name
            logger: Logger instance
            
        Returns:
            VLMSceneCaptioning: Configured instance with validated parameters
        """
        # Extract and validate parameters
        params = {
            'system_prompt': config_params.get('system_prompt', ''),
            'user_prompt': config_params.get('user_prompt', ''),
            'retry': config_params.get('parameters', {}).get('retry', 0),
            'temperature': config_params.get('parameters', {}).get('temperature', 0.5),
            'top_p': config_params.get('parameters', {}).get('top_p', 1.0),
            'frequency_penalty': config_params.get('parameters', {}).get('frequency_penalty', 0.0),
            'max_tokens': config_params.get('parameters', {}).get('max_tokens', 1000),
            'stream': config_params.get('parameters', {}).get('stream', False),
            'endpoint': endpoint,
            'model': model
        }
        
        validated_params = validate_and_cast_config_params(params, cls._PARAM_TYPES, logger)
        
        return cls(
            system_prompt=validated_params['system_prompt'],
            user_prompt=validated_params['user_prompt'],
            retry=validated_params['retry'],
            temperature=validated_params['temperature'],
            top_p=validated_params['top_p'],
            frequency_penalty=validated_params['frequency_penalty'],
            max_tokens=validated_params['max_tokens'],
            stream=validated_params['stream'],
            endpoint=validated_params['endpoint'],
            model=validated_params['model'],
            logger=logger,
        )

    def get_video_caption(self, video_path: str) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            result, _, content = omni.client.read_file(video_path)
            if result != omni.client.Result.OK:
                raise Exception(f"Failed to read file {video_path}")
            with open(os.path.join(temp_dir, "video.mp4"), "wb") as f:
                f.write(memoryview(content).tobytes())
            logging.info(f"Uploaded video to {os.path.join(temp_dir, 'video.mp4')}")

            try:
                client = OpenAI(
                    base_url=self.endpoint, api_key="not-used", timeout=7200
                )
                
                # Encode video as base64 for NIM VLM API
                with open(os.path.join(temp_dir, "video.mp4"), "rb") as video_file:
                    video_b64 = base64.b64encode(video_file.read()).decode()
                
                conversation = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.user_prompt},
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{video_b64}"
                                }
                            },
                        ],
                    },
                ]
                
                logging.info("Sending chat completion request with video")
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=conversation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    max_tokens=self.max_tokens,
                    stream=self.stream,
                    extra_body={
                        # Optimized video processing parameters for CUDA compatibility
                        "media_io_kwargs": {
                            "video": {"fps": 4.0}  # Default frame rate
                        },
                        "mm_processor_kwargs": {
                            "videos_kwargs": {
                                "min_pixels": 1568,
                                "max_pixels": 32768  # Reduced from default 262144 to avoid GPU memory issues
                            }
                        }
                    },
                )
                assistant_message = chat_response.choices[0].message
                logging.info(assistant_message.content)

                match = re.search(
                    r"<answer>(.*?)</answer>", assistant_message.content, re.DOTALL
                )
                if match:
                    caption = match.group(1).strip()
                    caption = caption.replace("*", "")
                    caption = caption.replace("#", "")
                    caption = caption.replace("  -", "")
                    caption = caption.replace("- ", "")
                    # Remove numbered list items (e.g., "1. ", "2. ", "10. ", etc.)
                    caption = re.sub(r"^\d+\.\s*", "", caption, flags=re.MULTILINE)
                    caption = caption.replace("  ", " ")  # Remove double spaces
                    caption = caption.replace(
                        "\n\n", "\n"
                    )  # Remove extra line breaks
                    caption = caption.strip()  # Remove leading/trailing whitespace
                    logging.info(f"Generated caption: {caption}")
                    return caption
                else:
                    logging.error(
                        f"No caption found in the response: {assistant_message.content}"
                    )
                    raise Exception("No caption found in the response")

            except Exception as e:
                self.logger.error(f"Failed to get video caption: {e}", exc_info=True)
                if self.retry > 0:
                    self.logger.info(f"Retrying {self.retry - 1} times")
                    return self._get_video_caption(video_path, self.retry - 1)
                else:
                    raise e
