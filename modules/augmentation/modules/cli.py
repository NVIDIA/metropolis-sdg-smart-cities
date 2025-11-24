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

import argparse
import os
import logging
import yaml
import json
import time

import omni.client

from scene_captioning.vlm import VLMSceneCaptioning
from template_generation.llm import LLMTemplateGenerator
from prompt_generation.core import PromptGenerator
from cosmos_execution.gradio import GradioCosmosExecutor
from utils.common import replace_words, validate_config_structure, validate_sample_data_availability
from utils.prompt_loader import load_system_prompt_from_config

omni.client.initialize()


def configure_s3(logger: logging.Logger):
    """
    Configure S3 access for omni.client using environment variables.
    """
    url = os.getenv("AWS_ENDPOINT_URL")
    bucket = os.getenv("AWS_S3_BUCKET")
    region = os.getenv("AWS_DEFAULT_REGION")
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not all([url, bucket, region, access_key_id, secret_access_key]):
        logger.debug("Skipping S3 configuration as some environment variables are missing.")
        return False
    
    try:
        result = omni.client.set_s3_configuration(
            url=url,
            bucket=bucket,
            region=region,
            accessKeyId=access_key_id,
            secretAccessKey=secret_access_key
        )
        logger.info(f"Configuration set successfully: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to configure S3: {e}")
        return False


def validate_environment(logger: logging.Logger):
    """
    Validate and warn about missing environment variables based on examples/.env.
    """
    # Required environment variables
    required_env_vars = [
        # VLM/LLM/Cosmos
        "VLM_ENDPOINT_URL",
        "VLM_ENDPOINT_MODEL",
        "LLM_ENDPOINT_URL",
        "LLM_ENDPOINT_MODEL",
        "COSMOS_ENDPOINT_URL",

        # Logging
        "LOG_LEVEL",

        # API Keys
        "BUILD_NVIDIA_API_KEY",
    ]
    
    # Optional environment variables
    optional_env_vars = [
        # AWS S3 (optional in configure_s3)
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_ENDPOINT_URL",
        "AWS_S3_BUCKET",
        "AWS_S3_ADDRESSING_STYLE",
    ]
    
    for var in required_env_vars:
        if not os.getenv(var):
            logger.warning(f"Environment variable {var} is not set")
    
    for var in optional_env_vars:
        if not os.getenv(var):
            logger.debug(f"Environment variable {var} is not set")

    logger.info("Environment variables validated according to examples /.env")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Initialize logger first
    logger = logging.getLogger(__name__)
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Suppress httpx INFO logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print()
    logger.info("="*80)
    logger.info("Cosmos Augmentation")
    logger.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    # Configure S3 access
    configure_s3(logger)
    
    # Load the config
    result, _, content = omni.client.read_file(args.config)
    if result != omni.client.Result.OK:
        logger.error(f"Failed to read config file {args.config}")
        return
    config = yaml.safe_load(memoryview(content).tobytes())
    validate_environment(logger)
    
    # Validate configuration structure
    if not validate_config_structure(config, logger):
        logger.error("Configuration validation failed. Please fix the configuration file.")
        return
    
    # Get config directory for resolving relative prompt file paths
    config_dir = os.path.dirname(args.config)
    if "video_captioning" in config:
        # Use the new from_config method with type validation
        vlm_scene_captioning = VLMSceneCaptioning.from_config(
            config_params=config["video_captioning"],
            endpoint=os.getenv("VLM_ENDPOINT_URL", config["endpoints"]["vlm"]["url"]),
            model=os.getenv("VLM_ENDPOINT_MODEL", config["endpoints"]["vlm"]["model"]),
            logger=logger,
        )
    if "template_generation" in config:
        # Load system prompt from file or use direct text
        system_prompt = load_system_prompt_from_config(config["template_generation"], config_dir, logger)
        
        # Use the new from_config method with type validation
        llm_template_generator = LLMTemplateGenerator.from_config(
            config_params=config["template_generation"],
            system_prompt=system_prompt,
            endpoint=os.getenv("LLM_ENDPOINT_URL", config["endpoints"]["llm"]["url"]),
            model=os.getenv("LLM_ENDPOINT_MODEL", config["endpoints"]["llm"]["model"]),
            logger=logger,
        )

    if "cosmos" in config:
        # Get model version from config (defaults to "ct1" if not specified)
        model_version = config["cosmos"].get("model_version", None)

        # Set up random seed for cosmos augmentation
        if config["cosmos"]["parameters"]["seed"] is None or config["cosmos"]["parameters"]["seed"] == "None":
            config["cosmos"]["parameters"]["seed"] = int(time.time())
            logger.debug(f"No seed provided; using current time as seed for cosmos augmentation: {config['cosmos']['parameters']['seed']}")
        else:
            logger.debug(f"Using seed {config['cosmos']['parameters']['seed']} for cosmos augmentation")
        
        cosmos_executor = GradioCosmosExecutor(
            endpoint=os.getenv(
                "COSMOS_ENDPOINT_URL", config["endpoints"]["cosmos"]["url"]
            ),
            sigma=config["cosmos"]["parameters"]["sigma"],
            seed=config["cosmos"]["parameters"]["seed"],
            guidance=config["cosmos"]["parameters"]["guidance"],
            num_steps=config["cosmos"]["parameters"]["num_steps"],
            modalities=config["cosmos"]["parameters"]["modalities"],
            weights=config["cosmos"]["parameters"]["weights"],
            positive_prompt=config["cosmos"]["parameters"]["positive_prompt"],
            negative_prompt=config["cosmos"]["parameters"]["negative_prompt"],
            inference_name=config["cosmos"]["parameters"]["inference_name"],
            logger=logger,
            model_version=model_version,  # None defaults to "ct1", "ct25" converts to "ct2.5"
        )

    overwrite_caption = os.getenv("OVERWRITE_CAPTION", "true")
    
    for sample in config["data"]:

        logger.info("="*80)
        logger.info(f"Processing sample: {sample['inputs']['rgb']}")
        try:
            # Validate sample data availability
            if not validate_sample_data_availability(sample, config, logger):
                logger.error(f"Sample validation failed, skipping sample: {sample}")
                continue
            
            # Initialize variables that might be used later
            caption = None
            template = None
            template_description = None
            prompt = None
            control_videos = {}
            
            # Check if prompt is already present and skip
            try:
                result, _, content = omni.client.read_file(sample["output"]["caption"])
                logger.debug(f"Caption file read result: {result}")
            except Exception as e:
                logger.error(f"Failed to read caption file {sample['output']['caption']}: {e}")
                continue
            

            if result != omni.client.Result.OK or overwrite_caption == "true":
                logger.debug("Caption not present, proceeding")

                # Run video captioning
                if "video_captioning" in config:
                    logger.info("Running video captioning...")
                    try:
                        caption = vlm_scene_captioning.get_video_caption(
                            sample["inputs"]["rgb"]
                        )
                        logger.debug(f"Caption: {caption}")
                        logger.info("Video captioning completed successfully")
                    except Exception as e:
                        logger.error(f"Failed to generate video caption for {sample['inputs']['rgb']}: {e}")
                        continue

                # Run template generation
                if "template_generation" in config:
                    logger.info("Running template generation and prompt polishing...")
                    try:
                        if caption is None:
                            logger.error("Caption is required for template generation but was not generated")
                            continue
                            
                        template = llm_template_generator.generate_template_singleview(
                            caption, list(config["template_generation"]["variables"].keys())
                        )
                        template_description = replace_words(caption, template)
                    except Exception as e:
                        logger.error(f"Failed to generate template: {e}")
                        continue

                # Run prompt generation
                if "prompt_generation" in config:
                    try:
                        if template_description is None:
                            logger.error("Template description is required for prompt generation but was not generated")
                            continue

                        if config["prompt_generation"]["seed"] is None or config["prompt_generation"]["seed"] == "None":
                            config["prompt_generation"]["seed"] = int(time.time())
                            logger.debug(f"No seed provided; using current time {config['prompt_generation']['seed']} as seed for prompt generation")
                        else:
                            logger.debug(f"Using seed {config['prompt_generation']['seed']} for prompt generation")
                        
                        prompt_generator = PromptGenerator(
                            template=template_description,
                            attributes=config["template_generation"]["variables"],
                            seed=config["prompt_generation"]["seed"],
                        )
                        prompt, selections = prompt_generator.generate()
                        
                        # Polish the prompt with LLM to make it natural
                        if "template_generation" in config:
                            try:
                                polished_prompt = llm_template_generator.polish_prompt(prompt)
                                logger.debug(f"Polished prompts: {polished_prompt}")
                                prompt = polished_prompt
                                logger.info("Prompt polishing completed successfully")
                            except Exception as e:
                                logger.error(f"Failed to polish prompt, using raw prompt: {e}")
                                # Continue with unpolished prompt

                        # Write the prompt to file
                        try:
                            result = omni.client.write_file(
                                sample["output"]["caption"], prompt.encode("utf-8")
                            )
                            if result != omni.client.Result.OK:
                                logger.error(f"Failed to write templated prompt to {sample['output']['caption']}")
                                continue
                            logger.info(f"Prompt written successfully to {sample['output']['caption']}")
                        except Exception as e:
                            logger.error(f"Failed to write prompt file {sample['output']['caption']}: {e}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Failed during prompt generation: {e}")
                        continue
            else:
                try:
                    logger.debug("Prompt already present, skipping")
                    prompt = memoryview(content).tobytes().decode("utf-8")
                    logger.debug(f"Prompt: {prompt}")
                    logger.debug("Using existing prompt from file")
                except Exception as e:
                    logger.error(f"Failed to decode existing prompt: {e}")
                    continue

            # Run cosmos #TODO move this outside as well to improve parallelism on an orchestrator level
            if "cosmos" in config:
                print()
                cosmos_execution_start_time = time.time()
                logger.info("Running cosmos execution, please expect a long wait...")
                try:
                    if prompt is None:
                        logger.error("Prompt is required for cosmos execution but was not generated or loaded")
                        continue
                    
                    # Load control videos for modalities specified in config
                    control_videos = {}
                    try:
                        if "parameters" in config["cosmos"] and "modalities" in config["cosmos"]["parameters"]:
                            for modality in config["cosmos"]["parameters"]["modalities"]:
                                if modality in sample["inputs"]["controls"]:
                                    control_videos[modality] = sample["inputs"]["controls"][modality]
                            logger.debug(f"Loaded control videos: {list(control_videos.keys())}")
                    except Exception as e:
                        logger.error(f"Failed to load control videos for sample {sample}: {e}")
                        continue
                        
                    logger.debug(f"Prompts used for cosmos execution: {prompt}")
                    logger.debug(f"Control videos: {control_videos}")
                    logger.debug(f"Output video: {sample['output']['video']}")
                    
                    success, output_path = cosmos_executor.execute(
                        prompt,
                        sample["inputs"]["rgb"],
                        control_videos,
                        sample["output"]["video"],
                    )
                    if not success:
                        logger.error("Failed to execute cosmos")
                        continue
                    sample["output"]["video"] = output_path
                    elapsed_minutes = (time.time() - cosmos_execution_start_time) / 60
                    logger.info(f"Cosmos execution completed successfully, time taken: {elapsed_minutes:.2f} minutes, saved video to: {output_path}")
                except Exception as e:
                    logger.error(f"Failed during cosmos execution: {e}")
                    continue
            else:
                logger.warning("No cosmos config found, skipping cosmos")

            # Write metadata
            try:
                metadata = {
                    "prompt": prompt,
                    # "selections": selections,
                    # "caption": caption,
                    "caption_path": sample["output"]["caption"],
                    "output_video_path": sample["output"]["video"],
                    "original_video_path": sample["inputs"]["rgb"],
                    "control_videos": control_videos,
                }
                result = omni.client.write_file(
                    sample["output"]["metadata"], json.dumps(metadata).encode("utf-8")
                )
                if result != omni.client.Result.OK:
                    logger.error(f"Failed to write metadata to {sample['output']['metadata']}")
                    continue
                logger.info(f"Metadata written successfully to {sample['output']['metadata']}")
            except Exception as e:
                logger.error(f"Failed to write metadata file {sample['output']['metadata']}: {e}")
                continue
                
            logger.debug(f"Sample processing completed successfully: {sample}")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Unexpected error processing sample {sample}: {e}")
            continue


if __name__ == "__main__":
    main()
