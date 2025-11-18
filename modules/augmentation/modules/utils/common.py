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

import io
import sys
import re
import logging
import yaml
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Union, Any


class PrintCaptureHandler(logging.StreamHandler):
    """A custom stream handler that forwards captured output to a logger."""

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def emit(self, record):
        # Forward only if target logger differs from this handler's owner
        if record.name != self.logger.name:
            self.logger.info(self.format(record))
        else:
            super().emit(record)  # fall back to normal stream handling


@contextmanager
def capture_prints(logger):
    """Capture print statements and redirect them to a logger while still printing to terminal.

    Args:
        logger: The logger instance to send captured print statements to.

    Example:
        logger = logging.getLogger(__name__)
        with capture_prints(logger):
            print("This will be both printed and logged")
    """
    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    # Track if we're currently in a logging operation to prevent recursion
    _capturing = False

    class TeeStdout:
        def write(self, data):
            nonlocal _capturing
            new_stdout.write(data)
            # Only log non-empty lines, and prevent recursion
            if data.strip() and not _capturing:
                try:
                    _capturing = True
                    print(f"Capturing print: {data.strip()}")
                    logger.info(data.strip())
                finally:
                    _capturing = False

        def flush(self):
            new_stdout.flush()
            old_stdout.flush()

    sys.stdout = TeeStdout()
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout


def replace_words(
    text: str, replacements: List[Dict[str, Union[str, List[str]]]]
) -> str:
    """
    Replace words in a text string based on a list of replacement dictionaries.
    Processes categories in the order they appear in the list.
    Within each category, longer words are replaced first to handle overlapping cases.
    Case-insensitive matching is used, so "Blue", "BLUE", and "blue" will all match.

    Args:
        text (str): The input text string
        replacements (list[dict]): List of dictionaries, each containing:
            - 'category': str - the new word to replace with
            - 'words': list[str] - list of words to be replaced

    Returns:
        str: Text with words replaced according to the mapping

    Example:
        >>> text = "The RED-and-White bright CAT"
        >>> replacements = [
        ...     {"category": "color", "words": ["red", "white"]},  # Note: lowercase
        ...     {"category": "lighting", "words": ["bright"]}      # Will match any case
        ... ]
        >>> replace_words(text, replacements)
        'The {color}-and-{color} {lighting} CAT'
    """
    # Make a copy of the text
    result = text

    # Process each category in the order they appear in the list
    for replacement in replacements:
        category = replacement["category"]
        # Sort words within this category by length (longest first)
        words = sorted(replacement["words"], key=len, reverse=True)

        # Replace each word in this category
        for old_word in words:
            # Create patterns for different cases
            patterns = [
                # Match word with hyphens
                rf"(-{re.escape(old_word)}-)|\b{re.escape(old_word)}\b",
                # Match at start of hyphenated word
                rf"\b{re.escape(old_word)}-",
                # Match at end of hyphenated word
                rf"-{re.escape(old_word)}\b",
            ]

            # Apply each pattern
            for pattern in patterns:
                result = re.sub(pattern, f"{{{category}}}", result, flags=re.IGNORECASE)

    return result


# YAML Configuration Utilities
def read_yaml_config(yaml_file_path: str) -> Dict[str, Any]:
    """
    Read and parse a YAML configuration file.

    Args:
        yaml_file_path (str): Path to the YAML configuration file

    Returns:
        Dict[str, Any]: Parsed YAML configuration

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    if not Path(yaml_file_path).exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file_path}")

    with open(yaml_file_path, "r") as f:
        return yaml.safe_load(f)


def get_output_dir_from_config(config: Dict[str, Any]) -> str:
    """
    Extract the output_dir parameter from Isaac Sim Replicator configuration.

    Args:
        config (Dict[str, Any]): Parsed YAML configuration

    Returns:
        str: The output directory path

    Raises:
        KeyError: If the output_dir parameter is not found
    """
    try:
        return config["isaacsim.replicator.agent"]["replicator"]["parameters"][
            "output_dir"
        ]
    except KeyError as e:
        raise KeyError(
            f"Could not find 'output_dir' parameter in configuration. Missing key: {e}"
        )


def get_output_dir_from_yaml_file(yaml_file_path: str) -> str:
    """
    Read the output_dir parameter directly from a YAML file.

    Args:
        yaml_file_path (str): Path to the YAML configuration file

    Returns:
        str: The output directory path

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        KeyError: If the output_dir parameter is not found
        yaml.YAMLError: If the YAML file is malformed
    """
    config = read_yaml_config(yaml_file_path)
    return get_output_dir_from_config(config)


def get_replicator_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all replicator parameters from Isaac Sim Replicator configuration.

    Args:
        config (Dict[str, Any]): Parsed YAML configuration

    Returns:
        Dict[str, Any]: All replicator parameters

    Raises:
        KeyError: If the replicator parameters section is not found
    """
    try:
        return config["isaacsim"]["replicator"]["agent"]["replicator"]["parameters"]
    except KeyError as e:
        raise KeyError(
            f"Could not find replicator parameters in configuration. Missing key: {e}"
        )


def validate_output_dir(output_dir: str, create_if_missing: bool = False) -> bool:
    """
    Validate that an output directory exists and optionally create it.

    Args:
        output_dir (str): Path to the output directory
        create_if_missing (bool): Whether to create the directory if it doesn't exist

    Returns:
        bool: True if directory exists or was created successfully, False otherwise
    """
    output_path = Path(output_dir)

    if output_path.exists():
        return True

    if create_if_missing:
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    return False


def cleanup_oauth_token_file(
    token_file_path: str = "augmentation_steps/scene_captioning/py_llm_oauth_token.json",
) -> bool:
    """
    Remove the OAuth token file for security purposes.

    Args:
        token_file_path (str): Path to the OAuth token file to remove

    Returns:
        bool: True if file was removed or didn't exist, False if removal failed
    """
    try:
        token_path = Path(token_file_path)
        if token_path.exists():
            token_path.unlink()
            print(f"Removed OAuth token file: {token_file_path}")
            return True
        else:
            # File doesn't exist, which is fine
            return True
    except Exception as e:
        # Log the error but don't fail the entire process
        print(f"Warning: Could not remove OAuth token file {token_file_path}: {e}")
        return False


def cleanup_sensitive_files(base_dir: str = ".") -> None:
    """
    Clean up sensitive files before execution.

    Args:
        base_dir (str): Base directory to search for sensitive files
    """
    sensitive_files = [
        "augmentation_steps/scene_captioning/py_llm_oauth_token.json",
        "*.key",
        "*.pem",
        "*.p12",
        "*.pfx",
        "credentials.json",
        "token.json",
    ]

    base_path = Path(base_dir)

    for pattern in sensitive_files:
        try:
            if "*" in pattern:
                # Handle glob patterns
                for file_path in base_path.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"Removed sensitive file: {file_path}")
            else:
                # Handle specific file paths
                file_path = base_path / pattern
                if file_path.exists() and file_path.is_file():
                    file_path.unlink()
                    print(f"Removed sensitive file: {file_path}")
        except Exception as e:
            print(f"Warning: Could not remove sensitive file {pattern}: {e}")


# Type Validation and Casting Functions
def validate_and_cast_config_params(params: Dict[str, Any], param_types: Dict[str, type], logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Validate and cast configuration parameters to their expected types.
    
    Args:
        params (Dict[str, Any]): Dictionary of parameters to validate and cast
        param_types (Dict[str, type]): Dictionary mapping parameter names to expected types
        logger (logging.Logger, optional): Logger for error reporting
    
    Returns:
        Dict[str, Any]: Dictionary with parameters cast to correct types
    
    Raises:
        TypeError: If a parameter cannot be cast to the expected type
        ValueError: If a parameter value is invalid for the expected type
    """
    validated_params = {}
    
    for param_name, expected_type in param_types.items():
        if param_name not in params:
            continue  # Skip missing parameters - let the calling code handle defaults
            
        value = params[param_name]
        
        # If already the correct type, use as-is
        if isinstance(value, expected_type):
            validated_params[param_name] = value
            continue
            
        # Attempt type casting
        try:
            if expected_type is bool:
                # Handle boolean conversion specially
                if isinstance(value, str):
                    if value.lower() in ('true', '1', 'yes', 'on'):
                        validated_params[param_name] = True
                    elif value.lower() in ('false', '0', 'no', 'off'):
                        validated_params[param_name] = False
                    else:
                        raise ValueError(f"Cannot convert '{value}' to boolean")
                else:
                    validated_params[param_name] = bool(value)
            elif expected_type in (int, float):
                # Handle numeric conversion
                validated_params[param_name] = expected_type(value)
            elif expected_type is str:
                # Handle string conversion
                validated_params[param_name] = str(value)
            else:
                # For other types, try direct casting
                validated_params[param_name] = expected_type(value)
                
            if logger:
                logger.debug(f"Cast parameter '{param_name}' from {type(value).__name__} to {expected_type.__name__}: {value} -> {validated_params[param_name]}")
                
        except (ValueError, TypeError) as e:
            error_msg = f"Cannot cast parameter '{param_name}' with value '{value}' (type: {type(value).__name__}) to {expected_type.__name__}: {e}"
            if logger:
                logger.error(error_msg)
            raise TypeError(error_msg)
    
    return validated_params


# Configuration Validation Functions
def validate_config_structure(config: dict, logger: logging.Logger) -> bool:
    """Validate the configuration structure and required fields."""
    try:
        # Check required top-level sections
        if "data" not in config:
            logger.error("Missing required 'data' section in config")
            return False
        
        if not isinstance(config["data"], list) or len(config["data"]) == 0:
            logger.error("'data' section must be a non-empty list")
            return False
        
        # Validate data samples
        for i, sample in enumerate(config["data"]):
            if "inputs" not in sample:
                logger.error(f"Sample {i}: Missing 'inputs' section")
                return False
            
            if "output" not in sample:
                logger.error(f"Sample {i}: Missing 'output' section")
                return False
            
            # Check required input fields
            if "rgb" not in sample["inputs"]:
                logger.error(f"Sample {i}: Missing 'inputs.rgb' field")
                return False
            
            # Check required output fields
            required_outputs = ["video", "caption", "metadata"]
            for output_field in required_outputs:
                if output_field not in sample["output"]:
                    logger.error(f"Sample {i}: Missing 'output.{output_field}' field")
                    return False
        
        # Validate endpoints if any modules are configured
        if any(key in config for key in ["video_captioning", "template_generation", "cosmos"]):
            if "endpoints" not in config:
                logger.error("Missing 'endpoints' section but modules are configured")
                return False
        
        # Validate video_captioning config
        if "video_captioning" in config:
            vlm_config = config["video_captioning"]
            required_vlm_fields = ["system_prompt", "user_prompt", "parameters"]
            for field in required_vlm_fields:
                if field not in vlm_config:
                    logger.error(f"video_captioning: Missing required field '{field}'")
                    return False
            
            if "vlm" not in config["endpoints"]:
                logger.error("video_captioning is configured but 'endpoints.vlm' is missing")
                return False
        
        # Validate template_generation config
        if "template_generation" in config:
            template_config = config["template_generation"]
            if "parameters" not in template_config:
                logger.error("template_generation: Missing 'parameters' section")
                return False
            
            if "variables" not in template_config:
                logger.error("template_generation: Missing 'variables' section")
                return False
            
            if "llm" not in config["endpoints"]:
                logger.error("template_generation is configured but 'endpoints.llm' is missing")
                return False
        
        # Validate cosmos config
        if "cosmos" in config:
            cosmos_config = config["cosmos"]
            if "parameters" not in cosmos_config:
                logger.error("cosmos: Missing 'parameters' section")
                return False
            
            cosmos_params = cosmos_config["parameters"]
            required_cosmos_fields = ["sigma", "seed", "guidance", "num_steps"]
            for field in required_cosmos_fields:
                if field not in cosmos_params:
                    logger.error(f"cosmos.parameters: Missing required field '{field}'")
                    return False
            
            # Check if modalities is defined when cosmos is used
            if "modalities" not in cosmos_params:
                logger.error("cosmos.parameters: Missing 'modalities' field. Please define the control modalities to use (e.g., ['edge', 'depth', 'seg'])")
                return False
            
            if not isinstance(cosmos_params["modalities"], list) or len(cosmos_params["modalities"]) == 0:
                logger.error("cosmos.parameters.modalities must be a non-empty list")
                return False
            
            if "weights" not in cosmos_params:
                logger.error("cosmos.parameters: Missing 'weights' field")
                return False
            
            # Validate that weights match modalities
            for modality in cosmos_params["modalities"]:
                if modality not in cosmos_params["weights"]:
                    logger.error(f"cosmos.parameters.weights: Missing weight for modality '{modality}'")
                    return False
            
            # Require positive_prompt and negative_prompt
            if "positive_prompt" not in cosmos_params:
                logger.error("cosmos.parameters: Missing required field 'positive_prompt'")
                return False
            
            if "negative_prompt" not in cosmos_params:
                logger.error("cosmos.parameters: Missing required field 'negative_prompt'")
                return False
            
            if "cosmos" not in config["endpoints"]:
                logger.error("cosmos is configured but 'endpoints.cosmos' is missing")
                return False
        
        # Validate prompt_generation dependencies
        if "prompt_generation" in config:
            if "template_generation" not in config:
                logger.error("prompt_generation requires template_generation to be configured")
                return False
        
        logger.info("Configuration structure validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error during config validation: {e}")
        return False


def validate_sample_data_availability(sample: dict, config: dict, logger: logging.Logger) -> bool:
    """Validate that required data is available for a specific sample."""
    try:
        # Check if cosmos is configured and validate control inputs
        if "cosmos" in config:
            cosmos_params = config["cosmos"]["parameters"]
            if "modalities" in cosmos_params:
                for modality in cosmos_params["modalities"]:
                    if "controls" not in sample["inputs"]:
                        logger.error(f"Sample missing 'controls' section but cosmos requires modality '{modality}'")
                        return False
                    
                    if modality not in sample["inputs"]["controls"]:
                        logger.error(f"Sample missing control input for required modality '{modality}'")
                        return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating sample data: {e}")
        return False
