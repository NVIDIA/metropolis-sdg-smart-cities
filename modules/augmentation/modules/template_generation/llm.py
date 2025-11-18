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
import os
import re
from typing import Dict, List, Any, Tuple
from openai import OpenAI
import logging
from utils.common import validate_and_cast_config_params


class LLMTemplateGenerator:
    # Define expected parameter types for validation
    _PARAM_TYPES = {
        'system_prompt': str,
        'retry': int,
        'retry_policy': str,
        'temperature': float,
        'top_p': float,
        'frequency_penalty': float,
        'presence_penalty': float,
        'max_tokens': int,
        'stream': bool,
        'endpoint': str,
        'model': str,
        'polish_temperature': float
    }
    
    def __init__(
        self,
        system_prompt: str,
        retry: int,
        retry_policy: str,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        max_tokens: int,
        stream: bool,
        endpoint: str,
        model: str,
        logger: logging.Logger,
        polish_temperature: float = 0.8,
    ):
        # Validate and cast parameters to ensure correct types
        params = {
            'system_prompt': system_prompt,
            'retry': retry,
            'retry_policy': retry_policy,
            'temperature': temperature,
            'top_p': top_p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
            'max_tokens': max_tokens,
            'stream': stream,
            'endpoint': endpoint,
            'model': model,
            'polish_temperature': polish_temperature
        }
        
        validated_params = validate_and_cast_config_params(params, self._PARAM_TYPES, logger)
        
        self.system_prompt = validated_params['system_prompt']

        self.retry = validated_params['retry']
        self.retry_policy = validated_params['retry_policy']
        self.temperature = validated_params['temperature']
        self.top_p = validated_params['top_p']
        self.frequency_penalty = validated_params['frequency_penalty']
        self.presence_penalty = validated_params['presence_penalty']
        self.max_tokens = validated_params['max_tokens']
        self.stream = validated_params['stream']
        self.endpoint = validated_params['endpoint']
        self.model = validated_params['model']
        self.polish_temperature = validated_params['polish_temperature']

        if os.environ.get("BUILD_NVIDIA_API_KEY"):
            api_key = os.environ.get("BUILD_NVIDIA_API_KEY")
        else:
            logger.error("BUILD_NVIDIA_API_KEY is not set")
            raise ValueError("BUILD_NVIDIA_API_KEY is not set")

        self.client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )

        self.logger = logger
    
    @classmethod
    def from_config(cls, config_params: dict, system_prompt: str, endpoint: str, model: str, logger: logging.Logger):
        """
        Create LLMTemplateGenerator instance from configuration dictionary with type validation.
        
        Args:
            config_params: Dictionary containing LLM configuration parameters
            system_prompt: System prompt text
            endpoint: LLM endpoint URL
            model: LLM model name
            logger: Logger instance
            
        Returns:
            LLMTemplateGenerator: Configured instance with validated parameters
        """
        # Extract and validate parameters
        params = {
            'system_prompt': system_prompt,
            'retry': config_params.get('parameters', {}).get('retry', 1),
            'retry_policy': config_params.get('parameters', {}).get('retry_policy', 'default'),
            'temperature': config_params.get('parameters', {}).get('temperature', 0.0),
            'top_p': config_params.get('parameters', {}).get('top_p', 0.95),
            'frequency_penalty': config_params.get('parameters', {}).get('frequency_penalty', 0.0),
            'presence_penalty': config_params.get('parameters', {}).get('presence_penalty', 0.0),
            'max_tokens': config_params.get('parameters', {}).get('max_tokens', 4096),
            'stream': config_params.get('parameters', {}).get('stream', True),
            'polish_temperature': config_params.get('parameters', {}).get('polish_temperature', 0.8),
            'endpoint': endpoint,
            'model': model
        }
        
        validated_params = validate_and_cast_config_params(params, cls._PARAM_TYPES, logger)
        
        return cls(
            system_prompt=validated_params['system_prompt'],
            retry=validated_params['retry'],
            retry_policy=validated_params['retry_policy'],
            temperature=validated_params['temperature'],
            top_p=validated_params['top_p'],
            frequency_penalty=validated_params['frequency_penalty'],
            presence_penalty=validated_params['presence_penalty'],
            max_tokens=validated_params['max_tokens'],
            stream=validated_params['stream'],
            endpoint=validated_params['endpoint'],
            model=validated_params['model'],
            logger=logger,
            polish_temperature=validated_params['polish_temperature'],
        )

    def _sanitize_model_output(self, text: str) -> str:
        """
        Remove artifacts such as Markdown code fences and JS-style comments
        that may appear in model output and break strict JSON parsing.

        The sanitizer preserves newlines while stripping:
        - Markdown code fences like ```json and closing ```
        - Block comments /* ... */
        - Single-line comments starting with // or # (outside of string literals)
        """
        # Remove Markdown code fences
        sanitized = re.sub(r"```[a-zA-Z]*", "", text)
        sanitized = re.sub(r"```", "", sanitized)

        # Remove block comments
        sanitized = re.sub(r"/\*.*?\*/", "", sanitized, flags=re.DOTALL)

        # Remove single-line // comments while respecting quotes
        def strip_line_comment(line: str) -> str:
            in_single_quote = False
            in_double_quote = False
            i = 0
            out_chars: List[str] = []
            while i < len(line):
                ch = line[i]

                # Detect start of // comment only when not inside quotes
                if not in_single_quote and not in_double_quote:
                    if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                        break
                    if ch == "#":
                        break

                # Handle escapes within quotes
                if ch == "\\" and (in_single_quote or in_double_quote):
                    out_chars.append(ch)
                    i += 1
                    if i < len(line):
                        out_chars.append(line[i])
                    i += 1
                    continue

                if ch == "'" and not in_double_quote:
                    in_single_quote = not in_single_quote
                    out_chars.append(ch)
                    i += 1
                    continue

                if ch == '"' and not in_single_quote:
                    in_double_quote = not in_double_quote
                    out_chars.append(ch)
                    i += 1
                    continue

                out_chars.append(ch)
                i += 1

            return "".join(out_chars)

        sanitized = "\n".join(
            strip_line_comment(line) for line in sanitized.splitlines()
        )
        return sanitized

    def _parse_llm_response(self, result: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract the JSON template.

        Args:
            result (str): The raw LLM response

        Returns:
            List[Dict[str, Any]]: The parsed template

        Raises:
            ValueError: If the response cannot be parsed as JSON
        """
        # Step 1: cut away any leading chain-of-thought like </think> blocks
        think_close_idx = result.rfind("</think>")
        base_text = (
            result[think_close_idx + len("</think>") :].strip()
            if think_close_idx != -1
            else result.strip()
        )
        self.logger.debug(
            "Post-think text length=%d (think_found=%s)",
            len(base_text),
            think_close_idx != -1,
        )

        # Step 2: prefer content inside the first fenced code block, if present
        fence_match = re.search(
            r"```(?:json|javascript|js|python)?\s*([\s\S]*?)\s*```",
            base_text,
            flags=re.IGNORECASE,
        )
        candidate_text = fence_match.group(1).strip() if fence_match else base_text
        self.logger.debug(
            "Fenced block %s; candidate length=%d",
            "found" if fence_match else "not found",
            len(candidate_text),
        )

        # Step 3: sanitize comments and stray fences but DO NOT collapse whitespace
        sanitized = self._sanitize_model_output(candidate_text)
        self.logger.debug("Sanitized candidate length=%d", len(sanitized))

        # Step 4: attempt to decode the first JSON value (streaming decoder)
        def _decode_first_json_value(text: str) -> Tuple[Any, int, int]:
            decoder = json.JSONDecoder()
            # Scan for likely JSON starts and attempt raw_decode from there
            for match in re.finditer(r"[\[{]", text):
                start_idx = match.start()
                try:
                    obj, end_idx = decoder.raw_decode(text, idx=start_idx)
                    return obj, start_idx, end_idx
                except json.JSONDecodeError:
                    continue
            raise json.JSONDecodeError("No decodable JSON value found", text, 0)

        try:
            obj, start, end = _decode_first_json_value(sanitized)
            snippet_preview = sanitized[
                max(0, start - 60) : min(len(sanitized), end + 60)
            ]
            self.logger.debug(
                "Decoded JSON segment start=%d end=%d preview=%r",
                start,
                end,
                snippet_preview
                if len(snippet_preview) < 500
                else snippet_preview[:500] + "...",
            )
        except json.JSONDecodeError as e:
            # Add rich diagnostics to help debug failures
            self.logger.error(
                "JSON decode failed: msg=%s pos=%s lineno=%s colno=%s; sanitized_head=%r",
                getattr(e, "msg", ""),
                getattr(e, "pos", None),
                getattr(e, "lineno", None),
                getattr(e, "colno", None),
                sanitized[:500],
            )
            error_msg = (
                "Could not parse JSON from model output. "
                "Enable debug logs to inspect sanitized text and decoder position."
            )
            raise ValueError(error_msg) from e

        # Step 5: normalize to expected shape (list of {category, words})
        try:
            normalized = self._normalize_template_output(obj)
        except ValueError as norm_err:
            self.logger.error("Could not normalize model output to expected template format: %s", norm_err)
            raise ValueError("Model output JSON must be a list of objects") from norm_err

        return normalized

    def _normalize_template_output(self, obj: Any) -> List[Dict[str, Any]]:
        """
        Normalize various plausible model output shapes into
        List[Dict[str, Any]] with keys 'category' and 'words'.

        Accepted inputs:
        - List[Dict]: already-correct or partially-correct items
        - Dict with keys 'category' and 'words'
        - Dict wrapper like {'items': [...]} or {'templates': [...]} etc.
        - Dict mapping category -> words (list, string, number, or dict)
        """

        def coerce_words(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(v) for v in value if v is not None]
            if isinstance(value, dict):
                return [str(v) for v in value.values() if v is not None]
            if value is None:
                return []
            return [str(value)]

        def build_item(category: Any, words: Any) -> Dict[str, Any]:
            return {
                "category": str(category) if category is not None else "",
                "words": coerce_words(words),
            }

        # Case 1: already a list
        if isinstance(obj, list):
            normalized_list: List[Dict[str, Any]] = []
            for item in obj:
                if isinstance(item, dict):
                    if "category" in item and "words" in item:
                        normalized_list.append(build_item(item.get("category"), item.get("words")))
                    else:
                        # Best-effort: treat single-key dicts as {category: words}
                        if len(item) == 1:
                            [(cat, wrds)] = item.items()
                            normalized_list.append(build_item(cat, wrds))
            if not normalized_list:
                raise ValueError("List did not contain any valid template items")
            return normalized_list

        # Case 2: single dict item with category/words
        if isinstance(obj, dict):
            if "category" in obj and "words" in obj:
                return [build_item(obj.get("category"), obj.get("words"))]

            # Case 3: wrapper keys that contain a list
            for key in ("templates", "items", "data", "list", "result", "output"):
                inner = obj.get(key)
                if isinstance(inner, list):
                    return self._normalize_template_output(inner)

            # Case 4: mapping category -> words
            if all(isinstance(k, str) for k in obj.keys()):
                if all(isinstance(v, (list, str, int, float, dict, type(None))) for v in obj.values()):
                    return [build_item(cat, wrds) for cat, wrds in obj.items()]

        raise ValueError(f"Unsupported model output type: {type(obj).__name__}")

    def _create_user_prompt(self, description: str, categories: List[str]) -> str:
        """
        Create the user prompt for template generation.

        Args:
            description (str): The text description to process
            categories (List[str]): List of categories to match

        Returns:
            str: The formatted user prompt
        """
        return f"""Remember, I want it as a list of JSON objects with 2 attributes:
                            \t- category: The category to match and replace words with.
                            \t- words: A list of words that should be replaced with the category. \
                                        This should not be a list of more JSON objects.
                        Output requirements:
                        - Output MUST be ONLY a JSON array ([]) of objects with keys 'category' and 'words'.
                        - Do NOT wrap the array in any outer object or additional fields.
                        - Do NOT include markdown code fences or any explanatory text.
                        For example:
                        [
                            \t{{\"category\": \"color\", \
                                \"words\": [\"orange\", \"red\", \"blue\", \"white\", \
                                                \"red-and-white\", \"gray\"]}}
                        ]
                        Categories to match: {", ".join(categories)}
                        Text: {description}"""

    def generate_template_singleview(
        self, description: str, categories: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate a template from a description and categories using LLM.

        Args:
            description (str): The text description to process
            categories (List[str]): List of categories to match and replace

        Returns:
            List[Dict[str, Any]]: List of replacement dictionaries

        Raises:
            RuntimeError: If template generation fails
        """
        self.logger.info(f"Generating template for categories: {categories}")
        self.logger.debug(
            f"Using description: {description[:100]}..."
        )  # Log first 100 chars

        try:
            user_prompt = self._create_user_prompt(description, categories)

            self.logger.debug("Making API call to generate template")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stream=self.stream,
            )

            result = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    result += chunk.choices[0].delta.content

            # Parse the result
            template = self._parse_llm_response(result)

            self.logger.info("Successfully generated and parsed template")
            self.logger.debug(f"Generated template: {template}")
            return template

        except Exception as e:
            error_msg = f"Error in template generation: {e}"
            self.logger.error(error_msg)
            # TODO: Implement retry policy
            # if retry_policy is "check_vars", check if the variables are in the template , so that we generate a template which meets the requirements
            if self.retry > 0:
                self.retry -= 1
                self.logger.info(f"Retrying {self.retry} times")
                return self.generate_template_singleview(description, categories)
            else:
                self.logger.error(
                    f"Failed to generate template after {self.retry} retries"
                )
                self.logger.error(f"Error: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def polish_prompt(self, raw_prompt: str) -> str:
        """
        Polish the raw templated prompt to make it natural and fluent.
        
        Args:
            raw_prompt (str): The raw prompt with potential awkward phrases
        
        Returns:
            str: The polished, natural-sounding prompt
        """
        self.logger.info("Polishing prompt for natural language")
        
        user_prompt = f"""{raw_prompt}"""

        try:
            self.logger.debug("Making API call to polish prompt")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Rewrite text to fix awkward or repetitive phrases while preserving all details. Output ONLY the rewritten text with no preamble or explanation."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.polish_temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stream=self.stream,
            )

            result = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    result += chunk.choices[0].delta.content

            polished = result.strip()
            # Remove any chain-of-thought content and keep only content after the last </think>
            think_close_idx = polished.rfind("</think>")
            if think_close_idx != -1:
                self.logger.info("Stripping chain-of-thought content up to last </think> tag")
                polished = polished[think_close_idx + len("</think>") :].strip()
            
            # Normalize formatting similar to VLM caption cleaning
            polished = polished.replace("*", "")
            polished = polished.replace("#", "")
            polished = polished.replace("  -", "")
            polished = polished.replace("- ", "")
            polished = re.sub(r"^\d+[\.\)]\s*", "", polished, flags=re.MULTILINE)
            polished = polished.replace("  ", " ")  # collapse double spaces
            polished = polished.replace("\n\n", "\n")  # remove extra line breaks
            polished = polished.strip()
            
            return polished

        except Exception as e:
            self.logger.error(f"Error polishing prompt: {e}")
            self.logger.warning("Falling back to raw prompt")
            return raw_prompt