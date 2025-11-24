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

"""
Prompt Generator

This module provides the PromptGenerator class for generating text prompts with
randomized variables from predefined sets of options.
"""
import itertools
import random
import logging
from typing import Dict, List


class PromptGenerator:
    """A class for generating text prompts with randomized variables.

    This class allows users to create text templates where specific variables
    can be randomized from predefined sets of options.

    Attributes:
        template (str): The base template string with placeholders for variables.
        attributes (dict): Mapping of placeholder names to lists of possible values.
    """

    def __init__(
        self, template: str, attributes: Dict[str, List[str]] = None, seed: int = None
    ):
        """Initialize the PromptGenerator with a template string.

        Args:
            template (str): A string containing placeholders in curly braces
                (e.g., "The {color} car is {size}").
            attributes (Dict[str, List[str]], optional): Dictionary mapping variable names to lists of possible values.
                Defaults to None.
            seed (int, optional): Random seed for reproducible generation. Defaults to None.
        """
        self._random = random.Random(seed)
        self.template = template
        self.logger = logging.getLogger(__name__)
        if attributes is None:
            self.attributes = {}
        else:
            self.attributes = attributes

    def set_seed(self, seed: int = None) -> None:
        """Set the random seed for reproducible generation.

        Args:
            seed (int | None, optional): The seed value for the random number generator.
                Defaults to None.
        """
        self._random = random.Random(seed)

    def set_template(self, template: str) -> None:
        """Set the template for the prompt generator.

        Args:
            template (str): The template string to set.
        """
        self.template = template

    def add_attribute(self, placeholder: str, attribute_values: List[str]) -> None:
        """Add a pool of possible values for a specific placeholder.

        Args:
            placeholder (str): The name of the placeholder in the template.
            attribute_values (List[str]): List of possible values for this placeholder.
        """
        self.attributes[placeholder] = attribute_values

    def generate(self) -> str:
        """Generate a prompt by randomly selecting values from the pools.

        Returns:
            str: The generated prompt with randomly selected values.

        Raises:
            KeyError: If the template contains a placeholder without a corresponding
                value pool.
        """
        selections = {}
        try:
            for placeholder in self.attributes:
                if self.attributes.get(placeholder):
                    selections[placeholder] = self._random.choice(
                        self.attributes[placeholder]
                    )
                else:
                    selections[placeholder] = ""
            self.logger.debug(selections)
            return self.template.format(**selections), selections
        except KeyError as e:
            self.logger.error(f"KeyError: {e}")
            return self.template, {}

    def all_combinations(self):
        """Returns an iterator that yields all possible combinations of attributes.

        This method generates every possible combination of the attributes defined
        in self.attributes, returning them as dictionaries that can be used with
        the template.

        Returns:
            Iterator[dict]: An iterator yielding dictionaries where each key is a
                placeholder and each value is one of the possible attributes for
                that placeholder.

        Example:
            >>> generator = PromptGenerator("The {color} {animal}")
            >>> generator.add_attribute("color", ["red", "blue"])
            >>> generator.add_attribute("animal", ["cat", "dog"])
            >>> for combo in generator.all_combinations():
            ...     print(generator.template.format(**combo))
            The red cat
            The red dog
            The blue cat
            The blue dog
        """
        # Get all attribute names and their possible values
        placeholders = list(self.attributes.keys())
        value_lists = [
            self.attributes[p] if self.attributes.get(p) else [""] for p in placeholders
        ]

        # Generate all combinations
        for values in itertools.product(*value_lists):
            yield dict(zip(placeholders, values))

    def get_combinations_count(self) -> int:
        """Get the total number of possible combinations.

        Returns:
            int: The total number of possible combinations.
        """
        if not self.attributes:
            return 1

        count = 1
        for values in self.attributes.values():
            count *= len(values) if values else 1
        return count
