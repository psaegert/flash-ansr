import os
from typing import Any
import random

from flash_ansr import ExpressionSpace
from flash_ansr.utils import load_config


class FlashASNRPreprocessor:
    def __init__(self, expression_space: ExpressionSpace, format_probs: dict | None = None) -> None:
        self.expression_space = expression_space

        # By default, do not change the input
        self.format_probs = format_probs or {'complexity': 0}

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "FlashASNRPreprocessor":
        config_ = load_config(config)

        if "preprocessor" in config_.keys():
            config_ = config_["preprocessor"]

        # If the config is a string, convert relative paths within the config to absolute paths
        if isinstance(config, str) and isinstance(config_["expression_space"], str):
            if config_["expression_space"].startswith('.'):
                config_["expression_space"] = os.path.join(os.path.dirname(config), config_["expression_space"])

        return cls(
            ExpressionSpace.from_config(config_["expression_space"]),
            config_.get("format_probs", None)
        )

    def format(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch['complexities'] = []
        batch['input_num'] = []
        for i, input_ids in enumerate(batch['input_ids']):
            complexity = None
            modified_input_ids = batch['input_ids'][i]
            input_num = []

            if random.random() < self.format_probs['complexity']:
                complexity = len(input_ids)
                modified_input_ids = [self.expression_space.tokenizer['<ctrl_complexity>']] + [self.expression_space.tokenizer['<num>']] + input_ids
                input_num.append((1, complexity))

            batch['complexities'].append(complexity)
            batch['input_ids'][i] = modified_input_ids
            batch['input_num'].append(input_num)

        return batch
