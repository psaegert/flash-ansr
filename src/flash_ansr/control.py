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
        if not isinstance(batch['input_ids'][0], list):
            # The input is a single instance
            for k, diff in self._format_single(batch).items():
                batch[k] = diff

            return batch

        else:
            # The input is a batch of instances
            new_fields: set[str] = set()
            for i, instance in enumerate(zip(*batch.values())):
                for k, diff in self._format_single(dict(zip(batch.keys(), instance))).items():
                    if k not in batch:
                        new_fields.add(k)
                        batch[k] = []

                    if k in new_fields:
                        batch[k].append(diff)
                    else:
                        batch[k][i] = diff

        return batch

    def _format_single(self, instance: dict[str, Any]) -> dict[str, Any]:
        complexity = None
        modified_input_ids = instance['input_ids']
        input_num = []

        if random.random() < self.format_probs['complexity']:
            complexity = len(instance['input_ids'])
            modified_input_ids = [self.expression_space.tokenizer['<ctrl_complexity>']] + [self.expression_space.tokenizer['<num>']] + instance['input_ids']
            input_num.append((1, complexity))

        return {
            'complexities': complexity,
            'input_ids': modified_input_ids,
            'input_num': input_num
        }
