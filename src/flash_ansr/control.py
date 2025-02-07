from typing import Any
import random

from flash_ansr.models.transformer_utils import Tokenizer


class ControlFormatter:
    def __init__(self, tokenizer: Tokenizer, format_probs: dict | None = None) -> None:
        self.tokenizer = tokenizer

        # By default, do not change the input
        self.format_probs = format_probs or {'complexity': 0}

    def format(self, batch: dict[str, Any]) -> None:
        batch['complexities'] = []
        batch['input_num'] = []
        for i, input_ids in enumerate(batch['input_ids']):
            complexity = None
            modified_input_ids = batch['input_ids'][i]
            input_num = []

            if random.random() < self.format_probs['complexity']:
                complexity = len(input_ids)
                modified_input_ids = [self.tokenizer['<ctrl_complexity>']] + [self.tokenizer['<num>']] + input_ids
                input_num.append((1, complexity))

            batch['complexities'].append(complexity)
            batch['input_ids'][i] = modified_input_ids
            batch['input_num'].append(input_num)
