from __future__ import annotations

from collections import Counter
from contextlib import suppress

from flash_ansr import get_path
from flash_ansr.data import FlashANSRDataset


def main() -> None:
    dataset = FlashANSRDataset.from_config(get_path('configs', 'v22.0-60M', 'dataset_train.yaml'))
    stats: Counter[str] = Counter()
    combo_counts: Counter[tuple[bool, bool, bool, bool]] = Counter()

    try:
        iterator = dataset.iterate(
            steps=128,
            batch_size=128,
            n_support=None,
            preprocess=True,
            persistent=True,
            num_workers=2,
            prefetch_factor=1,
            verbose=False,
        )

        for batch in iterator:
            prompt_metadata = batch.get('prompt_metadata')
            prompt_mask = batch.get('prompt_mask')
            input_ids = batch.get('input_ids')
            if prompt_metadata is None or prompt_mask is None or input_ids is None:
                continue

            batch_size = len(prompt_metadata)
            stats['samples'] += batch_size

            for idx in range(batch_size):
                metadata = prompt_metadata[idx]
                allowed_present = bool(metadata['allowed_terms'])
                include_present = bool(metadata['include_terms'])
                exclude_present = bool(metadata['exclude_terms'])

                stats['allowed_nonempty'] += int(allowed_present)
                stats['include_nonempty'] += int(include_present)
                stats['exclude_nonempty'] += int(exclude_present)

                mask = prompt_mask[idx]
                with suppress(TypeError, AttributeError):
                    mask = mask.tolist()
                prompt_enabled = bool(mask) and any(mask)
                stats['prompt_enabled'] += int(prompt_enabled)

                tokens = dataset.tokenizer.decode(input_ids[idx], special_tokens=True)
                complexity_present = '<complexity>' in tokens
                allowed_tokens_present = '<allowed_term>' in tokens
                include_tokens_present = '<include_term>' in tokens
                exclude_tokens_present = '<exclude_term>' in tokens

                stats['complexity_tokens'] += int(complexity_present)
                stats['allowed_tokens'] += int(allowed_tokens_present)
                stats['include_tokens'] += int(include_tokens_present)
                stats['exclude_tokens'] += int(exclude_tokens_present)

                combo_counts[(allowed_present, include_present, exclude_present, complexity_present)] += 1

    finally:
        dataset.shutdown()

    samples = max(stats['samples'], 1)
    print('Total samples:', samples)
    for key in sorted(k for k in stats if k != 'samples'):
        print(f"{key:>20}: {stats[key]} ({stats[key] / samples:.2%})")

    print('\nCo-occurrence (allowed, include, exclude, complexity):')
    for combo, count in sorted(combo_counts.items()):
        allowed_present, include_present, exclude_present, complexity_present = combo
        combo_label = ''.join(str(int(flag)) for flag in combo)
        print(
            f"  {combo_label} -> {count} ({count / samples:.2%})",
            f"[allowed={allowed_present}, include={include_present}, exclude={exclude_present}, complexity={complexity_present}]",
        )


if __name__ == '__main__':
    main()
