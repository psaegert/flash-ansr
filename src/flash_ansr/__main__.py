import datetime
import argparse
import sys
from copy import deepcopy


def main(argv: str = None) -> None:
    parser = argparse.ArgumentParser(description='Neural Symbolic Regression')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    generate_skeleton_pool_parser = subparsers.add_parser("generate-skeleton-pool")
    generate_skeleton_pool_parser.add_argument('-s', '--size', type=str, required=True, help='Size of the skeleton pool')
    generate_skeleton_pool_parser.add_argument('-o', '--output-dir', type=str, required=True, help='Path to the output directory')
    generate_skeleton_pool_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    generate_skeleton_pool_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')
    generate_skeleton_pool_parser.add_argument('--output-reference', type=str, default='relative', help='Reference type for the output directory')
    generate_skeleton_pool_parser.add_argument('--output-recursive', type=bool, default=True, help='Whether to recursively save the configuration')

    import_test_data_parser = subparsers.add_parser("import-data")
    import_test_data_parser.add_argument('-i', '--input', type=str, required=True, help='Path to the dataset file (CSV or YAML) from Biggio et al. or other benchmarks')
    import_test_data_parser.add_argument('-b', '--base-skeleton-pool', type=str, required=True, help='Path to the base skeleton pool')
    import_test_data_parser.add_argument('-p', '--parser', type=str, required=True, help='Name of the parser to use')
    import_test_data_parser.add_argument('-e', '--simplipy-engine', type=str, required=True, help='Path to the expression space configuration file')
    import_test_data_parser.add_argument('-o', '--output-dir', type=str, required=True, help='Path to the output directory')
    import_test_data_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    filter_skeleton_pool_parser = subparsers.add_parser("filter-skeleton-pool")
    filter_skeleton_pool_parser.add_argument('-s', '--source', type=str, required=True, help='Path to the source skeleton pool')
    filter_skeleton_pool_parser.add_argument('-f', '--holdouts', nargs='+', required=True, help='Paths to the holdout skeleton pools')
    filter_skeleton_pool_parser.add_argument('-o', '--output-dir', type=str, required=True, help='Path to the output directory')
    filter_skeleton_pool_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    split_skeleton_pool_parser = subparsers.add_parser("split-skeleton-pool")
    split_skeleton_pool_parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input skeleton pool')
    split_skeleton_pool_parser.add_argument('-t', '--train-size', type=float, default=0.8, help='Size of the training set')
    split_skeleton_pool_parser.add_argument('-r', '--random-state', type=int, default=None, help='Random seed for shuffling')

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    train_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')
    train_parser.add_argument('-o', '--output-dir', type=str, default='.', help='Path to the output directory')
    train_parser.add_argument('-ci', '--checkpoint-interval', type=int, default=None, help='Interval for saving checkpoints')
    train_parser.add_argument('-vi', '--validate-interval', type=int, default=None, help='Interval for validating the model')
    train_parser.add_argument('-w', '--num_workers', type=int, default=None, help='Number of worker processes for data generation')
    train_parser.add_argument('--project', type=str, default='neural-symbolic-regression', help='Name of the wandb project')
    train_parser.add_argument('--entity', type=str, default='psaegert', help='Name of the wandb entity')
    train_parser.add_argument('--name', type=str, default=None, help='Name of the wandb run')
    train_parser.add_argument('--mode', type=str, default='online', help='Mode for wandb logging')
    train_parser.add_argument('--resume-from', type=str, default=None, help='Path to a checkpoint directory to resume from')
    train_parser.add_argument('--resume-step', type=int, default=None, help='Override the inferred resume step when resuming')

    evaluate_run_parser = subparsers.add_parser("evaluate-run", help="Run an evaluation from a unified config")
    evaluate_run_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the evaluation run config file')
    evaluate_run_parser.add_argument('-n', '--limit', type=int, default=None, help='Override the sample limit specified in the config')
    evaluate_run_parser.add_argument('-o', '--output-file', type=str, default=None, help='Override the output file path from the config')
    evaluate_run_parser.add_argument('--save-every', type=int, default=None, help='Override periodic save frequency')
    evaluate_run_parser.add_argument('--no-resume', action='store_true', help='Ignore previous results even if the output file exists')
    evaluate_run_parser.add_argument('--experiment', type=str, default=None, help='Name of the experiment defined in the config to execute')
    evaluate_run_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    wandb_stats_parser = subparsers.add_parser("wandb-stats")
    wandb_stats_parser.add_argument('--project', type=str, default='neural-symbolic-regression', help='Name of the wandb project')
    wandb_stats_parser.add_argument('--entity', type=str, default='psaegert', help='Name of the wandb entity')
    wandb_stats_parser.add_argument('-o', '--output-file', type=str, default='wandb_stats.csv', help='Path to the output file')

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the dataset configuration file')
    benchmark_parser.add_argument('-n', '--samples', type=int, default=10_000, help='Number of samples to evaluate')
    benchmark_parser.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size for the dataset')
    benchmark_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    install_parser = subparsers.add_parser("install", help="Install a model")
    install_parser.add_argument("model", type=str, help="Model identifier to install")

    remove_parser = subparsers.add_parser("remove", help="Remove a model")
    remove_parser.add_argument("path", type=str, help="Path to the model to remove")

    find_simplifications_parser = subparsers.add_parser("find-simplifications")
    find_simplifications_parser.add_argument('-e', '--simplipy-engine', type=str, required=True, help='Path to the expression space configuration file')
    find_simplifications_parser.add_argument('-n', '--max_n_rules', type=int, default=None, help='Maximum number of rules to find')
    find_simplifications_parser.add_argument('-l', '--max_pattern_length', type=int, default=7, help='Maximum length of the patterns to find')
    find_simplifications_parser.add_argument('-t', '--timeout', type=int, default=None, help='Timeout for the search of simplifications in seconds')
    find_simplifications_parser.add_argument('-d', '--dummy-variables', type=int, nargs='+', default=None, help='Dummy variables to use in the simplifications')
    find_simplifications_parser.add_argument('-m', '--max-simplify-steps', type=int, default=5, help='Maximum number of simplification steps')
    find_simplifications_parser.add_argument('-x', '--X', type=int, default=1024, help='Number of samples to use for comparison of images')
    find_simplifications_parser.add_argument('-c', '--C', type=int, default=1024, help='Number of samples of constants to put in to placeholders')
    find_simplifications_parser.add_argument('-r', '--constants-fit-retries', type=int, default=5, help='Number of retries for fitting the constants')
    find_simplifications_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output json file')
    find_simplifications_parser.add_argument('-s', '--save-every', type=int, default=100, help='Save the simplifications every n rules')
    find_simplifications_parser.add_argument('--reset-rules', action='store_true', help='Reset the rules before finding new ones')
    find_simplifications_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    # Evaluate input
    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'generate-skeleton-pool':
            if args.verbose:
                print(f'Generating skeleton pool from {args.config}')
            from flash_ansr.expressions import SkeletonPool

            skeleton_pool = SkeletonPool.from_config(args.config)
            skeleton_pool.create(size=int(args.size), verbose=args.verbose)

            if args.verbose:
                print(f"Saving skeleton pool to {args.output_dir}")
            skeleton_pool.save(directory=args.output_dir, config=args.config, reference=args.output_reference, recursive=args.output_recursive)

        case 'import-data':
            if args.verbose:
                print(f'Importing data from {args.input}')
            from simplipy import SimpliPyEngine
            from flash_ansr.expressions import SkeletonPool
            from flash_ansr.compat import ParserFactory
            from flash_ansr.utils.config_io import load_config
            from flash_ansr.utils.paths import substitute_root_path

            import pandas as pd
            import yaml
            from pathlib import Path

            simplipy_engine = SimpliPyEngine.load(args.simplipy_engine, install=True)
            base_skeleton_pool = SkeletonPool.from_config(args.base_skeleton_pool)
            input_path = substitute_root_path(args.input)
            path_obj = Path(input_path)

            if path_obj.suffix.lower() in {'.yaml', '.yml'}:
                with open(input_path, 'r', encoding='utf-8') as handle:
                    raw_data = yaml.safe_load(handle)

                if not isinstance(raw_data, dict):
                    raise ValueError('Expected YAML benchmark file to contain a mapping of equation identifiers to entries.')

                records = []
                for identifier, payload in raw_data.items():
                    if not isinstance(payload, dict):
                        continue

                    record = {'id': identifier}
                    record.update(payload)
                    if 'prepared' in record and record['prepared'] is None:
                        # Normalise missing prepared expressions to empty strings for downstream filtering.
                        record['prepared'] = ''
                    records.append(record)

                df = pd.DataFrame.from_records(records)
            else:
                df = pd.read_csv(input_path)

            data_parser = ParserFactory.get_parser(args.parser)
            test_skeleton_pool: SkeletonPool = data_parser.parse_data(df, simplipy_engine, base_skeleton_pool, verbose=args.verbose)

            if args.verbose:
                print(f"Saving test set to {args.output_dir}")

            test_skeleton_pool.save(directory=args.output_dir, config=args.base_skeleton_pool, reference='relative', recursive=True)

        case 'split-skeleton-pool':
            print(f'Splitting skeleton pool from {args.input}')
            import os
            from flash_ansr.expressions import SkeletonPool

            print(f"Loading skeleton pool from {args.input}")

            config, skeleton_pool = SkeletonPool.load(args.input)
            train_skeleton_pool, val_skeleton_pool = skeleton_pool.split(train_size=args.train_size, random_state=args.random_state)

            train_path = os.path.join(args.input, 'train')
            val_path = os.path.join(args.input, 'val')

            train_config = deepcopy(config)
            val_config = deepcopy(config)

            print(f"Saving training pool to {train_path}")
            print(f"Saving validation pool to {val_path}")

            train_skeleton_pool.save(directory=train_path, config=train_config, reference='relative', recursive=True)
            val_skeleton_pool.save(directory=val_path, config=val_config, reference='relative', recursive=True)

        case 'train':
            if args.verbose:
                print(f'Training model from {args.config}')
            from flash_ansr.train.train import Trainer
            from flash_ansr.utils.config_io import load_config, save_config
            from flash_ansr.utils.paths import substitute_root_path

            trainer = Trainer.from_config(args.config)

            config = load_config(args.config)

            try:
                trainer.run(
                    project_name=args.project,
                    entity=args.entity,
                    name=args.name,
                    steps=config['steps'],
                    preprocess=config.get('preprocess', False),
                    device=config['device'],
                    compile_mode=config.get('compile_mode'),
                    checkpoint_interval=args.checkpoint_interval,
                    checkpoint_directory=substitute_root_path(args.output_dir),
                    validate_interval=args.validate_interval,
                    validate_size=config.get('val_size', None),
                    validate_batch_size=config.get('val_batch_size', None),
                    wandb_watch_log=config.get('wandb_watch_log', None),
                    wandb_watch_log_freq=config.get('wandb_watch_log_freq', 1000),
                    wandb_mode=args.mode,
                    num_workers=args.num_workers,
                    resume_from=args.resume_from,
                    resume_step=args.resume_step,
                    verbose=args.verbose,
                )
            except KeyboardInterrupt:
                print("Training interrupted. Saving model...")

            trainer.model.save(directory=args.output_dir, errors='ignore')

            save_config(
                load_config(args.config, resolve_paths=True),
                directory=substitute_root_path(args.output_dir),
                filename='train.yaml',
                reference='relative',
                recursive=True,
                resolve_paths=True)

            print(f"Saved model to {args.output_dir}")

        case 'evaluate-run':
            from flash_ansr.eval.run_config import build_evaluation_run, EvaluationRunPlan
            from flash_ansr.utils.config_io import load_config
            from flash_ansr.utils.paths import substitute_root_path

            config_path = substitute_root_path(args.config)
            if args.verbose:
                print(f"Running evaluation plan from {config_path}")

            raw_config = load_config(config_path)
            experiment_map = raw_config.get("experiments") if isinstance(raw_config, dict) else None

            def _execute_plan(plan: EvaluationRunPlan, experiment_name: str | None = None) -> None:
                label = f"[{experiment_name}] " if experiment_name else ""
                if plan.completed or plan.engine is None:
                    if args.verbose:
                        target = plan.total_limit or 'configured'
                        print(f"{label}Evaluation already completed ({plan.existing_results}/{target}). Nothing to do.")
                    return

                plan.engine.run(
                    limit=plan.remaining,
                    save_every=plan.save_every,
                    output_path=plan.output_path,
                    verbose=args.verbose,
                    progress=args.verbose,
                )

                if args.verbose:
                    total = plan.engine.result_store.size
                    destination = plan.output_path or 'memory'
                    print(f"{label}Evaluation finished with {total} samples (saved to {destination}).")

            if experiment_map and args.experiment is None:
                experiment_names = list(experiment_map.keys())
                if args.verbose:
                    count = len(experiment_names)
                    print(f"No --experiment provided; running all {count} experiments defined in config.")
                for experiment_name in experiment_names:
                    if args.verbose:
                        print(f"--> {experiment_name}")
                    plan = build_evaluation_run(
                        config=config_path,
                        limit_override=args.limit,
                        output_override=args.output_file,
                        save_every_override=args.save_every,
                        resume=None if not args.no_resume else False,
                        experiment=experiment_name,
                    )
                    _execute_plan(plan, experiment_name)
            else:
                plan = build_evaluation_run(
                    config=config_path,
                    limit_override=args.limit,
                    output_override=args.output_file,
                    save_every_override=args.save_every,
                    resume=None if not args.no_resume else False,
                    experiment=args.experiment,
                )
                _execute_plan(plan, args.experiment)

        case 'wandb-stats':
            print(f'Fetching stats from wandb project {args.project} and entity {args.entity}')
            import os
            import wandb
            import pandas as pd

            from flash_ansr.utils.paths import substitute_root_path

            api = wandb.Api()  # type: ignore

            runs = api.runs(f'{args.entity}/{args.project}')
            runs = {run.id: {'run': run} for run in runs}

            for key, value in runs.items():
                start_time = datetime.datetime.strptime(value['run'].created_at, '%Y-%m-%dT%H:%M:%S') + datetime.timedelta(hours=2)  # HACK: This is a hack to convert to CET
                end_time = datetime.datetime.strptime(value['run'].heartbeatAt, '%Y-%m-%dT%H:%M:%S') + datetime.timedelta(hours=2)
                runs[key]['start_time'] = start_time
                runs[key]['end_time'] = end_time
                runs[key]['duration'] = end_time - start_time
                runs[key]['name'] = value['run'].name

                df = pd.DataFrame.from_dict(runs, orient='index').drop(columns=['run'])

            save_path = substitute_root_path(args.output_file)
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path)

        case 'benchmark':
            if args.verbose:
                print(f'Benchmarking dataset {args.config}')
            from flash_ansr.data import FlashANSRDataset
            from flash_ansr.utils.config_io import load_config, save_config
            from flash_ansr.utils.paths import substitute_root_path
            import pandas as pd

            dataset = FlashANSRDataset.from_config(substitute_root_path(args.config))

            results = dataset._benchmark(n_samples=args.samples, batch_size=args.batch_size, verbose=args.verbose)

            print(f'Iteration time: {1e3 * results["mean_iteration_time"]:.0f} ± {1e3 * results["std_iteration_time"]:.0f} ms')
            print(f'Range:          {1e3 * results["min_iteration_time"]:.0f} - {1e3 * results["max_iteration_time"]:.0f} ms')

        case 'install':
            from flash_ansr.model.manage import install_model
            install_model(args.model)

        case 'remove':
            from flash_ansr.model.manage import remove_model
            remove_model(args.path)

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == '__main__':
    main()
