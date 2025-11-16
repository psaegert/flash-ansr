import datetime
import argparse
import sys
import pickle
from copy import deepcopy


def main(argv: str = None) -> None:
    parser = argparse.ArgumentParser(description='Neural Symbolic Regression')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    compile_data_parser = subparsers.add_parser("compile-data")
    compile_data_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    compile_data_parser.add_argument('-n', '--size', type=int, default=None, help='Size of the dataset')
    compile_data_parser.add_argument('-b', '--batch-size', type=int, default=None, help='Batch size for the dataset')
    compile_data_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')
    compile_data_parser.add_argument('-o', '--output-dir', type=str, required=True, help='Path to the output directory')
    compile_data_parser.add_argument('--output-reference', type=str, default='relative', help='Reference type for the output directory')
    compile_data_parser.add_argument('--output-recursive', type=bool, default=True, help='Whether to recursively save the configuration')

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

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    evaluate_parser.add_argument('-m', '--model', type=str, required=True, help='Path to the model or model configuration')
    evaluate_parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the dataset or dataset configuration')
    evaluate_parser.add_argument('-n', '--size', type=int, default=None, help='Size of the dataset')
    evaluate_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')
    evaluate_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output file')
    evaluate_parser.add_argument('-s', '--save-every', type=int, default=5, help='Save the evaluation results every n samples')

    evaluate_fastsrb_parser = subparsers.add_parser("evaluate-fastsrb")
    evaluate_fastsrb_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the evaluation configuration file')
    evaluate_fastsrb_parser.add_argument('-m', '--model', type=str, required=True, help='Path to the model directory or configuration')
    evaluate_fastsrb_parser.add_argument('-b', '--benchmark', type=str, default=None, help='Path to the FastSRB benchmark YAML file (overrides config)')
    evaluate_fastsrb_parser.add_argument('-e', '--equations', nargs='+', default=None, help='Subset of equation identifiers to evaluate')
    evaluate_fastsrb_parser.add_argument('-n', '--size', type=int, default=None, help='Maximum number of benchmark problems to evaluate')
    evaluate_fastsrb_parser.add_argument('-s', '--save-every', type=int, default=5, help='Save the evaluation results every n samples')
    evaluate_fastsrb_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output file')
    evaluate_fastsrb_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    evaluate_nesymres_parser = subparsers.add_parser("evaluate-nesymres")
    evaluate_nesymres_parser.add_argument('-ce', '--config-equation', type=str, required=True, help='Path to the configuration file for the equation setting')
    evaluate_nesymres_parser.add_argument('-cm', '--config-model', type=str, required=True, help='Path to the configuration file for the model')
    evaluate_nesymres_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    evaluate_nesymres_parser.add_argument('-m', '--model', type=str, required=True, help='Path to the model or model configuration')
    evaluate_nesymres_parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the dataset or dataset configuration')
    evaluate_nesymres_parser.add_argument('-e', '--simplipy-engine', type=str, required=True, help='Path to the expression space configuration file')
    evaluate_nesymres_parser.add_argument('-n', '--size', type=int, default=None, help='Size of the dataset')
    evaluate_nesymres_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')
    evaluate_nesymres_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output file')

    evaluate_pysr_parser = subparsers.add_parser("evaluate-pysr")
    evaluate_pysr_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    evaluate_pysr_parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the dataset or dataset configuration')
    evaluate_pysr_parser.add_argument('-e', '--simplipy-engine', type=str, required=True, help='Path to the expression space configuration file')
    evaluate_pysr_parser.add_argument('-n', '--size', type=int, default=None, help='Size of the dataset')
    evaluate_pysr_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')
    evaluate_pysr_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output file')

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
        case 'compile-data':
            print('Deprecation Warning: The compile-data function is deprecated in favor of procedurally generated datasets.')

            if args.verbose:
                print(f'Compiling data from {args.config}')
            from flash_ansr.data import FlashANSRDataset

            dataset = FlashANSRDataset.from_config(args.config)
            dataset.compile(size=args.size, batch_size=args.batch_size, verbose=args.verbose)
            dataset.save(directory=args.output_dir, config=args.config, reference=args.output_reference, recursive=args.output_recursive)

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

        case 'evaluate':
            if args.verbose:
                print(f'Evaluating config {args.config} with model {args.model} on {args.dataset}')
            import os
            from flash_ansr import FlashANSR
            from flash_ansr.eval.evaluation import Evaluation
            from flash_ansr.utils.config_io import load_config, unfold_config
            from flash_ansr.utils.generation import create_generation_config
            from flash_ansr.utils.paths import substitute_root_path
            from flash_ansr.data import FlashANSRDataset
            from flash_ansr.model import FlashANSRModel
            import pprint

            if os.path.isdir(substitute_root_path(args.model)):
                # Load the model
                _, model = FlashANSRModel.load(args.model)
                print(f"Model loaded from {args.model}")
            elif os.path.isfile(substitute_root_path(args.model)):
                # The model is specified with a file
                _, model = FlashANSRModel.load(substitute_root_path(args.model))
                print(f"Model loaded from {substitute_root_path(args.model)}")
            else:
                raise ValueError(f"Invalid model configuration: {args.model}")

            if os.path.isdir(substitute_root_path(args.dataset)):
                # Load the dataset
                _, dataset = FlashANSRDataset.load(args.dataset)
                print(f"Dataset loaded from {args.dataset}")
            elif isinstance(args.dataset, dict) or os.path.isfile(substitute_root_path(args.dataset)):
                # The dataset is specified with a config dict or file
                dataset = FlashANSRDataset.from_config(substitute_root_path(args.dataset))
                print(f"Dataset initialized from config {args.dataset}")
            else:
                raise ValueError(f"Invalid dataset configuration: {args.dataset}")

            evaluation = Evaluation.from_config(substitute_root_path(args.config))

            if args.verbose:
                print(f"Loaded evaluation config from {args.config}")
                pprint.pprint(unfold_config(load_config(substitute_root_path(args.config))))

            evaluation_config = load_config(substitute_root_path(args.config))

            generation_config = create_generation_config(
                method=evaluation_config['generation_config']['method'],
                **evaluation_config['generation_config'].get('kwargs', {}),
            )

            size_todo = args.size
            resolved_output_file = substitute_root_path(args.output_file)
            results_dict = None

            if os.path.exists(resolved_output_file):
                if args.verbose:
                    print(f"Loading existing evaluation results from {args.output_file} ...")
                with open(resolved_output_file, 'rb') as read_handle:
                    results_dict = pickle.load(read_handle)

                if size_todo is not None and results_dict:
                    processed = len(results_dict['expression']) if 'expression' in results_dict else len(next(iter(results_dict.values())))
                    size_todo -= processed
            else:
                results_dict = None

            if size_todo is not None and size_todo <= 0:
                if args.verbose:
                    print(f"Evaluation already completed for {args.size} samples. Exiting.")
                sys.exit(0)

            results_dict = evaluation.evaluate(
                model=FlashANSR.load(
                    directory=substitute_root_path(args.model),
                    generation_config=generation_config,
                    n_restarts=evaluation_config['n_restarts'],
                    refiner_method=evaluation_config.get("refiner_method", 'curve_fit_lm'),
                    refiner_p0_noise=evaluation_config["refiner_p0_noise"],
                    refiner_p0_noise_kwargs=evaluation_config.get("refiner_p0_noise_kwargs", None),
                    parsimony=evaluation_config['parsimony'],
                    refiner_workers=evaluation_config.get("refiner_workers", None),
                ),
                dataset=dataset,
                results_dict=results_dict,
                size=size_todo,
                save_every=args.save_every,
                output_file=args.output_file,
                verbose=args.verbose)

            if args.verbose:
                print(f"Saving evaluation results to {args.output_file} ...")

            output_dir = os.path.dirname(resolved_output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(resolved_output_file, 'wb') as write_handle:
                pickle.dump(results_dict, write_handle)

            if args.verbose:
                print(f"Saved evaluation results to {args.output_file}")

        case 'evaluate-fastsrb':
            if args.verbose:
                print(f"Evaluating FastSRB benchmark with model {args.model}")

            import os
            import pprint
            from flash_ansr import FlashANSR
            from flash_ansr.benchmarks import FastSRBBenchmark
            from flash_ansr.eval import FastSRBEvaluation
            from flash_ansr.model import FlashANSRModel
            from flash_ansr.utils.config_io import load_config, unfold_config
            from flash_ansr.utils.generation import create_generation_config
            from flash_ansr.utils.paths import substitute_root_path

            evaluation_config_path = substitute_root_path(args.config)
            evaluation = FastSRBEvaluation.from_config(evaluation_config_path)

            if args.verbose:
                print(f"Loaded evaluation config from {args.config}")
                pprint.pprint(unfold_config(load_config(evaluation_config_path)))

            benchmark_path_value = args.benchmark or evaluation.benchmark_path
            resolved_benchmark = substitute_root_path(benchmark_path_value)
            if not os.path.isfile(resolved_benchmark):
                raise FileNotFoundError(f"Benchmark specification not found: {resolved_benchmark}")
            if args.verbose:
                print(f"Using FastSRB benchmark {benchmark_path_value}")

            evaluation_config = load_config(evaluation_config_path)
            generation_config = create_generation_config(
                method=evaluation_config['generation_config']['method'],
                **evaluation_config['generation_config'].get('kwargs', {}),
            )

            model_path = substitute_root_path(args.model)
            if os.path.isdir(model_path) or os.path.isfile(model_path):
                FlashANSRModel.load(model_path)
                if args.verbose:
                    print(f"Model loaded from {args.model}")
            else:
                raise ValueError(f"Invalid model configuration: {args.model}")

            benchmark = FastSRBBenchmark(resolved_benchmark, random_state=evaluation.benchmark_random_state)

            resolved_output_file = substitute_root_path(args.output_file)
            results_dict = None
            if os.path.exists(resolved_output_file):
                if args.verbose:
                    print(f"Loading existing FastSRB evaluation results from {args.output_file} ...")
                with open(resolved_output_file, 'rb') as read_handle:
                    results_dict = pickle.load(read_handle)

            results_dict = evaluation.evaluate(
                model=FlashANSR.load(
                    directory=model_path,
                    generation_config=generation_config,
                    n_restarts=evaluation_config['n_restarts'],
                    refiner_method=evaluation_config.get('refiner_method', 'curve_fit_lm'),
                    refiner_p0_noise=evaluation_config['refiner_p0_noise'],
                    refiner_p0_noise_kwargs=evaluation_config.get('refiner_p0_noise_kwargs', None),
                    parsimony=evaluation_config['parsimony'],
                    device=evaluation_config['device'],
                    refiner_workers=evaluation_config.get('refiner_workers', None),
                ),
                benchmark=benchmark,
                eq_ids=args.equations,
                results_dict=results_dict,
                size=args.size,
                save_every=args.save_every,
                output_file=args.output_file,
                verbose=args.verbose,
            )

            if args.verbose:
                print(f"Saving FastSRB evaluation results to {args.output_file} ...")

            output_dir = os.path.dirname(resolved_output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(resolved_output_file, 'wb') as write_handle:
                pickle.dump(results_dict, write_handle)

            if args.verbose:
                print(f"Saved FastSRB evaluation results to {args.output_file}")

        case 'evaluate-nesymres':
            if args.verbose:
                print(f'Evaluating model from {args.model} on {args.dataset}')
            import os
            from simplipy import SimpliPyEngine
            from flash_ansr.compat.evaluation_nesymres import NeSymReSEvaluation
            from flash_ansr.utils.config_io import load_config
            from flash_ansr.utils.paths import substitute_root_path
            from flash_ansr.data import FlashANSRDataset
            from flash_ansr.compat.nesymres import load_nesymres

            evaluation_config = load_config(substitute_root_path(args.config))

            model, fitfunc = load_nesymres(
                eq_setting_path=substitute_root_path(args.config_equation),
                config_path=substitute_root_path(args.config_model),
                weights_path=substitute_root_path(args.model),
                beam_size=evaluation_config['beam_width'],
                n_restarts=evaluation_config['n_restarts'],
                device=evaluation_config['device']
            )

            if os.path.isdir(substitute_root_path(args.dataset)):
                # Load the dataset
                _, dataset = FlashANSRDataset.load(args.dataset)
                print(f"Dataset loaded from {args.dataset}")
            elif isinstance(args.dataset, dict) or os.path.isfile(substitute_root_path(args.dataset)):
                # The dataset is specified with a config dict or file
                dataset = FlashANSRDataset.from_config(substitute_root_path(args.dataset))
                print(f"Dataset initialized from config {args.dataset}")
            else:
                raise ValueError(f"Invalid dataset configuration: {args.dataset}")

            evaluation = NeSymReSEvaluation.from_config(substitute_root_path(args.config))

            results_dict = evaluation.evaluate(
                model=model,
                fitfunc=fitfunc,
                dataset=dataset,
                simplipy_engine=SimpliPyEngine.load(args.simplipy_engine, install=True),
                size=args.size,
                verbose=args.verbose)

            if args.verbose:
                print(f"Saving evaluation results to {args.output_file} ...")

            output_dir = os.path.dirname(substitute_root_path(args.output_file))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(substitute_root_path(args.output_file), 'wb') as write_handle:
                pickle.dump(results_dict, write_handle)

            if args.verbose:
                print(f"Saved evaluation results to {args.output_file}")

        case 'evaluate-pysr':
            if args.verbose:
                print(f'Evaluating PySR on {args.dataset}')
            import os
            from simplipy import SimpliPyEngine
            from flash_ansr.compat.evaluation_pysr import PySREvaluation
            from flash_ansr.utils.config_io import load_config
            from flash_ansr.utils.paths import substitute_root_path
            from flash_ansr.data import FlashANSRDataset

            evaluation_config = load_config(substitute_root_path(args.config))

            if os.path.isdir(substitute_root_path(args.dataset)):
                # Load the dataset
                _, dataset = FlashANSRDataset.load(args.dataset)
                print(f"Dataset loaded from {args.dataset}")
            elif isinstance(args.dataset, dict) or os.path.isfile(substitute_root_path(args.dataset)):
                # The dataset is specified with a config dict or file
                dataset = FlashANSRDataset.from_config(substitute_root_path(args.dataset))
                print(f"Dataset initialized from config {args.dataset}")
            else:
                raise ValueError(f"Invalid dataset configuration: {args.dataset}")

            evaluation = PySREvaluation.from_config(substitute_root_path(args.config))

            size_todo = args.size

            if os.path.exists(substitute_root_path(args.output_file)):
                if args.verbose:
                    print(f"Loading existing evaluation results from {args.output_file} ...")

                with open(substitute_root_path(args.output_file), 'rb') as read_handle:
                    results_dict = pickle.load(read_handle)

                if size_todo is not None:
                    size_todo -= len(results_dict['expression'])  # type: ignore
            else:
                results_dict = None

            if size_todo is not None and size_todo <= 0:
                if args.verbose:
                    print(f"Evaluation already completed for {args.size} samples. Exiting.")
                sys.exit(0)

            results_dict = evaluation.evaluate(
                dataset=dataset,
                results_dict=results_dict,
                size=size_todo,
                save_every=1,
                output_file=args.output_file,
                verbose=args.verbose)

            if args.verbose:
                print(f"Saving evaluation results to {args.output_file} ...")

            output_dir = os.path.dirname(substitute_root_path(args.output_file))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(substitute_root_path(args.output_file), 'wb') as write_handle:
                pickle.dump(results_dict, write_handle)

            if args.verbose:
                print(f"Saved evaluation results to {args.output_file}")

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
