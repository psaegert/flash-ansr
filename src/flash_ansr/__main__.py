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
    import_test_data_parser.add_argument('-i', '--input', type=str, required=True, help='Path to the csv file from Biggio et al. or other datasets')
    import_test_data_parser.add_argument('-b', '--base-skeleton-pool', type=str, required=True, help='Path to the base skeleton pool')
    import_test_data_parser.add_argument('-p', '--parser', type=str, required=True, help='Name of the parser to use')
    import_test_data_parser.add_argument('-e', '--expression-space', type=str, required=True, help='Path to the expression space configuration file')
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

    evaluate_nesymres_parser = subparsers.add_parser("evaluate-nesymres")
    evaluate_nesymres_parser.add_argument('-ce', '--config-equation', type=str, required=True, help='Path to the configuration file for the equation setting')
    evaluate_nesymres_parser.add_argument('-cm', '--config-model', type=str, required=True, help='Path to the configuration file for the model')
    evaluate_nesymres_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    evaluate_nesymres_parser.add_argument('-m', '--model', type=str, required=True, help='Path to the model or model configuration')
    evaluate_nesymres_parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the dataset or dataset configuration')
    evaluate_nesymres_parser.add_argument('-e', '--expression-space', type=str, required=True, help='Path to the expression space configuration file')
    evaluate_nesymres_parser.add_argument('-n', '--size', type=int, default=None, help='Size of the dataset')
    evaluate_nesymres_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')
    evaluate_nesymres_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output file')

    evaluate_pysr_parser = subparsers.add_parser("evaluate-pysr")
    evaluate_pysr_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    evaluate_pysr_parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the dataset or dataset configuration')
    evaluate_pysr_parser.add_argument('-e', '--expression-space', type=str, required=True, help='Path to the expression space configuration file')
    evaluate_pysr_parser.add_argument('-n', '--size', type=int, default=None, help='Size of the dataset')
    evaluate_pysr_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')
    evaluate_pysr_parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to the output file')

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

    # Evaluate input
    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'compile-data':
            print('Deprecation Warning: The compile-data function is deprecated in favor of procedurally generated datasets.')

            if args.verbose:
                print(f'[NSR] Compiling data from {args.config}')
            from flash_ansr.data import FlashANSRDataset

            dataset = FlashANSRDataset.from_config(args.config)
            dataset.compile(size=args.size, batch_size=args.batch_size, verbose=args.verbose)
            dataset.save(directory=args.output_dir, config=args.config, reference=args.output_reference, recursive=args.output_recursive)

        case 'generate-skeleton-pool':
            if args.verbose:
                print(f'[NSR] Generating skeleton pool from {args.config}')
            from flash_ansr.expressions import SkeletonPool

            skeleton_pool = SkeletonPool.from_config(args.config)
            skeleton_pool.create(size=int(args.size), verbose=args.verbose)

            if args.verbose:
                print(f"Saving skeleton pool to {args.output_dir}")
            skeleton_pool.save(directory=args.output_dir, config=args.config, reference=args.output_reference, recursive=args.output_recursive)

        case 'import-data':
            if args.verbose:
                print(f'[NSR] Importing data from {args.input}')
            from flash_ansr.expressions import SkeletonPool, ExpressionSpace
            from flash_ansr.compat import ParserFactory
            from flash_ansr.utils import substitute_root_path

            import pandas as pd

            expression_space = ExpressionSpace.from_config(args.expression_space)
            base_skeleton_pool = SkeletonPool.from_config(args.base_skeleton_pool)
            df = pd.read_csv(substitute_root_path(args.input))

            data_parser = ParserFactory.get_parser(args.parser)
            test_skeleton_pool: SkeletonPool = data_parser.parse_data(df, expression_space, base_skeleton_pool, verbose=args.verbose)

            if args.verbose:
                print(f"Saving test set to {args.output_dir}")

            test_skeleton_pool.save(directory=args.output_dir, config=args.base_skeleton_pool, reference='relative', recursive=True)

        case 'split-skeleton-pool':
            print(f'[NSR] Splitting skeleton pool from {args.input}')
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
                print(f'[NSR] Training model from {args.config}')
            from flash_ansr.train.train import Trainer
            from flash_ansr.utils import substitute_root_path, load_config, save_config

            trainer = Trainer.from_config(args.config)

            try:
                trainer.run_from_config(
                    project_name=args.project,
                    entity=args.entity,
                    name=args.name,
                    verbose=args.verbose,
                    checkpoint_interval=args.checkpoint_interval,
                    checkpoint_directory=substitute_root_path(args.output_dir),
                    validate_interval=args.validate_interval,
                    wandb_mode=args.mode)
            except KeyboardInterrupt:
                print("Training interrupted. Saving model...")

            trainer.model.save(directory=args.output_dir, errors='ignore')  # , config=load_config(load_config(args.config)["model"]), reference='relative', recursive=True)

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
                print(f'[NSR] Evaluating config {args.config} with model {args.model} on {args.dataset}')
            import os
            from flash_ansr import FlashANSR, GenerationConfig
            from flash_ansr.eval.evaluation import Evaluation
            from flash_ansr.utils import substitute_root_path, load_config
            from flash_ansr.data import FlashANSRDataset
            from flash_ansr.models import FlashANSRTransformer

            if os.path.isdir(substitute_root_path(args.model)):
                # Load the model
                _, model = FlashANSRTransformer.load(args.model)
                print(f"Model loaded from {args.model}")
            elif os.path.isfile(substitute_root_path(args.model)):
                # The model is specified with a file
                _, model = FlashANSRTransformer.load(substitute_root_path(args.model))
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

            # Use the same expression space as the model for correct tokenization
            # FIXME: Is this necessary?
            # dataset.skeleton_pool.expression_space = model.expression_space

            evaluation = Evaluation.from_config(substitute_root_path(args.config))

            evaluation_config = load_config(substitute_root_path(args.config))

            if 'generation_config' in evaluation_config:
                generation_config = GenerationConfig(**evaluation_config['generation_config'])
            else:
                generation_config = GenerationConfig(
                    method='beam_search',
                    beam_width=evaluation_config['beam_width'],
                    equivalence_pruning=evaluation_config['equivalence_pruning'],
                    max_len=evaluation_config['max_len'],
                )

            results_dict = evaluation.evaluate(
                model=FlashANSR.load(
                    directory=substitute_root_path(args.model),
                    generation_config=generation_config,
                    n_restarts=evaluation_config['n_restarts'],
                    numeric_head=evaluation_config['numeric_head'],
                    refiner_method=evaluation_config.get("refiner_method", 'curve_fit_lm'),
                    refiner_p0_noise=evaluation_config["refiner_p0_noise"],
                    refiner_p0_noise_kwargs=evaluation_config.get("refiner_p0_noise_kwargs", None),
                    parsimony=evaluation_config.get("parsimony", 0),
                ),
                dataset=dataset,
                size=args.size,
                verbose=args.verbose)

            if args.verbose:
                print(f"Saving evaluation results to {args.output_file} ...")

            output_dir = os.path.dirname(substitute_root_path(args.output_file))
            os.makedirs(output_dir, exist_ok=True)

            with open(substitute_root_path(args.output_file), 'wb') as f:
                pickle.dump(results_dict, f)

            if args.verbose:
                print(f"Saved evaluation results to {args.output_file}")

        case 'evaluate-nesymres':
            if args.verbose:
                print(f'[NSR] Evaluating model from {args.model} on {args.dataset}')
            import os
            from flash_ansr import ExpressionSpace
            from flash_ansr.compat.evaluation_nesymres import NeSymReSEvaluation
            from flash_ansr.utils import substitute_root_path, load_config
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
                expression_space=ExpressionSpace.from_config(substitute_root_path(args.expression_space)),
                size=args.size,
                verbose=args.verbose)

            if args.verbose:
                print(f"Saving evaluation results to {args.output_file} ...")

            output_dir = os.path.dirname(substitute_root_path(args.output_file))
            os.makedirs(output_dir, exist_ok=True)

            with open(substitute_root_path(args.output_file), 'wb') as f:
                pickle.dump(results_dict, f)

            if args.verbose:
                print(f"Saved evaluation results to {args.output_file}")

        case 'evaluate-pysr':
            if args.verbose:
                print(f'[NSR] Evaluating PySR on {args.dataset}')
            import os
            from flash_ansr import ExpressionSpace
            from flash_ansr.compat.evaluation_pysr import PySREvaluation
            from flash_ansr.utils import substitute_root_path, load_config
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

            results_dict = evaluation.evaluate(
                dataset=dataset,
                size=args.size,
                expression_space=ExpressionSpace.from_config(substitute_root_path(args.expression_space)),
                verbose=args.verbose)

            if args.verbose:
                print(f"Saving evaluation results to {args.output_file} ...")

            output_dir = os.path.dirname(substitute_root_path(args.output_file))
            os.makedirs(output_dir, exist_ok=True)

            with open(substitute_root_path(args.output_file), 'wb') as f:
                pickle.dump(results_dict, f)

            if args.verbose:
                print(f"Saved evaluation results to {args.output_file}")

        case 'wandb-stats':
            print(f'[NSR] Fetching stats from wandb project {args.project} and entity {args.entity}')
            import os
            import wandb
            import pandas as pd

            from flash_ansr.utils import substitute_root_path

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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path)

        case 'benchmark':
            if args.verbose:
                print(f'[NSR] Benchmarking dataset {args.config}')
            from flash_ansr.data import FlashANSRDataset
            from flash_ansr.utils import substitute_root_path, load_config, save_config
            import pandas as pd

            dataset = FlashANSRDataset.from_config(substitute_root_path(args.config))

            results = dataset._benchmark(n_samples=args.samples, batch_size=args.batch_size, verbose=args.verbose)

            print(f'Iteration time: {1e3 * results["mean_iteration_time"]:.0f} ± {1e3 * results["std_iteration_time"]:.0f} ms')
            print(f'Range:          {1e3 * results["min_iteration_time"]:.0f} - {1e3 * results["max_iteration_time"]:.0f} ms')

        case 'install':
            from flash_ansr.models.manage import install_model
            install_model(args.model)

        case 'remove':
            from flash_ansr.models.manage import remove_model
            remove_model(args.path)

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == '__main__':
    main()
