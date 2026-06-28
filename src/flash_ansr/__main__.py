import datetime
import argparse
import sys


def main(argv: str = None) -> None:
    parser = argparse.ArgumentParser(description='Neural Symbolic Regression')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # NOTE (flash-ansr 0.7): the skeleton-pool data CLI (generate / import / filter / split)
    # moved out of flash-ansr — pool construction, ingestion and holdout now live in the
    # `symbolic-data` package (and its CLI). flash-ansr keeps the model-side commands below.

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
    train_parser.add_argument('--mode', type=str, default='disabled', help="Mode for wandb logging ('online', 'offline', or 'disabled'; default disabled — pass --mode online to log)")
    train_parser.add_argument('--resume-from', type=str, default=None, help='Path to a checkpoint directory to resume from')
    train_parser.add_argument('--resume-step', type=int, default=None, help='Override the inferred resume step when resuming')

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
            from flash_ansr.utils.paths import substitute_root_path

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
