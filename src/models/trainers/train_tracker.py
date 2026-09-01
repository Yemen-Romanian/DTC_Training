"""Entry point for tracker training.

Any config value can be overridden from the command line, so a run variation needs no edit to
the TOML: ``--seed 3``, ``--set training_params.lr=0.001``, ``--unset train_path.synthetic``.
This is what lets ``run_experiments.py`` drive a chain of runs off one base config.
"""
import argparse

from models.trainers.trainer import Trainer
from models.trainers.trainables_factory import create_trainable
from utils.config import Config, to_toml_literal
from utils.seeding import apply_seed


#: Convenience flags, and the config key each one writes. They are lowered into the same
#: --set machinery so there is a single override path, and --set is applied afterwards so an
#: explicit --set always wins over the sugar.
SUGAR_FLAGS = {
    'seed': 'training_params.seed',
    'epochs': 'training_params.epochs_num',
    'lr': 'training_params.lr',
    'batch_size': 'training_params.batch_size',
    'data_workers': 'training_params.data_workers_num',
    'experiment_name': 'experiment_name',
    'run_name': 'run_name',
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Main script for tracker training from scratch",
        epilog="Example: --seed 3 --epochs 30 --unset train_path.synthetic "
               "--set train_path.visdrone=C:/datasets/VisDrone/train",
    )
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                        help="Override any config key by dotted path; repeatable. The value is "
                             "parsed as a TOML literal (42, 0.005, true, [4, 9]) and kept as a "
                             "plain string otherwise - so pass Windows paths unquoted.")
    parser.add_argument('--unset', dest='removals', action='append', default=[], metavar='KEY',
                        help="Remove a config key by dotted path; repeatable. Applied before "
                             "--set, so the two together read as replacement.")

    parser.add_argument('--seed', type=int, help="Shorthand for --set training_params.seed=...")
    parser.add_argument('--epochs', type=int, help="Shorthand for --set training_params.epochs_num=...")
    parser.add_argument('--lr', type=float, help="Shorthand for --set training_params.lr=...")
    parser.add_argument('--batch_size', type=int, help="Shorthand for --set training_params.batch_size=...")
    parser.add_argument('--data_workers', type=int,
                        help="Shorthand for --set training_params.data_workers_num=...")
    parser.add_argument('--experiment_name', type=str,
                        help="MLflow experiment this run belongs to.")
    parser.add_argument('--run_name', type=str,
                        help="Prefix for this run's output/ directory (a timestamp is appended).")
    return parser


def collect_overrides(args) -> list:
    """Sugar flags first, then explicit --set, so --set wins on a conflict."""
    sugar = [f"{key}={to_toml_literal(getattr(args, flag))}"
             for flag, key in SUGAR_FLAGS.items() if getattr(args, flag, None) is not None]
    return sugar + args.overrides


def main(argv=None):
    args = build_parser().parse_args(argv)

    config = Config(args.config_path)
    config.apply_overrides(collect_overrides(args), args.removals)

    # Before create_trainable: that is where the weights are randomly initialized.
    apply_seed(config.get_training_param('seed', None))
    trainable_model = create_trainable(config.get_model_config())
    trainer = Trainer(trainable_model, config)
    trainer.train()


if __name__ == '__main__':
    main()
