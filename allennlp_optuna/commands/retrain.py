import argparse
from overrides import overrides
import os
import tempfile

from optuna import load_study
from optuna.integration.allennlp import dump_best_config

from allennlp.commands.subcommand import Subcommand

from allennlp.commands.train import train_model_from_file
from allennlp_optuna.commands.best_params import fetch_best_params


def train_model_from_args_with_optuna(args: argparse.Namespace):
    # Set hyperparameters
    for k, v in fetch_best_params(args.storage, args.study_name).items():
        os.environ[k] = str(v)

    study = load_study(study_name=args.study_name, storage=args.storage)
    with tempfile.TemporaryDirectory() as tmp_param_dir:
        tmp_param_path = os.path.join(tmp_param_dir, "allennlp.jsonnet")
        dump_best_config(args.param_path, tmp_param_path, study)

        train_model_from_file(
            parameter_filename=tmp_param_path,
            serialization_dir=args.serialization_dir,
            overrides=args.overrides,
            recover=args.recover,
            force=args.force,
            node_rank=0,
            include_package=args.include_package,
            dry_run=False,
            file_friendly_logging=False,
        )


@Subcommand.register("retrain")
class Retrain(Subcommand):
    """Retraining a model."""
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = (
            """Train the specified model on the specified dataset using hyperparameters found by Optuna."""
        )
        subparser = parser.add_parser(self.name, description=description, help="Train a model with hyperparameter found by Optuna.")

        subparser.add_argument(
            "param_path", type=str, help="path to parameter file describing the model to be trained"
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir",
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--node-rank", type=int, default=0, help="rank of this node in the distributed setup"
        )

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "do not train a model, but create a vocabulary, show dataset statistics and "
                "other training information"
            ),
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        # Optuna
        subparser.add_argument(
            "--study-name", default=None, help="The name of the study to start optimization on."
        )

        subparser.add_argument(
            "--storage",
            type=str,
            help=(
                "The path to storage. "
                "allennlp-optuna supports a valid URL" "for sqlite3, mysql, postgresql, or redis."
            ),
            default="sqlite:///allennlp_optuna.db",
        )

        subparser.set_defaults(func=train_model_from_args_with_optuna)
        return subparser
