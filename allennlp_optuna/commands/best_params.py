import argparse
from overrides import overrides
from typing import Any
from typing import Dict

from allennlp.commands.subcommand import Subcommand
import optuna


def fetch_best_params(storage: str, study_name: str) -> Dict[str, Any]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study.best_params


def show_best_params(args: argparse.Namespace) -> None:
    best_params = fetch_best_params(args.storage, args.study_name)
    print(" ".join("{}={}".format(k, v) for k, v in best_params.items()))


@Subcommand.register("best-params")
class BestParam(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Export best hyperparameters in the trials."""
        subparser = parser.add_parser(self.name, description=description, help="Export best hyperparameters.")

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

        subparser.set_defaults(func=show_best_params)
        return subparser
