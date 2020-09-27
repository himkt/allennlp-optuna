"""
The `train` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.
"""

import argparse
import json
import logging
import os.path
from functools import partial

import optuna
from allennlp.commands.subcommand import Subcommand
from optuna import Trial
from optuna.integration import AllenNLPExecutor
from overrides import overrides


logger = logging.getLogger(__name__)


def optimize_hyperparameters(args: argparse.Namespace) -> None:
    config_file = args.param_path
    hparam_path = args.hparam_path
    serialization_dir = args.serialization_dir

    def _objective(
        trial: Trial,
        hparam_path: str,
    ) -> float:

        for hparam in json.load(open(hparam_path)):
            attr_type = hparam["type"]
            suggest = getattr(trial, "suggest_{}".format(attr_type))
            suggest(**hparam["keyword"])

        optuna_serialization_dir = os.path.join(serialization_dir, "trial_{}".format(trial.number))
        executor = AllenNLPExecutor(trial, config_file, optuna_serialization_dir)
        return executor.run()

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///allennlp.db",
        pruner=optuna.pruners.HyperbandPruner(),
    )

    objective = partial(
        _objective,
        hparam_path=hparam_path,
    )
    study.optimize(objective, n_trials=50, timeout=600)


@Subcommand.register("allenopt")
class AllenOpt(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(self.name, description=description, help="Train a model.")

        subparser.add_argument(
            "param_path",
            type=str,
            help="path to parameter file describing the model to be trained",
        )

        subparser.add_argument(
            "hparam_path",
            type=str,
            help="path to hyperparameter file",
            default="hyper_params.json",
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        subparser.set_defaults(func=optimize_hyperparameters)
        return subparser
