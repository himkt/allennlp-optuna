import os.path
import subprocess
import tempfile


def test_tune():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = "sqlite:///" + os.path.join(tmpdir, "allennlp_optuna.db")
        command = [
            "allennlp",
            "tune",
            "test_fixtures/config/classifier.jsonnet",
            "test_fixtures/config/hparams.json",
            "--serialization-dir",
            tmpdir,
            "--storage",
            storage,
            "--n-trials",
            "3",
        ]
        subprocess.check_call(command)


def test_tune_with_pruner():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = "sqlite:///" + os.path.join(tmpdir, "allennlp_optuna.db")
        command = [
            "allennlp",
            "tune",
            "test_fixtures/config/classifier_with_pruning.jsonnet",
            "test_fixtures/config/hparams.json",
            "--optuna-param-path",
            "test_fixtures/config/optuna.json",
            "--serialization-dir",
            tmpdir,
            "--storage",
            storage,
            "--n-trials",
            "3",
        ]
        subprocess.check_call(command)


def test_tune_with_pruner_without_attribute():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = "sqlite:///" + os.path.join(tmpdir, "allennlp_optuna.db")
        command = [
            "allennlp",
            "tune",
            "test_fixtures/config/classifier_with_pruning.jsonnet",
            "test_fixtures/config/hparams.json",
            "--optuna-param-path",
            "test_fixtures/config/optuna_without_attribute.json",
            "--serialization-dir",
            tmpdir,
            "--storage",
            storage,
            "--n-trials",
            "3",
        ]
        subprocess.check_call(command)
