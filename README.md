# AllenOpt: AllenNLP subcommand for hyperparameter optimization


## 0. Install

```
pip install allenopt @ https://github.com/himkt/allenopt.git@master
```


## 1. Optimization


### 1.1 AllenNLP config

Model configuration written in Jsonnet.

You have to replace values of hyperparameters with jsonnet function `std.extVar`.
Remember casting external variables to desired types by `std.parseInt`, `std.parseJson`.

```jsonnet
local lr = 0.1;  // before
↓↓↓
local lr = std.parseJson(std.extVar('lr'));  // after
```

For more information, please refer to [AllenNLP Guide](https://guide.allennlp.org/hyperparameter-optimization).


### 1.2 Define hyperparameter search speaces

You can define search space in Json.

Each hyperparameter config must have `type` and `keyword`.
You can see what parameters are available for each hyperparameter in
[Optuna API reference](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial).

```json
[
  {
    "type": "int",
    "attributes": {
      "name": "embedding_dim",
      "low": 64,
      "high": 128
    }
  },
  {
    "type": "int",
    "attributes": {
      "name": "max_filter_size",
      "low": 2,
      "high": 5
    }
  },
  {
    "type": "int",
    "attributes": {
      "name": "num_filters",
      "low": 64,
      "high": 256
    }
  },
  {
    "type": "int",
    "attributes": {
      "name": "output_dim",
      "low": 64,
      "high": 256
    }
  },
  {
    "type": "float",
    "attributes": {
      "name": "dropout",
      "low": 0.0,
      "high": 0.5
    }
  },
  {
    "type": "float",
    "attributes": {
      "name": "lr",
      "low": 5e-3,
      "high": 5e-1,
      "log": true
    }
  }
]
```

Parameters for `suggest_#{type}` are available for config of `type=#{type}`. (e.g. when `type=float`,
you can see the available parameters in [suggest\_float](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float)

Please see the [example](./config/hparams.json) in detail.


### 1.3 [Optional] Specify Optuna configurations

You can choose a pruner/sample implemented in Optuna.
To specify a pruner/sampler, create a JSON config file

The example of [optuna.json](./config/optuna.json) looks like:

```json
{
    "pruner": {
        "type": "HyperbandPruner",
        "attributes": {
            "min_resource": 1,
            "reduction_factor": 5
        }
    },
    "sampler": {
        "type": "TPESampler",
        "attributes": {
            "n_startup_trials": 5
        }
    }
}
```


### 1.4 Optimize hyperparameters by allennlp cli


```shell
poetry run allennlp tune \
    config/imdb_optuna.jsonnet \
    config/hparams.json \
    --optuna-config config/optuna.json \
    --serialization-dir result \
    --study-name test
```


## 2. Get best hyperparameters

```shell
poetry run allennlp best-params \
    --study-name test
```


## 3. Retrain a model with optimized hyperparameters

```shell
poetry run allennlp retrain \
    config/imdb_optuna.jsonnet \
    --serialization-dir retrain_result \
    --study-name test
```


## 4. Hyperparameter optimization at scale!

you can run optimizations in parallel.
You can easily run distributed optimization by adding an option
`--skip-if-exists` to `allennlp tune` command.

```
poetry run allennlp tune \
    config/imdb_optuna.jsonnet \
    config/hparams.json \
    --optuna-config config/optuna.json \
    --serialization-dir result \
    --study-name test \
    --skip-if-exists
```

AllenOpt uses SQLite as a default storage for storing results.
You can easily run distributed optimization **over machines**
by using MySQL or PostgreSQL as a storage.

For example, if you want to use MySQL as a storage,
the command should be like following:

```
poetry run allennlp tune \
    config/imdb_optuna.jsonnet \
    config/hparams.json \
    --optuna-config config/optuna.json \
    --serialization-dir result \
    --study-name test \
    --storage mysql://<user_name>:<passwd>@<db_host>/<db_name> \
    --skip-if-exists
```

You can run the above command on each machine to
run multi-node distributed optimization.

If you want to know about a mechanism of Optuna distributed optimization,
please see the official documentation:
https://optuna.readthedocs.io/en/stable/tutorial/004_distributed.html
