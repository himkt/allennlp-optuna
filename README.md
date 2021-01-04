# AllenNLP subcommand for hyperparameter optimization

`allennlp-optuna` is [AllenNLP](https://github.com/allenai/allennlp) plugin for
hyperparameter optimization using [Optuna](https://github.com/optuna/optuna).


#### Supported environments

Machine \ Device | Single GPU             | Multi GPUs
---------------- | ---------------------- | ---------------
Single Node      | :white_check_mark:     | Partial
Multi Nodes      | :white_check_mark:     | Partial

AllenNLP provides a way of distributed training (https://medium.com/ai2-blog/c4d7c17eb6d6).
Unfortunately, `allennlp-optuna` doesn't fully support this feature.
With multiple GPUs, you can run hyperparameter optimization.
But you cannot enable a pruning feature.
(For more detail, please see [himkt/allennlp-optuna#20](https://github.com/himkt/allennlp-optuna/issues/20)
and [optuna/optuna#1990](https://github.com/optuna/optuna/issues/1990))

Alternatively, `allennlp-optuna` supports distributed optimization with multiple machines.
Please read the [tutorial](https://allennlp-optuna.readthedocs.io/en/latest/tutorial/hyperparameter_optimization_at_scale.html) about
distributed optimization in `allennlp-optuna`.
You can also learn about a mechanism of Optuna in the [paper](https://arxiv.org/pdf/1907.10902.pdf)
or [documentation](https://optuna.readthedocs.io/en/stable/).


#### Documentation

You can read the documentation on [readthedocs](https://allennlp-optuna.readthedocs.io/).


## 1. Installation

```sh
pip install allennlp_optuna

# Create .allennlp_plugins at the top of your repository or $HOME/.allennlp/plugins
# For more information, please see https://github.com/allenai/allennlp#plugins
echo 'allennlp_optuna' >> .allennlp_plugins
```


## 2. Optimization


### 2.1. AllenNLP config

Model configuration written in Jsonnet.

You have to replace values of hyperparameters with jsonnet function `std.extVar`.
Remember casting external variables to desired types by `std.parseInt`, `std.parseJson`.

```jsonnet
local lr = 0.1;  // before
↓↓↓
local lr = std.parseJson(std.extVar('lr'));  // after
```

For more information, please refer to [AllenNLP Guide](https://guide.allennlp.org/hyperparameter-optimization).


### 2.2. Define hyperparameter search speaces

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


### 2.3. Optimize hyperparameters by allennlp cli


```shell
allennlp tune \
    config/imdb_optuna.jsonnet \
    config/hparams.json \
    --serialization-dir result/hpo \
    --study-name test
```


### 2.4. [Optional] Specify Optuna configurations

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

And add a epoch callback to your configuration.
(https://guide.allennlp.org/hyperparameter-optimization#6)

```
  epoch_callbacks: [
    {
      type: 'optuna_pruner',
    }
  ],
```

- [`config/imdb_optuna.jsonnet`](./config/imdb_optuna.jsonnet) is a simple configuration for allennlp-optuna
- [`config/imdb_optuna_with_pruning.jsonnet`](./config/imdb_optuna_with_pruning.jsonnet) is a configuration using Optuna pruner (and TPEsampler)

```sh
$ diff config/imdb_optuna.jsonnet config/imdb_optuna_with_pruning.jsonnet
32d31
<   datasets_for_vocab_creation: ['train'],
58a58,62
>     epoch_callbacks: [
>       {
>         type: 'optuna_pruner',
>       }
>     ],
```

Then, you can use a pruning callback by running following:

```shell
allennlp tune \
    config/imdb_optuna_with_pruning.jsonnet \
    config/hparams.json \
    --optuna-param-path config/optuna.json \
    --serialization-dir result/hpo_with_optuna_config \
    --study-name test_with_pruning
```



## 3. Get best hyperparameters

```shell
allennlp best-params \
    --study-name test
```


## 4. Retrain a model with optimized hyperparameters

```shell
allennlp retrain \
    config/imdb_optuna.jsonnet \
    --serialization-dir retrain_result \
    --study-name test
```


## 5. Hyperparameter optimization at scale!

you can run optimizations in parallel.
You can easily run distributed optimization by adding an option
`--skip-if-exists` to `allennlp tune` command.

```
allennlp tune \
    config/imdb_optuna.jsonnet \
    config/hparams.json \
    --optuna-param-path config/optuna.json \
    --serialization-dir result \
    --study-name test \
    --skip-if-exists
```

allennlp-optuna uses SQLite as a default storage for storing results.
You can easily run distributed optimization **over machines**
by using MySQL or PostgreSQL as a storage.

For example, if you want to use MySQL as a storage,
the command should be like following:

```
allennlp tune \
    config/imdb_optuna.jsonnet \
    config/hparams.json \
    --optuna-param-path config/optuna.json \
    --serialization-dir result \
    --study-name test \
    --storage mysql://<user_name>:<passwd>@<db_host>/<db_name> \
    --skip-if-exists
```

You can run the above command on each machine to
run multi-node distributed optimization.

If you want to know about a mechanism of Optuna distributed optimization,
please see the official documentation: https://optuna.readthedocs.io/en/latest/tutorial/10_key_features/004_distributed.html


#### Reference

- Cookpad Techlife (in Japanese): https://techlife.cookpad.com/entry/2020/11/06/110000
  - `allennlp-optuna` is used for optimizing hyperparameter of NER model
