
## Install

```
poetry install
```

## Run

```
poetry run allennlp allenopt \
    config/imdb_optuna.jsonnet \
    config/hparams.json \
    --optuna-config config/optuna.json  # settings for Optuna
    --serialization-dir out
```


## AllenNLP config

Model configuration written in Jsonnet.

You have to replace values of hyperparameters with jsonnet function `std.extVar`.
Remember casting external variables to desired types by `std.parseInt`, `std.parseJson`.

```jsonnet
local lr = 0.1;  // before

local lr = std.parseJson(std.extVar('lr'));  // after
```

For more information, please refer to [AllenNLP Guide](https://guide.allennlp.org/hyperparameter-optimization).


## Hyperparameter search space

You can define search space in Json.

Each hyperparameter config must have `type` and `keyword`.
You can see what parameters are available for each hyperparameter in [Optuna API reference](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial).

Parameters for `suggest_#{type}` are available for config of `type=#{type}`. (e.g. when `type=float`, you can see the available parameters in [suggest_float](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float)

Please see the [example](./config/hparams.json) in detail.


## Optuna config

You can choose pruners and samplers implemented in Optuna here.

The [example](./config/optuna.json) shows the usage of allenopt with `Hyperband` [1] and `TPESampler` [2].


## Reference

[1] Li+, [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://jmlr.org/papers/v18/16-558.html), 2018, JMLR.

[2] Bergstra+, [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization), 2011, Proceedings of NIPS.
