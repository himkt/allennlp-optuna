Optimize hyperparameters by allennlp cli
========================================


Optimize
--------

You can optimize hyperparameters by:

.. code-block:: bash

    poetry run allennlp tune \
        config/imdb_optuna.jsonnet \
        config/hparams.json \
        --optuna-param-path config/optuna.json \
        --serialization-dir result \
        --study-name test


Get best hyperparameters
------------------------

.. code-block:: bash

    poetry run allennlp best-params \
        --study-name test


Retrain a model with optimized hyperparameters
----------------------------------------------

.. code-block:: bash

    poetry run allennlp retrain \
        config/imdb_optuna.jsonnet \
        --serialization-dir retrain_result \
        --study-name test
