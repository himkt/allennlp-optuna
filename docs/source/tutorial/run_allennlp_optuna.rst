Optimize hyperparameters by allennlp cli
========================================


Optimize
--------

You can optimize hyperparameters by:

.. code-block:: bash

    allennlp tune \
        imdb_optuna.jsonnet \
        hparams.json \
        --serialization-dir result \
        --study-name test


Get best hyperparameters
------------------------

.. code-block:: bash

    allennlp best-params \
        --study-name test


Retrain a model with optimized hyperparameters
----------------------------------------------

.. code-block:: bash

    allennlp retrain \
        imdb_optuna.jsonnet \
        --serialization-dir retrain_result \
        --study-name test
