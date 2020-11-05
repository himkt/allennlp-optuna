Hyperparameter optimization at scale!
=====================================

you can run optimizations in parallel.
You can easily run distributed optimization by adding an option
`--skip-if-exists` to `allennlp tune` command.


.. code-block:: bash

    poetry run allennlp tune \
        config/imdb_optuna.jsonnet \
        config/hparams.json \
        --optuna-param-path config/optuna.json \
        --serialization-dir result \
        --study-name test \
        --skip-if-exists

allennlp-optuna uses SQLite as a default storage for storing results.
You can easily run distributed optimization **over machines**
by using MySQL or PostgreSQL as a storage.

For example, if you want to use MySQL as a storage,
the command should be like following:

.. code-block:: bash

    poetry run allennlp tune \
        config/imdb_optuna.jsonnet \
        config/hparams.json \
        --optuna-param-path config/optuna.json \
        --serialization-dir result \
        --study-name test \
        --storage mysql://<user_name>:<passwd>@<db_host>/<db_name> \
        --skip-if-exists

You can run the above command on each machine to
run multi-node distributed optimization.

If you want to know about a mechanism of Optuna distributed optimization,
please see the official documentation:
https://optuna.readthedocs.io/en/stable/tutorial/004_distributed.html
