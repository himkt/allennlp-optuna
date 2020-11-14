Installation
============

You can install allennlp-optuna by ``pip``.

.. code-block:: bash

    pip install allennlp-optuna

Then you have to create ``.allennlp_plugins``.

.. code-block:: bash

    echo 'allennlp_optuna' >> .allennlp_plugins

You can check if allennlp-optuna is successfully installed by running ``allennlp --help``.

.. code-block:: text

    usage: allennlp [-h] [--version]  ...

    Run AllenNLP

    optional arguments:
    -h, --help     show this help message and exit
    --version      show program's version number and exit

    Commands:

        best-params  Export best hyperparameters.
        evaluate     Evaluate the specified model + dataset.
        find-lr      Find a learning rate range.
        predict      Use a trained model to make predictions.
        print-results
                    Print results from allennlp serialization directories to the console.
        retrain      Train a model.
        test-install
                    Test AllenNLP installation.
        train        Train a model.
        tune         Train a model.

Can you see ``best-params``, ``retrain``, and ``tune`` in the help?
If so, congratulations! You have installed allennlp-optuna.
