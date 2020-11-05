Advanced configuration for Optuna
=================================

You can choose a pruner/sample implemented in Optuna.
To specify a pruner/sampler, create a JSON config file

The example of `optuna.json <./config/optuna.json>`_ looks like:


- ``optuna.json``

.. code-block:: json

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
