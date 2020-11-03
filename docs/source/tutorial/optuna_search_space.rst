Defining search space
=====================


Next, it's time to define a search space for hyperparameter.
A search space is represented as JSON element.
For example, a search space for embedding dimensionality looks like following:

.. code-block:: json

  {
    "type": "int",
    "attributes": {
      "name": "embedding_dim",
      "low": 64,
      "high": 128
    }
  }

``type`` should be ``int``, ``float``, or ``categorical``.
``attributes`` is arguments that Optuna takes.
``name`` is a name of hyperparameter.
``low`` and ``high`` are the range of a parameter.
For categorical distribution, ``choices`` is available.
For more information about ``attributes``, please see the `Optuna API reference <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial>`_
(``suggest_float``, ``suggest_int``, and ``suggest_categorical``).

The entire example of AllenNLP configuration for allennlp-optuna is following:

- ``hparams.json``

.. code-block:: json

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
