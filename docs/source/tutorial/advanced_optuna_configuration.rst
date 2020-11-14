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


Next, we have to add `optuna_pruner` to `epoch_callbacks`.

- ``imdb_optuna_with_pruning.jsonnet``

.. code-block:: text

  local batch_size = 64;
  local cuda_device = 0;
  local num_epochs = 15;
  local seed = 42;

  local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
  local dropout = std.parseJson(std.extVar('dropout'));
  local lr = std.parseJson(std.extVar('lr'));
  local max_filter_size = std.parseInt(std.extVar('max_filter_size'));
  local num_filters = std.parseInt(std.extVar('num_filters'));
  local output_dim = std.parseInt(std.extVar('output_dim'));
  local ngram_filter_sizes = std.range(2, max_filter_size);

  {
    numpy_seed: seed,
    pytorch_seed: seed,
    random_seed: seed,
    dataset_reader: {
      lazy: false,
      type: 'text_classification_json',
      tokenizer: {
        type: 'spacy',
      },
      token_indexers: {
        tokens: {
          type: 'single_id',
          lowercase_tokens: true,
        },
      },
    },
    train_data_path: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl',
    validation_data_path: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl',
    model: {
      type: 'basic_classifier',
      text_field_embedder: {
        token_embedders: {
          tokens: {
            embedding_dim: embedding_dim,
          },
        },
      },
      seq2vec_encoder: {
        type: 'cnn',
        embedding_dim: embedding_dim,
        ngram_filter_sizes: ngram_filter_sizes,
        num_filters: num_filters,
        output_dim: output_dim,
      },
      dropout: dropout,
    },
    data_loader: {
      shuffle: true,
      batch_size: batch_size,
    },
    trainer: {
      cuda_device: cuda_device,
      epoch_callbacks: [
        {
          type: 'optuna_pruner',
        }
      ],
      num_epochs: num_epochs,
      optimizer: {
        lr: lr,
        type: 'sgd',
      },
      validation_metric: '+accuracy',
    },
  }


Finally, you can run optimization with pruning:

.. code-block:: bash

    poetry run allennlp tune \
        config/imdb_optuna_with_pruning.jsonnet \
        config/hparams.json \
        --optuna-param-path config/optuna.json \
        --serialization-dir result/hpo \
        --study-name test-with-pruning
