AllenNLP configuration
======================


Original configuration
----------------------

Here is the example of AllenNLP configuration.

- ``imdb.jsonnet``

.. code-block:: text

  local batch_size = 64;
  local cuda_device = -1;
  local num_epochs = 15;
  local seed = 42;


  local embedding_dim = 32;
  local dropout = 0.5;
  local lr = 1e-3;
  local max_filter_size = 5;
  local num_filters = 32;
  local output_dim = 64;
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
    datasets_for_vocab_creation: ['train'],
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
      num_epochs: num_epochs,
      optimizer: {
        lr: lr,
        type: 'sgd',
      },
      validation_metric: '+accuracy',
    },
  }



Setup for allennlp-optuna
-------------------------

You have to change values of hyperparameter that you want to optimize.
For example, if you want to optimize a dimensionality of word embedding, a change should be following:

.. code-block:: diff

    < local embedding_dim = 32;
    ---
    > local embedding_dim = std.parseInt(std.extVar('embedding_dim'));

The sample configuration looks like following:

- ``imdb_optuna.jsonnet``

.. code-block:: text

  local batch_size = 64;
  local cuda_device = -1;
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
    datasets_for_vocab_creation: ['train'],
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
      num_epochs: num_epochs,
      optimizer: {
        lr: lr,
        type: 'sgd',
      },
      validation_metric: '+accuracy',
    },
  }

Well done, you have completed the setup AllenNLP configuration for allennlp-optuna.
