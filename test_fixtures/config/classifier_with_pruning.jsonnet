local dropout = std.parseJson(std.extVar('dropout'));
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local lr = std.parseJson(std.extVar('lr'));


{
  dataset_reader: {
    type: 'text_classification_json',
    token_indexers: {
      tokens: {
        type: 'single_id'
      },
    },
  },
  train_data_path: 'test_fixtures/data/train.jsonl',
  validation_data_path: 'test_fixtures/data/valid.jsonl',
  model: {
    type: 'basic_classifier',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: embedding_dim,
        },
      },
    },
    seq2vec_encoder: {
      type: 'lstm',
      input_size: embedding_dim,
      hidden_size: 5,
      dropout: dropout,
    }
  },
  data_loader: {
    batch_size: 1,
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    num_epochs: 5,
    callbacks: [
      { type: 'optuna_pruner' },
    ]
  },
}
