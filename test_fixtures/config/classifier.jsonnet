local dropout = std.parseJson(std.extVar('dropout'));
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local lr = std.parseJson(std.extVar('lr'));


{
  dataset_reader: {
    type: 'sequence_tagging',
    word_tag_delimiter: '/',
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      },
      token_characters: {
        type: 'characters',
      },
    },
  },
  train_data_path: 'test_fixtures/data/sentences.train',
  validation_data_path: 'test_fixtures/data/sentences.valid',
  model: {
    type: 'simple_tagger',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: embedding_dim,
        },
        token_characters: {
          type: 'character_encoding',
          embedding: {
            embedding_dim: embedding_dim,
          },
          encoder: {
            type: 'cnn',
            embedding_dim: embedding_dim,
            num_filters: 5,
            ngram_filter_sizes: [3],
          },
          dropout: dropout,
        },
      },
    },
    encoder: {
      type: 'lstm',
      input_size: embedding_dim + 5,
      hidden_size: 10,
      num_layers: 2,
      dropout: 0,
      bidirectional: true,
    },
  },
  data_loader: {
    batch_size: 32,
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    num_epochs: 1,
    patience: 10,
    cuda_device: -1,
  },
}
