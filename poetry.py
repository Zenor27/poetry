#! /usr/bin/env python3

import os, logging
# Remove all the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.disable(logging.WARNING)

import tensorflow as tf
import numpy as np
import sys

tf.enable_eager_execution()

# Some global variables

FILENAME = 'baudelaire.txt' 
# We can modify values ofc
SEQ_LENGTH = 30
BATCH_SZ = 64
BUFFER_SZ = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024
CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")
EPOCHS=30


if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools
    rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')


def get_text_file():
    with open(FILENAME, 'r', encoding='ISO-8859-1') as f:
        return f.read()


def get_vocab(text):
    return sorted(set(text))


def get_char_2_idx(vocab):
    char2idx = {u:i for i, u in enumerate(vocab)}
    return char2idx


def get_idx_2_char(vocab):
    return np.array(vocab)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_sz, batch_sz=BATCH_SZ):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_sz, EMBEDDING_DIM,
                                  batch_input_shape=[batch_sz, None]),
        rnn(RNN_UNITS,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_sz)
    ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def train(vocab_sz, steps_per_epoch):
    model = build_model(vocab_sz)
    model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PREFIX, save_weights_only=True)

    history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

    print(model.summary())


def display(vocab_sz, char2idx, start_string, idx2char):
    tf.train.latest_checkpoint(CHECKPOINT_DIR)
    model = build_model(vocab_sz, batch_sz=1)
    model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    model.build(tf.TensorShape([1, None]))

    def generate_text(start_string):
        num_generate = 1000

        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more WTF text.
        # Experiment to find the best setting.
        temperature = 0.7

        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))

    return generate_text(start_string)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('An option is required, -t to train the NN, -d to display a text from the NN')
        sys.exit(1)

    text = get_text_file()
    vocab = get_vocab(text)

    char2idx = get_char_2_idx(vocab)
    idx2char = get_idx_2_char(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    examples_per_epoch = len(text) // SEQ_LENGTH

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    steps_per_epoch = examples_per_epoch // BATCH_SZ

    dataset = dataset.shuffle(BUFFER_SZ).batch(BATCH_SZ, drop_remainder=True)

    vocab_sz = len(vocab)

    if sys.argv[1] == '-t':
        train(vocab_sz, steps_per_epoch)
    elif sys.argv[1] == '-d':
        print(display(vocab_sz, char2idx, u"Poesie", idx2char))
    else:
        print('unknown option, -t to train the NN, -d to display a text from the NN') 
