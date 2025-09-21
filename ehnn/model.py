from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model as KModel

class TimeAttention(layers.Layer):
    def __init__(self):
        super().__init__()
        self.attn = layers.Dense(1, activation="tanh")

    def call(self, inputs, **kwargs):
        w = tf.nn.softmax(self.attn(inputs), axis=1)
        return tf.reduce_sum(w * inputs, axis=1)

class EHNNHybrid(KModel):
    def __init__(self, num_norm: layers.LayerNormalization):
        super().__init__()
        self.cnn_xor = tf.keras.Sequential(
            [
                layers.Conv1D(64, 3, activation="relu", padding="same"),
                layers.MaxPooling1D(2),
                layers.Conv1D(128, 3, activation="relu", padding="same"),
                TimeAttention(),
            ]
        )
        self.cnn_kmer = tf.keras.Sequential(
            [
                layers.Conv1D(64, 3, activation="relu", padding="same"),
                layers.MaxPooling1D(2),
                layers.Conv1D(128, 3, activation="relu", padding="same"),
                TimeAttention(),
            ]
        )
        self.lstm = tf.keras.Sequential(
            [
                layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
                layers.GlobalMaxPooling1D(),
            ]
        )
        self.num_mlp = tf.keras.Sequential(
            [
                num_norm,
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
            ]
        )
        self.concat = layers.Concatenate()
        self.head = tf.keras.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def call(self, inputs, **kwargs):
        x_xor, x_kmer, x_num = inputs
        c1 = self.cnn_xor(x_xor)
        c2 = self.cnn_kmer(x_kmer)
        c = self.concat([c1, c2])
        lstm_out = self.lstm(tf.expand_dims(c, axis=1))
        num_out = self.num_mlp(x_num)
        fused = self.concat([c1, c2, lstm_out, num_out])
        return self.head(fused)
