from tensorflow.keras import Model, Input, layers
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout

import re
import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text



# -----------------------Code for LLM-----------------------------------


max_features = 75000
embedding_dim = 64
sequence_length = 512*2


def vectorizer_init():
    csv = pd.read_csv("/kaggle/input/hack-train/hack_train.csv")
    csvFull = csv.rename(columns={'text': 'answers', "label": "is_human"})
    answers_df = csvFull.drop(columns="src")
    # Checking the len
    answers_df = answers_df.explode('answers', ignore_index=True)
    answers_df = answers_df.dropna(subset=['answers'], ignore_index=True)

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_features,
        ngrams=(3, 5),
        output_mode="int",
        output_sequence_length=sequence_length,
        pad_to_max_tokens=True
    )
    vectorize_layer.adapt(answers_df['answers'])
    return vectorize_layer


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config["att"] = self.att
        config["ffn"] = self.ffn
        config["layernorm1"] = self.layernorm1
        config["layernorm2"] = self.layernorm2
        config["dropout1"] = self.dropout1
        config["dropout2"] = self.dropout2
        return config


def load_llm_model():
    inputs = Input(shape=(sequence_length,), dtype="int64")
    x = Embedding(max_features, embedding_dim)(inputs)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    transformer_block = TransformerBlock(embedding_dim, 2, 32)
    x = transformer_block(x, training=True)
    x = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.summary()

    model.load_weights("../../model/my_model.weights.h5")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


vectorize_layer = vectorizer_init()


# ----------------Functions to call in j---------------------


def call_llm(input, vectorize_layer=vectorize_layer):
    model = load_llm_model()
    new_answer_vectorized = vectorize_layer([input]).numpy()  # TODO: add input answer

    # Predict the label
    prediction = model.predict(new_answer_vectorized)
    return prediction