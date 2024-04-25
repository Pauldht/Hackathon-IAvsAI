from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd
import tensorflow as tf
import pickle
import sys

import torch
import torch.nn.init as init
import torch.nn as nn

import gensim.downloader

from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import torch.nn.functional as F
import nltk
nltk.download('punkt')

# -----------------------Code Utils-----------------------------------

pretrained_wv = gensim.downloader.load('glove-twitter-100')

def get_dataset():
    csv = pd.read_csv("../../../data/hack_train.csv")
    csvFull = csv.rename(columns={'text': 'answers', "label": "is_human"})
    answers_df = csvFull.drop(columns="src")
    answers_df = answers_df.explode('answers', ignore_index=True)
    answers_df = answers_df.dropna(subset=['answers'], ignore_index=True)
    return answers_df


answers_df = get_dataset()

# -----------------------Code for LLM-----------------------------------

max_features = 75000
embedding_dim = 64
sequence_length = 512*2


def vectorizer_init(answers_df):
    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        ngrams=(3, 5),
        output_mode="int",
        output_sequence_length=sequence_length,
        pad_to_max_tokens=True
    )
    vectorize_layer.adapt(answers_df['answers'])
    return vectorize_layer

#vectorize_layer = vectorizer_init(answers_df)

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

    model.load_weights("../../../model/my_model.weights.h5")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

"""
# ----------------DIFFERENT VECTORIZER--------------------
#           MUST BE CALL BEFORE ANY PREDICTION !!!!


cv = TfidfVectorizer()
cv.fit(answers_df["answers"])

pretrained_wv = gensim.downloader.load('glove-twitter-100')


# ----------------Functions to call--------------------
"""

class MultiLayerPerceptron2(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_classes):
        super(MultiLayerPerceptron2, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim) # Layer
        self.relu1 = nn.ReLU()                      # Activation
        self.dropout1 = nn.Dropout(p=0.1)           #Dropout

        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2) # Layer
        self.relu2 = nn.ReLU()# Activation
        self.dropout2 = nn.Dropout(p=0.1)#Dropout

        self.batch1 = nn.BatchNorm1d(hidden_dim//2) #Normalisation

        self.fc5 = nn.Linear(hidden_dim//2, num_classes) # Layer
        self.sigmoid = nn.Sigmoid()# Activation

    
        # weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.batch1(x)
         
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
    
def predict_fnn():
    model = MultiLayerPerceptron2(100, 128, 2) #0.87 0.87 0.87
    model.load_state_dict(torch.load("../../../model/fnn1.pth"))
    model.eval()
    return model
    
def document_vector(doc, wv):
  """Create document vectors by averaging word vectors."""
  words = word_tokenize(doc)
  word_vectors = np.array([wv[word] for word in words if word in wv])
  
  if len(word_vectors) == 0:
      return np.zeros(wv.vector_size)
  return np.mean(word_vectors, axis=0)

# -----------------------Code for FNN-----------------------------------


import xgboost as xgb

def predict_xgb(input, answers_df):
    cv = cv = TfidfVectorizer()
    cv.fit(answers_df["answers"])
    with open('../../../model/XGB_91.pickle', 'rb') as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict(cv.transform([input]))
    return prediction


def predict_llm(input):
    model = load_llm_model()
    new_answer_vectorized = vectorize_layer([input]).numpy()

    # Predict the label
    prediction = model.predict(new_answer_vectorized)
    return prediction


def predict_fnn_amy(input):
    doc_vector = document_vector(input, pretrained_wv)
    doc_vector = torch.tensor(doc_vector, dtype=torch.float32).unsqueeze(0)

    model = predict_fnn()

    with torch.no_grad():
        model.eval()
        output = model(doc_vector)

    # Appliquer softmax pour obtenir des probabilités
    probabilities = F.softmax(output, dim=1)
    proba =  max(probabilities[0][0], probabilities[0][1])
    if (proba > 0.50):
        return 1
    return 0


def predict_lr(input):
    with open('../../../model/LogisticRegression.pickle', 'rb') as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict([input])
    return prediction


def loading_model(text_input, df):
    prediction_xgb = 1#predict_xgb(text_input, df)
    prediction_llm = 1#predict_llm(text_input)
    prediction_fn = 1#predict_fnn_amy(text_input)
    predict_lr = 0#predict_lr(text_input)
    return [prediction_xgb, prediction_llm, prediction_fn, predict_lr]     


# BACKEND

from flask import Flask, jsonify
#import request

app = Flask(__name__)


@app.route('/get_prediction', methods=['GET'])
def get_data():


    #text_input = request.args.get('text_input')
    
    df = get_dataset()
    predi = loading_model("Hello this is a test", df)
    data = {'fnn': str(predi[2]), 'lr': str(predi[3]), 'xgb': str(predi[0]), 'llm': str(predi[1])}
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response

@app.route('/test', methods=['GET'])
def test_data():
    
    data = {'fnn': 1, 'lr': 1, 'xgb': 0, 'llm': 1}
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
    print("RUN SERVER")