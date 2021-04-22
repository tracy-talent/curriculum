import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import random
import numpy as np


class BILSTM_CRF(tf.keras.Model):
    
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim, 
                 vocab_size_char, 
                 vocab_size_bio):
        super(MyModel, self).__init__()
        
        self.char_embedding = layers.Embedding(vocab_size_char, embedding_dim)
        self.bilstm = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=True), merge_mode='sum')
        self.projection = layers.Dense(vocab_size_bio)
        self.transition_params = tf.keras.initializers.GlorotUniform()(shape=(vocab_size_bio, vocab_size_bio))

    def call(self, inputs_seq, inputs_seq_len, output_seq):
        inputs_embed = self.char_embedding(inputs_seq) # B, S, D
        rnn_outputs = self.bilstm(inputs_embed) # B, S, H
        logits_seq = self.projection(rnn_outputs) # B, S, L 
        
        return logits_seq

    
