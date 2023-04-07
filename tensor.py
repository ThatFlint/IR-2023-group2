import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)


# FeaturesDict({
#     'idx': tf.int32,
#     'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=2),
#     'sentence1': Text(shape=(), dtype=tf.string),
#     'sentence2': Text(shape=(), dtype=tf.string),
# })

class BertInputProcessor(tf.keras.layers.Layer):
    def __init__(self, tokenizer, packer):
        super().__init__()
        self.tokenizer = tokenizer
        self.packer = packer

    def call(self, inputs):
        token = self.tokenizer(inputs['query'])
        packed = self.packer(token)

        if 'label' in inputs:
            return packed, inputs['isChild']
        else:
            return packed


tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    lower_case=True)
