"""
Title: Automatic Speech Recognition with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2021/01/13
Last modified: 2021/01/13
Description: Implement a complete transformer and train it on sequence to sequence ASR.
"""
"""
## Introduction

Automatic speech recognition (ASR) consists of transcribing audio segments into text.
A popular way to handle this task is to treat it as a sequence-to-sequence
problem. The audio can be represented as a sequence of feature vectors,
and the text can be represented as a sequence of characters, words, or subword tokens.

ASR models typically require huge datasets and take a lot of time to train.
For this demonstration, we will use a simple dataset so as to reduce training time yet
display interesting results. Our model will be a Transformer (both encoder and decoder)
as proposed in the paper, "Attention is All You Need".

Our data consists of audio segments with a person speaking out one of the ten digits
(0-9). We convert the digits as a sequence of characters (e.g. 9 -> n,i,n,e) to form
a small dataset. The method shown below, however, can be applied to a real world
speech dataset, to learn the mapping from audio to a sequence of characters.

**References:**
- [Attention is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
"""
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


"""
## Transformer Input Layer

This layer computes the sum of position embeddings and feature embeddings to feed to
the transformer layers.
"""


class TransformerInput(layers.Layer):
    def __init__(
        self, input_type="tokens", nvocab=1000, nhid=64, nff=128, maxlen=100,
    ):
        super().__init__()
        self.input_type = input_type
        if input_type == "tokens":
            self.emb = tf.keras.layers.Embedding(nvocab, nhid)
        elif input_type == "feats":
            self.emb = tf.keras.layers.Dense(nhid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=nhid)

    def call(self, x):
        if self.input_type == "tokens":
            maxlen = tf.shape(x)[-1]
        elif self.input_type == "feats":
            maxlen = tf.shape(x)[1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


"""
## Transformer Encoder Layer
"""


class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Transformer Decoder Layer
"""


class TransformerDecoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.drop = layers.Dropout(rate)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Mask the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, trg):
        input_shape = tf.shape(trg)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        trg_att = self.self_att(trg, trg, attention_mask=causal_mask)
        trg_norm = self.ln1(trg + self.drop(trg_att))
        enc_out = self.enc_att(trg_norm, enc_out)
        enc_out_norm = self.ln2(self.drop(enc_out) + trg_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.ln3(enc_out_norm + self.drop(ffn_out))
        return ffn_out_norm


"""
## Complete Transformer Model
"""

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def masked_loss(real, pred):
    """ assuming pad token index = 0 """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


class Transformer(keras.Model):
    def __init__(
        self,
        input_type="tokens",
        nvocab=1000,
        ninp=32,
        nhid=64,
        nhead=2,
        nff=128,
        src_maxlen=100,
        trg_maxlen=100,
        nlayers=2,
        nclasses=10,
    ):
        super().__init__()
        self.nlayers = nlayers
        self.trg_maxlen = trg_maxlen
        self.input_type = input_type

        self.enc_input = TransformerInput(
            input_type=input_type, nvocab=1000, nhid=64, maxlen=src_maxlen
        )
        self.dec_input = TransformerInput(
            input_type="tokens", nvocab=nclasses, nhid=nhid, maxlen=trg_maxlen
        )
        for i in range(nlayers):
            setattr(self, f"enc_layer_{i}", TransformerEncoderLayer(nhid, nhead, nff))
            setattr(self, f"dec_layer_{i}", TransformerDecoderLayer(nhid, nhead, nff))

        self.final = layers.Dense(nclasses)

    def encode_src(self, src):
        x = self.enc_input(src)
        for i in range(self.nlayers):
            x = getattr(self, f"enc_layer_{i}")(x)
        return x

    def decode(self, enc_out, trg):
        y = self.dec_input(trg)
        for i in range(self.nlayers):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        src = inputs[0]
        trg = inputs[1]
        x = self.encode_src(src)
        y = self.decode(x, trg)
        return self.final(y)

    def train_step(self, batch):
        """ Process one batch inside model.fit() """
        src = batch["src"]
        trg = batch["trg"]
        dec_inp = trg[:, :-1]
        dec_trg = trg[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([src, dec_inp])
            loss = masked_loss(dec_trg, preds)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss}

    def generate(self, src, trg_start_token_idx):
        """ Return a batch of predicted token indices with greedy deecoding """
        bs = tf.shape(src)[0]
        dec_inp = tf.ones((bs, self.trg_maxlen), dtype=tf.int32) * trg_start_token_idx
        for i in range(self.trg_maxlen):
            preds = self([src, dec_inp])
            pred_idx = tf.argmax(preds, axis=-1, output_type=tf.int32)
            current_pred = tf.expand_dims(pred_idx[:, i], axis=-1)
            if i < self.trg_maxlen - 1:
                future_pad = tf.ones((bs, self.trg_maxlen - (i + 2)), dtype=tf.int32)
                dec_inp = tf.concat(
                    [dec_inp[:, : i + 1], current_pred, future_pad], axis=-1
                )
            else:
                dec_inp = tf.concat([dec_inp[:, : i + 1], current_pred], axis=-1)
        return pred_idx


"""
## Preprocess the ASR data

Due to the nature of our data, we already know that the vocabulary for our target characters
is 'a'-'z'. We add 3 extra tokens to this vocabulary:

- `'<'`: start token
- `'>'`: end token
- `'-'`: pad token

We tokenize the text labels with the above vocabulary.
For the audio, we compute Log Mel-spectrograms from the wav file using signal
processing functions present in TensorFlow. Log Mel-spectrogram is a popular
feature preprocessing setup in ASR experiments.
"""


def char_to_idx(ch):
    if ch == "<":
        return 27  # start token
    if ch == ">":
        return 28  # end token
    if ch == "-":
        return 0  # pad token
    return ord(ch) - 96  # a->1, b->2, etc


def filename_to_label(f):
    m = {
        "0": [char_to_idx(ch) for ch in "<zero>-"],
        "1": [char_to_idx(ch) for ch in "<one>--"],
        "2": [char_to_idx(ch) for ch in "<two>--"],
        "3": [char_to_idx(ch) for ch in "<three>"],
        "4": [char_to_idx(ch) for ch in "<four>-"],
        "5": [char_to_idx(ch) for ch in "<five>-"],
        "6": [char_to_idx(ch) for ch in "<six>--"],
        "7": [char_to_idx(ch) for ch in "<seven>"],
        "8": [char_to_idx(ch) for ch in "<eight>"],
        "9": [char_to_idx(ch) for ch in "<nine>-"],
    }
    return m[f.split("/")[-1][0]]


def fpath_to_logmelspec(f):
    sample_rate = 8000
    audio = tf.io.read_file(f)
    audio, _ = tf.audio.decode_wav(audio, 1, sample_rate)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.expand_dims(audio, axis=0)
    stfts = tf.signal.stft(
        audio, frame_length=1024, frame_step=256, fft_length=1024
    )  # A 1024-point STFT with frames of 64 ms and 75% overlap.
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 2000.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
    )
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return tf.squeeze(log_mel_spectrograms, axis=0)


def create_dataset(flist, bs=4):
    label_data = [filename_to_label(f) for f in flist]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(fpath_to_logmelspec)
    label_ds = tf.data.Dataset.from_tensor_slices(label_data)
    ds = tf.data.Dataset.zip((audio_ds, label_ds))
    ds = ds.map(lambda x, y: {"src": x, "trg": y})
    return ds.batch(bs)


"""
## Create a Dataset object
"""

data_path = keras.utils.get_file(
    "spoken_digit.tar.gz",
    "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.9.tar.gz",
)
command = f"tar -xvf {data_path} --directory ."
os.system(command)

root = "free-spoken-digit-dataset-1.0.9/recordings"
flist = [os.path.join(root, x) for x in os.listdir(root)]
random.shuffle(flist)
train_list = flist[:1800]
val_list = flist[1800:]
ds = create_dataset(train_list)
val_ds = create_dataset(val_list)

"""
## Callback to display predictions
"""


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch, trg_start_token_idx=27, trg_end_token_idx=28):
        self.batch = batch
        self.trg_start_token_idx = trg_start_token_idx
        self.idx_to_char = ["-"] + [chr(i + 96) for i in range(1, 27)] + ["<", ">"]

    def on_epoch_end(self, epoch, logs=None):
        src = self.batch["src"]
        trg = self.batch["trg"].numpy()
        bs = tf.shape(src)[0]
        preds = self.model.generate(src, self.trg_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target = ""
            for idx in trg[i, :]:
                target += self.idx_to_char[idx]
            prediction = "<"
            over = False
            for idx in preds[i, :]:
                if over:  # Add padding token once end token has beeen predicted
                    prediction += "-"
                    continue
                if idx == 28:
                    over = True
                prediction += self.idx_to_char[idx]
            print(f"target:     {target}")
            print(f"prediction: {prediction}")
            print()


"""
## Create & train the end-to-end model
"""

model = Transformer(
    input_type="feats",
    nvocab=1000,
    ninp=80,
    nhid=64,
    nhead=2,
    nff=128,
    src_maxlen=28,
    trg_maxlen=6,
    nlayers=2,
    nclasses=29,
)
for i in val_ds.take(1):
    batch = i  # Use the first batch of validation set to display outputs

cb = DisplayOutputs(batch)
model.compile(optimizer="adam")
model.fit(ds, callbacks=[cb], epochs=4)
