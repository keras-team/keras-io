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

For this demonstration, we will use LJSpeech dataset from the
[LibriVox](https://librivox.org/) project. It consists of short
audio clips of a single speaker reading passages from 7 non-fiction books.
Our model will be similar to the original Transformer (both encoder and decoder)
as proposed in the paper, "Attention is All You Need".


**References:**
- [Attention is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Very Deep Self-Attention Networks for End-to-End Speech Recognition](https://arxiv.org/pdf/1904.13377.pdf)
- [Speech Transformers](https://ieeexplore.ieee.org/document/8462506)
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
"""


import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


"""
## Transformer Input Layer

When processing past target tokens for the decoder, we compute the sum of 
position embeddings and token embeddings.

When processing audio features, we apply convolutional layers to downsample
them and process local relationships. It makes the training stable.
"""


class TransformerInput(layers.Layer):
    def __init__(
        self, input_type="tokens", num_vocab=1000, num_hid=64, num_ff=128, maxlen=100,
    ):
        super().__init__()
        self.input_type = input_type
        if input_type == "tokens":
            self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        elif input_type == "features":
            self.conv1 = tf.keras.layers.Conv1D(
                num_hid, 11, strides=2, padding="same", activation="relu"
            )
            self.conv2 = tf.keras.layers.Conv1D(
                num_hid, 11, strides=2, padding="same", activation="relu"
            )
            self.conv3 = tf.keras.layers.Conv1D(
                num_hid, 11, strides=2, padding="same", activation="relu"
            )
        else:
            raise ValueError("input_type must be one of ('tokens', 'features')")
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        if self.input_type == "tokens":
            maxlen = tf.shape(x)[-1]
            x = self.emb(x)
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            return x + positions
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return self.conv3(x)


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
        self.self_drop = layers.Dropout(0.5)
        self.enc_drop = layers.Dropout(0.0)
        self.ffn_drop = layers.Dropout(0.0)
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
        trg_norm = self.ln1(trg + self.self_drop(trg_att))
        enc_out = self.enc_att(trg_norm, enc_out)
        enc_out_norm = self.ln2(self.enc_drop(enc_out) + trg_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.ln3(enc_out_norm + self.ffn_drop(ffn_out))
        return ffn_out_norm


"""
## Complete Transformer Model

Our model takes audio spectrograms as input, and predicts a sequence of characters.
During training, we give the decoder the target character sequence shifted to the left
as input. During inference, the decoder uses its own past predictions to predict the 
next token.
"""

loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0.1, reduction="none"
)


def masked_loss(real, pred):
    """ assuming pad token index = 0 """
    one_hot = tf.one_hot(real, depth=34)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(one_hot, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_ff=128,
        src_maxlen=100,
        trg_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.trg_maxlen = trg_maxlen

        self.enc_input = TransformerInput(
            input_type="features", num_hid=num_hid, maxlen=src_maxlen
        )
        self.dec_input = TransformerInput(
            input_type="tokens",
            num_vocab=num_classes,
            num_hid=num_hid,
            maxlen=trg_maxlen,
        )
        for i in range(num_layers_enc):
            setattr(
                self,
                f"enc_layer_{i}",
                TransformerEncoderLayer(num_hid, num_head, num_ff),
            )
        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoderLayer(num_hid, num_head, num_ff),
            )

        self.final = layers.Dense(num_classes)

    def encode_src(self, src):
        x = self.enc_input(src)
        for i in range(self.num_layers_enc):
            x = getattr(self, f"enc_layer_{i}")(x)
        return x

    def decode(self, enc_out, trg):
        y = self.dec_input(trg)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        src = inputs[0]
        trg = inputs[1]
        x = self.encode_src(src)
        y = self.decode(x, trg)
        return self.final(y)

    @property
    def metrics(self):
        return [self.loss_metric]

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
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, src, trg_start_token_idx):
        """ inference for one batch of inputs using greedy decoding """
        bs = tf.shape(src)[0]
        enc = self.encode_src(src)
        dec_inp = tf.ones((bs, 1), dtype=tf.int32) * trg_start_token_idx
        dec_logits = []
        for i in range(self.trg_maxlen - 1):
            dec_out = self.decode(enc, dec_inp)
            logits = self.final(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_inp = tf.concat([dec_inp, last_logit], axis=-1)
        return dec_inp


"""
## Download dataset

Note: This requires ~3.6 GB of disk space and
takes ~5 minutes for the extraction of files.
"""


import keras
import os

keras.utils.get_file(
    os.path.join(os.getcwd(), "data.tar.gz"),
    "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    extract=True,
    archive_format="tar",
    cache_dir=".",
)


saveto = "./datasets/LJSpeech-1.1"
wavs = glob("{}/**/*.wav".format(saveto), recursive=True)

id_to_text = {}
with open(os.path.join(saveto, "metadata.csv")) as f:
    for line in f:
        id = line.strip().split("|")[0]
        text = line.strip().split("|")[2]
        id_to_text[id] = text


def get_data(wavs, id_to_text, maxlen=50):
    """ returns mapping of audio paths and transcription texts """
    data = []
    for w in wavs:
        id = w.split("/")[-1].split(".")[0]
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})
    return data


"""
## Preprocess dataset
"""


class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


max_target_len = 200  # all transcripts in out data are < 200 characters
data = get_data(wavs, id_to_text, max_target_len)
vectorize_layer = VectorizeChar(max_target_len)
len(vectorize_layer.get_vocabulary())


def create_text_ds(data):
    texts = [_["text"] for _ in data]
    text_ds = [vectorize_layer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds


def path_to_audio(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = tf.divide(tf.subtract(x, means), stddevs)
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def create_audio_ds(data):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(
        path_to_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return audio_ds


def create_tf_dataset(data, bs=4):
    audio_ds = create_audio_ds(data)
    text_ds = create_text_ds(data)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"src": x, "trg": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


"""
Check contents of one batch
"""


split = int(len(data) * 0.99)
train_data = data[:split]
test_data = data[split:]

ds = create_tf_dataset(train_data, bs=64)
val_ds = create_tf_dataset(test_data, bs=4)

for i in ds.take(1):
    print(i["src"].shape)
    print(i["trg"])


"""
## Callbacks to display predictions and to change learning rate
"""


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, idx_to_token, trg_start_token_idx=27, trg_end_token_idx=28
    ):
        """ Displays a batch of outputs after every epoch 
        Arguments:
        - batch: test batch containing the keys "src" and "trg"
        - idx_to_token: a List containing the vocabulary tokens corresponding to their indices
        - trg_start_token_idx: start token index in the target vocabulary
        - trg_end_token_idx: end token index in the target vocabulary
        """
        self.batch = batch
        self.trg_start_token_idx = trg_start_token_idx
        self.trg_end_token_idx = trg_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        src = self.batch["src"]
        trg = self.batch["trg"].numpy()
        bs = tf.shape(src)[0]
        preds = self.model.generate(src, self.trg_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target = ""
            for idx in trg[i, :]:
                target += "" + self.idx_to_char[idx]
            prediction = ""
            over = False
            for idx in preds[i, :]:
                if over:  # Add padding token once end token has beeen predicted
                    prediction += "-"
                    continue
                if idx == self.trg_end_token_idx:
                    over = True
                prediction += "" + self.idx_to_char[idx]
            print(f"target:     {target}")
            print(f"prediction: {prediction}")
            print()


def scheduler(epoch, lr):
    """ linear warm up - linear decay """
    init_lr = 0.00001
    lr_after_warmup = 0.001
    final_lr = 0.00001
    warmup_epochs = 15
    decay_epochs = 85
    if epoch < warmup_epochs:
        return init_lr + ((lr_after_warmup - init_lr) / (warmup_epochs - 1)) * epoch
    return max(
        final_lr,
        lr_after_warmup
        - (epoch - warmup_epochs) * (lr_after_warmup - final_lr) / (decay_epochs),
    )


"""
## Create & train the end-to-end model
"""


for i in val_ds.take(1):
    batch = i  # Use the first batch of validation set to display outputs

# vocabulary to convert predicted indices to characters
idx_to_char = vectorize_layer.get_vocabulary()
display_cb = DisplayOutputs(
    batch, idx_to_char, trg_start_token_idx=2, trg_end_token_idx=3
)  # set the arguments as per vocabulary index for '<' and '>'

schedule_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

model = Transformer(
    num_hid=200,
    num_head=2,
    num_ff=400,
    trg_maxlen=max_target_len,
    num_layers_enc=4,
    num_layers_dec=1,
    num_classes=34,
)
model.compile(optimizer="adam")

history = model.fit(ds, callbacks=[display_cb, schedule_cb], epochs=1)

"""
In practice, train for ~100 epochs. 

Some of the predicted text around epoch 35.
```
target:     <as they sat in the car, frazier asked oswald where his lunch was>
prediction: <as they sat in the car frazier his lunch ware mis lunch was>

target:     <under the entry for may one, nineteen sixty,>
prediction: <under the introus for may monee, nin the sixty,>
```
"""
