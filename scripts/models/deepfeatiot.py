import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Conv1D,
    GlobalAveragePooling1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    BatchNormalization,
    LayerNormalization,
    Bidirectional,
    GRU,
    concatenate,
    Dropout,
    ReLU,
)


# Custom Layer for GPT-2 Processing
class CustomGPT2Layer(tf.keras.layers.Layer):
    def __init__(self, gpt2_model):
        super(CustomGPT2Layer, self).__init__()
        self.gpt2_model = gpt2_model

    def call(self, input_ids, attention_mask):
        # Pass token IDs instead of embeddings
        outputs = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)


def conv_block(input_layer, kernel_size):
    activation = "relu"
    padding = "causal"
    dilation_rate = 1
    conv = Conv1D(
        128,
        kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
    )(input_layer)
    conv = Conv1D(
        128,
        kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
    )(conv)
    conv = Conv1D(
        128,
        kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
    )(conv)
    conv = MaxPooling1D(data_format="channels_last")(conv)
    conv = Conv1D(
        64,
        kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
    )(conv)
    conv = Conv1D(
        64,
        kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
    )(conv)
    conv = Conv1D(
        64,
        kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
    )(conv)
    conv = MaxPooling1D(data_format="channels_last")(conv)
    conv = Conv1D(
        64,
        kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
    )(conv)
    conv = Conv1D(
        64,
        kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
    )(conv)
    conv = Conv1D(
        64,
        kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate,
    )(conv)

    conv_global = GlobalAveragePooling1D(data_format="channels_last")(conv)
    return conv_global


def time_series_encoder_plus_plus(ts_input_layer, gpt_features):
    # GRU Blocks
    gru = Bidirectional(
        GRU(
            128,
            return_sequences=True,
        )
    )(ts_input_layer)
    gru = BatchNormalization()(gru)
    gru = Bidirectional(
        GRU(
            128,
            return_sequences=True,
        )
    )(gru)
    gru = BatchNormalization()(gru)
    gru = Bidirectional(GRU(64, return_sequences=False))(gru)
    gru = BatchNormalization()(gru)
    gru = ReLU()(gru)

    # # # CONV BLOCKS
    l11 = conv_block(input_layer=ts_input_layer, kernel_size=11)
    l7 = conv_block(input_layer=ts_input_layer, kernel_size=7)
    l5 = conv_block(input_layer=ts_input_layer, kernel_size=5)
    l3 = conv_block(input_layer=ts_input_layer, kernel_size=3)

    conv_concat = tf.keras.layers.Concatenate()([l11, l7, l5, l3])  # Depth Concatenate

    for _ in range(1):
        x = Conv1D(
            10000,
            9,
            activation=None,
            dilation_rate=4,
            padding="valid",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.05),
            # bias_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),
            use_bias=False,
            trainable=False,
        )(ts_input_layer)
        conv_max = GlobalMaxPooling1D(data_format="channels_last")(x)
        conv_ppv = tf.reduce_mean(tf.cast(x > 0, tf.float32), axis=1)
        concatenate_features_ul = concatenate([conv_max, conv_ppv])
    
    # # Dense Feature Transformation Blocks
    dense_nl = Dense(units=1024, activation="relu")(concatenate_features_ul)
    dense_nl = LayerNormalization()(dense_nl)
    dense_nl = Dense(units=64, activation="relu")(dense_nl)
    dense_nl = LayerNormalization()(dense_nl)

    dense_conv = Dense(units=64, activation="relu")(conv_concat)
    dense_conv = LayerNormalization()(dense_conv)

    dense_gru = Dense(units=64, activation="relu")(gru)
    dense_gru = LayerNormalization()(dense_gru)

    dense_gpt = Dense(units=64, activation="relu")(gpt_features)
    dense_gpt = LayerNormalization()(dense_gpt)

    merge = tf.keras.layers.Concatenate()(
        [
            dense_gru,
            dense_conv,
            dense_nl,
            dense_gpt,
        ]
    )
    
    return merge


# Define GPT-2 Model with Tokenization
def GPT2Model(seq_length):
    # Load the GPT-2 model configuration and model
    from transformers import GPT2Config, TFGPT2Model, GPT2Tokenizer
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Define inputs for token IDs & attention mask
    token_ids_input = Input(shape=(seq_length,), dtype=tf.int32, name="input_ids")
    attention_mask_input = Input(shape=(seq_length,), dtype=tf.int32, name="attention_mask")

    # Load GPT-2 model with frozen layers
    config = GPT2Config.from_pretrained("gpt2", output_hidden_states=True)
    gpt2_model = TFGPT2Model.from_pretrained("gpt2", config=config)
    for layer in gpt2_model.transformer.h:
        layer.trainable = False  # Freeze transformer layers

    # Custom GPT-2 layer that takes input_ids
    custom_gpt2_layer = CustomGPT2Layer(gpt2_model)

    # Pass token IDs and attention mask to GPT-2
    gpt2_outputs = custom_gpt2_layer(token_ids_input, attention_mask_input)

    # Apply Global Average Pooling to get final features
    gpt_features = tf.keras.layers.GlobalAveragePooling1D()(gpt2_outputs)

    # Build Model
    model = Model(
        inputs=[token_ids_input, attention_mask_input],
        outputs=gpt_features,
        name="GPT2_TimeSeries_Model"
    )

    return model, tokenizer  # Return both model & tokenize


def DeepFeatIoT(input_shape, num_labels):
    ts_input_layer = Input(shape=input_shape)
    gpt_features = Input(shape=(768,))
    ts_encoder_features = time_series_encoder_plus_plus(ts_input_layer, gpt_features)

    # NEW MLP HEAD
    l = Dense(units=128, activation="relu")(ts_encoder_features)
    l = Dropout(0.5)(l)
    l = LayerNormalization()(l)
    l = Dense(units=64, activation="relu")(l)
    l = Dropout(0.5)(l)
    l = LayerNormalization()(l)

    output_layer = Dense(units=num_labels, activation="softmax", dtype="float32")(l)
    model = Model(
        inputs=[ts_input_layer, gpt_features],
        outputs=output_layer,
        name="DeepFeatIoT",
    )
    return model


if __name__ == "__main__":
    print("This is a module")
