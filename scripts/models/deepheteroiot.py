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


def time_series_encoder(ts_input_layer):
    lstm = Bidirectional(
        GRU(
            128,
            return_sequences=True,
        )
    )(ts_input_layer)
    lstm = BatchNormalization()(lstm)
    lstm = Bidirectional(
        GRU(
            128,
            return_sequences=True,
        )
    )(lstm)
    lstm = BatchNormalization()(lstm)
    lstm = Bidirectional(GRU(64, return_sequences=False))(lstm)
    lstm = BatchNormalization()(lstm)

    l11 = conv_block(input_layer=ts_input_layer, kernel_size=11)
    l7 = conv_block(input_layer=ts_input_layer, kernel_size=7)
    l5 = conv_block(input_layer=ts_input_layer, kernel_size=5)
    l3 = conv_block(input_layer=ts_input_layer, kernel_size=3)

    conv_concat = concatenate([l11, l7, l5, l3])  # Depth Concatenate

    merge = concatenate([lstm, conv_concat])
    return merge


def DeepHeteroIoT(input_shape, num_labels):
    ts_input_layer = Input(shape=input_shape)
    ts_encoder = time_series_encoder(ts_input_layer)
    l = Dense(units=1024, activation="relu")(ts_encoder)
    l = LayerNormalization()(l)
    l = Dense(units=512, activation="relu")(l)
    l = LayerNormalization()(l)
    l = Dense(units=256, activation="relu")(l)
    l = LayerNormalization()(l)
    l = Dense(units=64, activation="relu")(l)
    l = LayerNormalization()(l)

    output_layer = Dense(units=num_labels, activation="softmax", dtype="float32")(l)
    model = Model(
        inputs=ts_input_layer,
        outputs=output_layer,
        name="DeepHeteroIoT",
    )
    return model


if __name__ == "__main__":
    print("This is a module")
