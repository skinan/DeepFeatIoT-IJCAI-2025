import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from models.deepfeatiot import DeepFeatIoT, GPT2Model
from models.deepheteroiot import DeepHeteroIoT

from dataset_loading import load_data

import tensorflow as tf


def train_model(model, train_data, test_data, config):
    # Define the initial learning rate and decay parameters
    # Define the learning rate schedule
    initial_learning_rate = config["learning_rate"]
    decay_steps = 100
    decay_rate = 0.1
    staircase = True

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, amsgrad=True, clipvalue=1.0
    )
    loss = tf.keras.losses.CategoricalFocalCrossentropy()
    metrics = [
        "accuracy",
        tf.keras.metrics.F1Score(average="macro", name="mac_f1_score"),
        tf.keras.metrics.F1Score(average="weighted", name="wvg_f1_score"),
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(model.summary(show_trainable=True))


    train_task_name = config["model_name"] + "_" + config["train_task_name"]
    dataset_name = config["dataset_name"]
    checkpointer_file = f"model_checkpoints/{dataset_name}/{train_task_name}.tf"

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        checkpointer_file,
        monitor="val_accuracy",
        verbose=0,
        mode="max",
        save_best_only=True,
        save_weights_only=False,
    )
    from datetime import datetime

    date_time = datetime.now().strftime("%Y%m%d-%H%M")
    tensorboard_logger = tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/{dataset_name}/{train_task_name}_{date_time}"
    )

    callbacks = [tensorboard_logger, checkpointer]
    # with tf.device("/gpu:2"):
    history = model.fit(
        train_data[0],
        train_data[1],
        epochs=config["epochs"],
        validation_data=(test_data[0], test_data[1]),
        callbacks=callbacks,
    )

    val_accuracy = history.history["val_accuracy"]
    val_mac_f1_score = history.history["val_mac_f1_score"]
    val_wvg_f1_score = history.history["val_wvg_f1_score"]

    # Find the index of the maximum value in val_accuracy
    max_index = np.argmax(val_accuracy)

    # Get the corresponding value from val_f1_score
    corresponding_mac_f1_score = val_mac_f1_score[max_index]
    corresponding_wvg_f1_score = val_wvg_f1_score[max_index]

    print("Max val_accuracy:", val_accuracy[max_index])
    print("Corresponding val_mac_f1_score:", corresponding_mac_f1_score)
    print("Corresponding val_wvg_f1_score:", corresponding_wvg_f1_score)

    # Define the file path where you want to save the best accuracy
    return (
        val_accuracy[max_index],
        corresponding_mac_f1_score,
        corresponding_wvg_f1_score,
    )


# Function to Extract GPT Features
def extract_gpt_features(X_train, X_test):
    seq_length = X_train.shape[1]  # Define sequence length (adjust as needed)
    model, tokenizer = GPT2Model(seq_length)  # Get model & tokenizer

    # Function to tokenize a time series sample
    def tokenize_time_series(sample):
        text_representation = " ".join(map(str, sample))  # Convert numbers to text
        tokenized = tokenizer.encode(text_representation, max_length=seq_length, truncation=True, padding="max_length")
        return tokenized

    # Tokenize all time series data
    X_train_tokenized = np.array([tokenize_time_series(sample) for sample in X_train])
    X_test_tokenized = np.array([tokenize_time_series(sample) for sample in X_test])

    # Create attention masks (1 where there are tokens, 0 for padding)
    attention_mask_train = np.where(X_train_tokenized > 0, 1, 0)
    attention_mask_test = np.where(X_test_tokenized > 0, 1, 0)

    # Extract features using GPT-2 model
    X_train_gpt = model.predict([X_train_tokenized, attention_mask_train], batch_size=1)
    X_test_gpt = model.predict([X_test_tokenized, attention_mask_test], batch_size=1)

    return X_train_gpt, X_test_gpt


def run(config):
    X, y = load_data(config["dataset_name"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=100
    )

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    y_train_encoded = tf.keras.utils.to_categorical(y_train)
    y_test_encoded = tf.keras.utils.to_categorical(y_test)

    y_train_encoded = tf.convert_to_tensor(y_train_encoded, dtype=tf.float32)
    y_test_encoded = tf.convert_to_tensor(y_test_encoded, dtype=tf.float32)


    if config["model_name"] == "DeepFeatIoT": #### DeepFeatIoT Code ##########
        X_train_gpt, X_test_gpt = extract_gpt_features(X_train, X_test)
        print("X_train_gpt: ", X_train_gpt.shape)
        
        model = DeepFeatIoT(
            input_shape=X_train.shape[1:], num_labels=len(np.unique(y_train))
        )
        accuracy_score, mac_f1_score, wvg_f1_score = train_model(
            model,
            ([X_train, X_train_gpt], y_train_encoded),
            ([X_test, X_test_gpt], y_test_encoded),
            config,
        )
        
    elif config["model_name"] == "DeepHeteroIoT": #### DeepHeteroIoT Code ##########
        model = DeepHeteroIoT(
            input_shape=X_train.shape[1:], num_labels=len(np.unique(y_train))
        )
        accuracy_score, mac_f1_score, wvg_f1_score = train_model(
            model, (X_train, y_train_encoded), (X_test, y_test_encoded), config=config
        )
    else:
        raise KeyError(
            "Invalid Model Name !!!"
        )
        
    return accuracy_score, mac_f1_score, wvg_f1_score


if __name__ == "__main__":
    config = {
        "dataset_name": "Swiss",  # Options: Urban / Swiss / Iowa / SBAS
        "learning_rate": 0.001,
        "train_task_name": "supervised",
        "model_name": "DeepFeatIoT",
        "epochs": 200,
    }

    with tf.device("/GPU:0"):
        accuracy_score_list = []
        mac_f1_score_list = []
        wvg_f1_score_list = []

        seed = None
        config["random_seed"] = seed

        for _ in range(10):
            task_name = config["dataset_name"] + " " + config["model_name"] + " " + str(config["random_seed"])

            accuracy_score, mac_f1_score, wvg_f1_score = run(config)

            accuracy_score_list.append(accuracy_score)
            mac_f1_score_list.append(mac_f1_score)
            wvg_f1_score_list.append(wvg_f1_score)
            
        # Save scores in text file for future reference
        file_name = (
            "results/" + config["dataset_name"] + "/" + config["model_name"] + ".txt"
        )

        # Multiply each element by 100 and format to 2 decimal points
        accuracy_score_list = [round(val * 100, 2) for val in accuracy_score_list]
        mac_f1_score_list = [round(val * 100, 2) for val in mac_f1_score_list]
        wvg_f1_score_list = [round(val * 100, 2) for val in wvg_f1_score_list]

        # Write the scores to the text file
        with open(file_name, "a+") as file:
            file.write(f"accuracy = {accuracy_score_list}\n")
            file.write(f"mac_f1_score = {mac_f1_score_list}\n")
            file.write(f"wvg_f1_score = {wvg_f1_score_list}\n")