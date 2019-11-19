from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model as LoadModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator

INPUT_DIR = "./data_dogs_v_cats"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
MODEL_FILE = "./models/dogs_v_cats.h5"

BATCH_SIZE= 128
EPOCHS = 1
IMG_HEIGHT = 150
IMG_WIDTH = 150
#K_FOLDS = 5

def load_train_dataframe(input_dir, label_file="labels.csv"):
    train_df = pd.read_csv(os.path.join(input_dir, label_file), dtype=str)
    return train_df


def predict_model(model, input_dir, batch_size=BATCH_SIZE, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    datagen = ImageDataGenerator(rescale=1./255.)
    test_gen = datagen.flow_from_directory(
        batch_size=batch_size,
        directory=input_dir,
        shuffle=False,
        class_mode="binary",
        target_size=(img_height, img_width)
    )
    num_steps = 100
    result = model.predict_generator(
        test_gen,
        steps=num_steps,
        verbose=1
    )
    pred_idx = np.argmax(result, axis=1)
    out = pd.DataFrame({"id":test_gen.filenames[:num_steps*batch_size], "label":pred_idx})
    out.to_csv("models/results.csv",index=False)


def create_model(img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    model = Sequential([
        Conv2D(16, 3, padding="same", activation="relu", input_shape=(img_height, img_width, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model


def train_model(input_dir, train_df, val_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, show_plot=False):
    datagen = ImageDataGenerator(rescale=1./255., validation_split=val_split)
    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=input_dir,
        x_col="id",
        y_col="label",
        subset="training",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="binary",
        target_size=(img_height, img_width)
    )
    val_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=input_dir,
        x_col="id",
        y_col="label",
        subset="validation",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="binary",
        target_size=(img_height, img_width)
    )

    model = create_model()

    model.summary()

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.n // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.n // batch_size
    )

    if show_plot:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    return model


def load_model(model_file=MODEL_FILE):
    model = LoadModel(model_file)
    return model


def main():
    #train_df = load_train_dataframe(TRAIN_DIR)
    #model = train_model(TRAIN_DIR, train_df, show_plot=True)
    # save model
    #model.save(MODEL_FILE)
    model = load_model()
    predict_model(model, TEST_DIR, batch_size=1)


if __name__ == "__main__":
    main()


"""
def evaluate_model(model, input_dir, test_df):
    loss,acc = model.evaluate(test_df["id"],  test_df["label"], verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


def cross_validate(model, input_dir, train_df, k_folds=K_FOLDS):
    kf = KFold(n_splits=k_folds, shuffle=True)
    for a_idx, b_idx in kf.split(train_df):
        a_df = train_df.iloc[a_idx]
        b_df = train_df.iloc[b_idx]
        print("cross validating:")
        print(a_df, len(a_idx), b_df, len(b_idx))
"""
