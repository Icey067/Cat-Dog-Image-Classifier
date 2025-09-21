import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomContrast)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
import pickle

mixed_precision.set_global_policy("mixed_float16")

image_size = (128, 128)
batch_size = 64

train_data = keras.utils.image_dataset_from_directory(
    directory="/mnt/c/Users/Aaditya/Desktop/Coding/Python/ML/CATS VS DOGS/train",
    labels="inferred",
    label_mode="int",
    batch_size=batch_size,
    image_size=image_size
)

val_data = keras.utils.image_dataset_from_directory(
    directory="/mnt/c/Users/Aaditya/Desktop/Coding/Python/ML/CATS VS DOGS/test",
    labels="inferred",
    label_mode="int",
    batch_size=batch_size,
    image_size=image_size
)

def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

autotune = tf.data.AUTOTUNE
train_data = train_data.map(process, num_parallel_calls=autotune)
val_data = val_data.map(process, num_parallel_calls=autotune)

train_data = train_data.cache(filename='./tf_train_cache').shuffle(500).prefetch(buffer_size=autotune)
val_data = val_data.cache(filename='./tf_validation_cache').prefetch(buffer_size=autotune)

model = Sequential([
    RandomFlip("horizontal", input_shape=(128, 128, 3)),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomContrast(0.2),

    Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, kernel_size=(3, 3), padding='same', activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.6),
    Dense(1, activation="sigmoid", dtype="float32")
])

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

model.summary()

my_callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath="best_model_v2.keras", save_best_only=True, monitor="val_loss")
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=my_callbacks
)

model.save("final_model_v2.keras")

with open('training_history_v2.pkl', 'wb') as file:
    pickle.dump(history.history, file)

print("Done training, models and history saved.")