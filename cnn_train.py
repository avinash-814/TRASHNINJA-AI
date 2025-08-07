import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Paths
data_dir = "dataset"  # 📂 Replace with your dataset folder name if different
model_path = "smartbin_cnn_model.keras"

# Hyperparameters
img_size = 128
batch_size = 32
epochs = 20

# 📦 Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# 📥 Load Training Data
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

# 📥 Load Validation Data
val_data = val_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# 🧠 CNN Model Architecture
model = Sequential([
    tf.keras.Input(shape=(img_size, img_size, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# ⚙️ Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ⏹️ Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 🚀 Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop]
)

# 💾 Save Model (new format)
model.save(model_path)
print(f"✅ Model trained and saved as {model_path}")

# 📊 Evaluate Accuracy
val_loss, val_accuracy = model.evaluate(val_data)
print(f"📈 Final Validation Accuracy: {val_accuracy * 100:.2f}%")
