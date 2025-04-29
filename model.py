import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import os


# To multiply images in the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # %20 seperated for validation
)

# Train data
train_generator = train_datagen.flow_from_directory(
    'waste/',
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

# Validation data
validation_generator = train_datagen.flow_from_directory(
    'waste/',
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dropout(0.5), # Disconnect some neurons to prevent overfitting
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid') # For binary classification
])

# Compiling the model
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=40
)

print("Model Training comleted")
# Saving the model
model.save('modelFinal_41.h5')
print("Model Saved")