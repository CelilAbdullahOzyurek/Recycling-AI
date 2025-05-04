import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout , BatchNormalization
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# To multiply images in the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.4,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest',
    horizontal_flip=True,
    validation_split=0.2
)

# Train data
train_generator = train_datagen.flow_from_directory(
    'waste/',
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

# Validation data
validation_generator = train_datagen.flow_from_directory(
    'waste/',
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    
    Dropout(0.5), # Disconnect some neurons to prevent overfitting
    
    Dense(512, activation='relu'),
    Dense(5, activation='softmax') # For binary classification
])

# Compiling the model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=40
)


print("Model Training comleted")
# Saving the model
model.save('model.h5')
print("Model Saved")