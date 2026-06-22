import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ========== CONFIGURATION (hardcoded) ==========
TRAIN_DIR = r"D:\Major Project\frame_training_data\train"
VAL_DIR   = r"D:\Major Project\frame_training_data\val"
TEST_DIR  = r"D:\Major Project\frame_training_data\test"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS_FROZEN = 3
EPOCHS_FINETUNE = 10
LEARNING_RATE_FROZEN = 0.001
LEARNING_RATE_FINETUNE = 0.0001
MOMENTUM = 0.9
# ================================================

# Data generators with preprocessing for Xception
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
    horizontal_flip=True,
    rotation_range=0.1,
    zoom_range=0.1
)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['REAL', 'FAKE']   # REAL=0, FAKE=1
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['REAL', 'FAKE']
)

# Build Xception model (pretrained, frozen first)
base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=SGD(learning_rate=LEARNING_RATE_FROZEN, momentum=MOMENTUM),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Stage 1: Training top layers only")
model.fit(train_generator, epochs=EPOCHS_FROZEN, validation_data=val_generator)

# Unfreeze top layers of base model (from layer 56 onward)
base_model.trainable = True
for layer in base_model.layers[:56]:
    layer.trainable = False

model.compile(optimizer=SGD(learning_rate=LEARNING_RATE_FINETUNE, momentum=MOMENTUM),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Stage 2: Fine-tuning")
model.fit(train_generator, epochs=EPOCHS_FINETUNE, validation_data=val_generator)

# Save model
model.save('xception_deepfake_image.h5')
print("Model saved as xception_deepfake_image.h5")

# Optional: evaluate on test set
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['REAL', 'FAKE'],
    shuffle=False
)
loss, acc = model.evaluate(test_generator)
print(f"Test accuracy: {acc:.4f}")