from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data_preparation import train_generator, val_generator, DATA_DIR
from plot_training import plot_training
import os

# Liczba klas = liczba folderów
NUM_CLASSES = len([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Detected {NUM_CLASSES} classes")

# Parametry
EPOCHS = 80
EPOCHS_FROZEN = 40
EPOCHS_UNFROZEN = 40
LEARNING_RATE = 1e-4

# Budowanie modelu
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Zamrożenie wszystkich warstw
for layer in base_model.layers:
    layer.trainable = False

# Kompilacja
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 🔹 Faza 1: 10 epok – trenowanie tylko nowej głowy
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FROZEN,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[
        ModelCheckpoint("best_model_vgg16.h5", save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
    ]
)

# 🔓 Odblokowanie wyższych warstw VGG16
print("🔓 Odblokowuję część warstw VGG16...")
for layer in base_model.layers[15:]:
    layer.trainable = True

# Kompilacja po odblokowaniu
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE / 10),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 🔹 Faza 2: 10 kolejnych epok
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    initial_epoch=EPOCHS_FROZEN,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[
        ModelCheckpoint("best_model_vgg16.h5", save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
    ]
)

# 🔚 Wykres
plot_training([history1, history2], "VGG16")
