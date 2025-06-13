from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data_preparation import train_generator, val_generator
from plot_training import plot_training

NUM_CLASSES = train_generator.num_classes
EPOCHS_FROZEN = 20
EPOCHS_UNFROZEN = 40
LEARNING_RATE = 1e-4

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Faza 1 - Zamrożony backbone
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy', metrics=['accuracy'])

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=EPOCHS_FROZEN,
    callbacks=[
        ModelCheckpoint("best_model_densenet121.h5", save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
    ]
)

# Faza 2 - Odblokowanie wyższych warstw
for layer in base_model.layers[-30:]:  # np. odblokuj 30 ostatnich warstw
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE / 10),
              loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    initial_epoch=EPOCHS_FROZEN,
    epochs=EPOCHS_UNFROZEN,
    callbacks=[
        ModelCheckpoint("best_model_densenet121.h5", save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
    ]
)

plot_training([history1, history2], "DenseNet121")
