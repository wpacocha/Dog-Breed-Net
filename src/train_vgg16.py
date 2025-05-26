from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data_preparation import train_generator, val_generator
from plot_training import plot_training

NUM_CLASSES = 120
EPOCHS = 10
LEARNING_RATE = 1e-4

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint("best_model_vgg16.h5", save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=2064,  # 16508 / 8
    validation_steps=509,  # 4072 / 8
    callbacks=callbacks
)


plot_training(history, "VGG16")
