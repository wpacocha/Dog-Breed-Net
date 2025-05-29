import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ścieżka do folderu ze zdjęciami na Google Drive
DATA_DIR = '/content/dogdataset/halfdata'
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2,
    dtype='float32'
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    color_mode='rgb'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    color_mode='rgb'
)

# (opcjonalnie) funkcja konwersji do tf.data.Dataset – NIE jest wymagana dla .fit()
# Możesz ją usunąć jeśli nie używasz tf.data pipeline
