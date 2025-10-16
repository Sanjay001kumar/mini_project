# validate.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model("trained_data_model.h5")

# Dataset paths
VAL_DIR = "C:\\Indian-Currency-Recognition-System-master\\data\\val"
IMG_SIZE = (192,192)
BATCH_SIZE = 32

# Validation Data
val_gen = ImageDataGenerator(rescale=1./255)
val_ds = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Run evaluation
val_loss, val_acc = model.evaluate(val_ds)
print(f"✅ Validation Accuracy: {val_acc:.4f}")
print(f"✅ Validation Loss: {val_loss:.4f}")
