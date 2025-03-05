import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# กำหนดให้ใช้ GPU ถ้ามี
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU for training")
else:
    print("GPU not found, using CPU instead")

# กำหนดพารามิเตอร์
img_size = (128, 128)  # เปลี่ยนขนาดรูปจาก (128, 128) เป็น (224, 224)
batch_size = 32  # ลด batch size จาก 32 เป็น 16
num_classes = 5

# โหลดและเตรียม dataset (แบ่ง train/validation อัตโนมัติ)
data_dir = "Instagram Photos/Intragram Images [Original]"

""" train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,  # เพิ่มเป็น 30 องศา
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.9, 1.1],  # เพิ่มการปรับความสว่าง
    shear_range=0.2,  # เพิ่ม shear transform
    zoom_range=0.2,  # เพิ่ม zoom
    horizontal_flip=True,
    validation_split=0.2
) """


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,  # ลด rotation range
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # แบ่ง 20% เป็น validation
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical',
    subset='validation'
)

# โหลด EfficientNetB7 เป็น base model
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# Unfreeze ชั้นสุดท้ายของ base model
base_model.trainable = True


# สร้างโมเดลใหม่โดยใช้ EfficientNetB7
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# คอมไพล์โมเดลด้วย Learning Rate ที่ต่ำลง
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


checkpoint = ModelCheckpoint(
    "food_classify.keras",
    verbose=1,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

# เทรนโมเดลในโหมด GPU ถ้ามี
epochs = 50
with tf.device('/GPU:0'):
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stopping,checkpoint,reduce_lr]
    )

# บันทึกโมเดล
model.save("food_classifier_mobilenetV3.h5")
