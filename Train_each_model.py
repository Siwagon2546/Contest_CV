import os
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Concatenate, Dense
from tensorflow.keras.utils import plot_model

# โหลด CSV
csv_path = "data_from_questionaire.csv"
df = pd.read_csv(csv_path)#

# รายการอาหารที่ต้องแยก
food_categories = ["Sushi", "Ramen", "Pizza", "Dessert", "Burger"] #, "Ramen", "Pizza", "Dessert", "Burger"]
image_folder = "Questionair Images"
input_shape = (256, 256, 3)

def load_and_preprocess_image(img_name):
    img_path = os.path.join(image_folder, img_name)
    img = load_img(img_path, target_size=(256, 256))
    img = img_to_array(img) / 255.0
    return img

def preprocess_data(df_subset):
    X1, X2, y = [], [], []
    for _, row in df_subset.iterrows():
        img1 = load_and_preprocess_image(row["Image 1"])
        img2 = load_and_preprocess_image(row["Image 2"])
        label = 0 if row["Winner"] == 1 else 1
        X1.append(img1)
        X2.append(img2)
        y.append(label)
    return np.array(X1), np.array(X2), np.array(y).reshape(-1, 1).astype(np.float32)

# เทรนแยกแต่ละชนิดอาหาร
for category in food_categories:
    print(f"Training model for {category}...")
    
    # เลือกเฉพาะข้อมูลที่เป็นประเภทอาหารนั้น ๆ
    df_subset = df[df["Menu"] == category]
    X1, X2, y = preprocess_data(df_subset)
    
    IMAGE_SIZE = (256, 256)
    
    # Create Model
    input_encoder = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_encoder)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu')(pool3)
    pool4 = MaxPooling2D((2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation='relu')(pool4)
    pool5 = MaxPooling2D((2, 2))(conv5)

    flat = Flatten()(pool2)
    encoder = Model(inputs=input_encoder, outputs=flat)

    input_1 = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    feature_1 = encoder(input_1)

    input_2 = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    feature_2 = encoder(input_2)

    concat = Concatenate()([feature_1, feature_2])
    dense1 = Dense(128, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=[input_1, input_2], outputs=output)

    # เปลี่ยน Loss ให้เหมาะสมกับ Sigmoid Output
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Callback บันทึกเฉพาะโมเดลที่ดีที่สุด
    checkpoint = ModelCheckpoint(f"food_comparator_{category}.keras", verbose=1, monitor="val_accuracy", save_best_only=True, mode="max")
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)

    # เทรนโมเดลเฉพาะประเภทอาหารนั้น
    model.fit([X1, X2], y, epochs=100, batch_size=16, validation_split=0.2, callbacks=[checkpoint, reduce_lr, early_stopping])

    print(f"Finished training {category}!")
