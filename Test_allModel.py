import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# กำหนด path
classify_model_path = "Best_classify_model.keras"
compare_model_paths = {
    "Sushi": "food_comparator_Sushi.keras",
    "Ramen": "food_comparator_Ramen.keras",
    "Pizza": "food_comparator_Pizza.keras",
    "Dessert": "food_comparator_Dessert.keras",
    "Burger": "food_comparator_Burger.keras"
}

test_csv_path = "test.csv"
output_csv_path = "test_results.csv"
image_folder = "Test Images"

# โหลดโมเดล classify
classify_model = keras.models.load_model(classify_model_path)
# โหลดโมเดลเปรียบเทียบ
compare_models = {category: keras.models.load_model(path) for category, path in compare_model_paths.items()}

# โหลดข้อมูลทดสอบ
df_test = pd.read_csv(test_csv_path)

# ฟังก์ชันโหลดและ preprocess รูป
def load_and_preprocess_image(img_name, target_size=(128, 128)):
    img_path = os.path.join(image_folder, img_name)
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)  # เพิ่มมิติสำหรับ batch

def show_comparison(img1_name, img2_name, winner,category):
    img1_path = os.path.join(image_folder, img1_name)
    img2_path = os.path.join(image_folder, img2_name)
    
    img1 = load_img(img1_path)
    img2 = load_img(img2_path)
    
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    fig.suptitle(category,fontsize=32)
    
    if winner == 1:
        ax[0].set_title("Winner", color='green')
        ax[1].set_title("Loser", color='red')
    else:
        ax[0].set_title("Loser", color='red')
        ax[1].set_title("Winner", color='green')
    
    plt.show()

# บันทึกผลลัพธ์
results = []

for _, row in df_test.iterrows():
    img1_name, img2_name = row["Image 1"], row["Image 2"]
    
    # ทำนายหมวดหมู่ของรูปแต่ละภาพ
    img1 = load_and_preprocess_image(img1_name)
    img2 = load_and_preprocess_image(img2_name)
    
    category1 = classify_model.predict(img1).argmax()
    category2 = classify_model.predict(img2).argmax()
    categories = ["Burger","Dessert","Pizza","Ramen","Sushi"]
    
    category = categories[category1]
    print(category, img1_name)
    compare_model = compare_models[category]
    
    # if(category == "Sushi"):
    img1_large = load_and_preprocess_image(img1_name, target_size=(256, 256))
    img2_large = load_and_preprocess_image(img2_name, target_size=(256, 256))
        # โหลดและรวมภาพสำหรับโมเดลเปรียบเทียบ
    # else:
    #     img1_large = load_and_preprocess_image(img1_name, target_size=(224, 224))
    #     img2_large = load_and_preprocess_image(img2_name, target_size=(224, 224))
    # #combined_image = np.concatenate((img1_large, img2_large), axis=2)
    
    # ทำนายว่ารูปไหนน่ากินกว่า
    prediction = compare_model.predict([img1_large,img2_large])
    winner = 1 if (prediction) < 0.5 else 2
    
    # แสดงผลรูปเปรียบเทียบ
    show_comparison(img1_name, img2_name, winner,category)
    
    # บันทึกผลลัพธ์
    results.append({"Image 1": img1_name, "Image 2": img2_name, "Winner": winner})

# สร้าง DataFrame และบันทึกไฟล์ CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)

print(f"Test results saved to {output_csv_path}")
