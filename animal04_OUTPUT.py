from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import glob
import os

# ============================================================
# 1️⃣ 1차 학습 모델 자동 찾기
# ============================================================
model_files = glob.glob('./animal_mobilenetv2_final_acc_0.9385.h5')
if not model_files:
    raise FileNotFoundError("1차 학습 모델 파일을 찾을 수 없습니다.")
# 최신(혹은 마지막) 모델 선택
latest_model_path = sorted(model_files)[-1]
print("불러올 모델:", latest_model_path)

# 모델 로드
model = load_model(latest_model_path)

# ============================================================
# 2️⃣ 클래스 순서 및 이미지 크기
# ============================================================
categories = [
    "Bear",
    "Cat",
    "Cattle",
    "Chicken",
    "Deer",
    "Dog",
    "Duck",
    "Fox",
    "Hamster",
    "Horse",
    "Lion",
    "Monkey",
    "Pig",
    "Rabbit",
    "Sheep",
    "Turtle"
]
image_w = 128
image_h = 128

# ============================================================
# 3️⃣ 이미지 1장 예측 함수
# ============================================================
def predict_image(file_path):
    """이미지 1장을 예측하는 함수"""
    try:
        img = Image.open(file_path)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))

        data = np.asarray(img) / 255.0
        data = data.reshape(1, image_w, image_h, 3)

        preds = model.predict(data)

        # 전체 확률 출력
        print("=== 전체 클래스 확률 ===")
        for i, p in enumerate(preds[0]):
            print(f"{categories[i]:10s} : {p*100:6.2f}%")
        print("-" * 40)

        # 가장 높은 확률 선택
        class_index = np.argmax(preds[0])
        class_name = categories[class_index]
        confidence = preds[0][class_index]

        print(f"파일: {file_path}")
        print("예측 class index:", class_index)
        print("예측 class name :", class_name)
        print(f"최종 확률        : {confidence*100:.2f}%")
        print("=" * 40)

        return class_name

    except Exception as e:
        print("에러:", e)

# ============================================================
# 4️⃣ 테스트할 이미지 지정
# ============================================================
img1 = "/home/leejiseok/Downloads/b123.jpg"
# img2 = "/home/user8/Downloads/boy.jpg"
# img3 = "/home/user8/Downloads/dogimage.jpg"
# img4 = "/home/user8/Downloads/park.jpg"
# img5 = "/home/user8/Downloads/제목 없는 디자인.jpg"
predict_image(img1)
# predict_image(img2)
# predict_image(img3)
# predict_image(img4)
# predict_image(img5)
