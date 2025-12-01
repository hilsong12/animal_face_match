from PIL import Image, ImageEnhance
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import random

img_dir = '/home/leejiseok/PycharmProjects/animalFace/animal_dataset'

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

target_count = 600  # ← 모든 클래스 600장으로 동일하게
X = []
Y = []

print("=== 이미지 불러오는 중 ===")

# 각 클래스 이미지 수 확인
class_counts = {}
class_files = {}

for idx, category in enumerate(categories):
    folder = os.path.join(img_dir, category)
    files = glob.glob(os.path.join(folder, '*.jpg'))
    class_files[category] = files
    class_counts[category] = len(files)
    print(f"{category:10s}: {len(files)}장 발견")

print(f"\n=== 모든 클래스 {target_count}장으로 균형 맞추기 ===\n")


def augment_image(img):
    """간단한 증강: 좌우 반전, 90도 회전, 밝기 변화"""
    # 랜덤 좌우 반전
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 랜덤 회전 (0, 90, 180, 270)
    rotations = [0, 90, 180, 270]
    img = img.rotate(random.choice(rotations))

    # 밝기 조절
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))

    return img


for idx, category in enumerate(categories):
    files = class_files[category]
    count = len(files)

    print(f"\n[{category}] 처리 중... (현재 {count}장)")

    # ------------------------------
    # 1) 이미지가 600장보다 많으면 → 600장 랜덤 선택
    # ------------------------------
    if count > target_count:
        print(f"{category}: {count}장 → 600장으로 줄이는 중...")
        files = random.sample(files, target_count)
        count = target_count

    # ------------------------------
    # 2) 원본 이미지 추가
    # ------------------------------
    for f in files:
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize((image_w, image_h))
            X.append(np.array(img))
            Y.append(idx)
        except:
            print(f"이미지 로드 실패: {f}")

    # ------------------------------
    # 3) 부족하면 증강으로 채움
    # ------------------------------
    needed = target_count - count
    if needed > 0:
        print(f"{category}: {count}장 → {target_count}장 되도록 {needed}장 증강")

    for _ in range(needed):
        f = random.choice(files)
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize((image_w, image_h))
            img = augment_image(img)
            X.append(np.array(img))
            Y.append(idx)
        except:
            print(f"증강 실패: {f}")

# ------------------------------
# numpy 배열 변환
# ------------------------------
X = np.array(X) / 255.0
Y = np.array(Y)

# ------------------------------
# Train/Test 분리 (클래스 균형 유지)
# ------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, shuffle=True, stratify=Y
)

# ------------------------------
# Numpy 저장
# ------------------------------
np.save('animal_multi_X_train.npy', X_train)
np.save('animal_multi_X_test.npy', X_test)
np.save('animal_multi_Y_train.npy', Y_train)
np.save('animal_multi_Y_test.npy', Y_test)

print("\n=== numpy 저장 완료! ===")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test :", X_test.shape)
print("Y_test :", Y_test.shape)
