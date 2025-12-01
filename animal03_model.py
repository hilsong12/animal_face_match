import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ============================================================
# 1) 데이터 불러오기
# ============================================================
base_path = '/home/leejiseok/PycharmProjects/animalFace/'

X_train = np.load(base_path + 'animal_multi_X_train.npy')
X_test  = np.load(base_path + 'animal_multi_X_test.npy')
Y_train = np.load(base_path + 'animal_multi_Y_train.npy')
Y_test  = np.load(base_path + 'animal_multi_Y_test.npy')

num_classes = 16

# MobileNetV2는 [-1,1] 입력이므로 변환 필요
X_train = preprocess_input(X_train * 255)
X_test  = preprocess_input(X_test * 255)

# ============================================================
# 2) 정상적/적당한 데이터 증강 설정
# ============================================================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.8,1.2],
    horizontal_flip=True,
)

train_gen = datagen.flow(X_train, Y_train, batch_size=32)

# ============================================================
# 3) MobileNetV2 모델 구성 (초기 Freeze)
# ============================================================
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # 초기엔 Freeze

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-3),
    metrics=['accuracy']
)

# callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# ============================================================
# 4) 1단계: Freeze 학습
# ============================================================
print("\n=== 1단계: Freeze 학습 시작 ===")
history1 = model.fit(
    train_gen,
    epochs=20,
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping, reduce_lr]
)

score = model.evaluate(X_test, Y_test)
print("\n>> Freeze 학습 결과")
print("Loss:", score[0])
print("Acc :", score[1])

# ============================================================
# 5) 2단계: Fine-Tuning (상위 30개 Layer Unfreeze)
# ============================================================
print("\n=== 2단계: Fine-Tuning ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(5e-5),   # 낮은 학습률
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=20,
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping, reduce_lr]
)

score = model.evaluate(X_test, Y_test)
print("\n>> Fine-Tuning 최종 결과")
print("Loss:", score[0])
print("Acc :", score[1])

# 모델 저장
model.save(f'./animal_mobilenetv2_final_acc_{score[1]:.4f}.h5')

# ============================================================
# 6) 학습 그래프 출력
# ============================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history1.history['loss'], label='loss_freeze')
plt.plot(history1.history['val_loss'], label='val_loss_freeze')
plt.plot(history2.history['loss'], label='loss_finetune')
plt.plot(history2.history['val_loss'], label='val_loss_finetune')
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(history1.history['accuracy'], label='acc_freeze')
plt.plot(history1.history['val_accuracy'], label='val_acc_freeze')
plt.plot(history2.history['accuracy'], label='acc_finetune')
plt.plot(history2.history['val_accuracy'], label='val_acc_finetune')
plt.legend()
plt.title("Accuracy Curve")
plt.show()
