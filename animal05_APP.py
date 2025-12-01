import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap

from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import glob

form_window = uic.loadUiType('./animal_face_match.ui')[0]


class AnimalClassifier:
    def __init__(self):

        # model_files = glob.glob('./animal_mobilenetv2_final_acc_*.h5')
        model_files = glob.glob('./animal_mobilenetv2_final_acc_0.9385.h5')
        if not model_files:
            raise FileNotFoundError("MobilenetV2 모델 파일을 찾을 수 없습니다.")

        self.model_path = sorted(model_files)[-1]
        print("불러올 모델:", self.model_path)

        self.model = load_model(self.model_path)

        self.categories = [
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
        self.image_w = 128
        self.image_h = 128

    def predict_image(self, file_path):
        try:
            img = Image.open(file_path)
            img = img.convert('RGB')
            img = img.resize((self.image_w, self.image_h))

            data = np.asarray(img) / 255.0
            data = data.reshape(1, self.image_w, self.image_h, 3)

            preds = self.model.predict(data)

            # ====== 전체 클래스 확률만 저장 ======
            result_text = "\n====== 전체 클래스 확률 ======\n"
            for i, p in enumerate(preds[0]):
                result_text += f"{self.categories[i]:10s} : {p*100:6.2f}%\n"
            result_text += "=================================\n"

            # 최종 결과 (여기는 result_text에 추가하지 않음)
            idx = np.argmax(preds[0])
            class_name = self.categories[idx]
            confidence = preds[0][idx]

            return class_name, confidence, result_text

        except Exception as e:
            print("이미지 예측 중 에러:", e)
            return None, 0.0, "예측 오류 발생"


class Exam(QWidget, form_window):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path = None
        self.classifier = AnimalClassifier()

        self.pushButton.clicked.connect(self.open_file)

    def open_file(self):
        self.path = QFileDialog.getOpenFileName(
            self,
            'Open Image',
            '/home/leejiseok/Downloads',
            'Image Files(*.jpg *.png);;All Files(*.*)'
        )

        file_path = self.path[0]
        if file_path == '':
            return

        print("선택된 파일:", file_path)

        pixmap = QPixmap(file_path)
        self.label.setPixmap(pixmap)

        class_name, confidence, result_text = self.classifier.predict_image(file_path)

        if class_name is not None:
            # label_2 → 전체 클래스 확률만 표시
            self.label_2.setText(result_text)

            # label_3 → 최종 예측만 표시
            self.label_3.setText(f"{class_name} / {confidence*100:.2f}%")
        else:
            self.label_3.setText("Error: 예측 실패")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())
