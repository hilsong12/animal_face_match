
이미지 

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
]\
Google Drive 
https://drive.google.com/drive/folders/1NK4rZQkltF9WXqm1sCI9GsKo90eIuEZ7?usp=sharing


🐾 딥러닝 기반 16종 동물 얼굴 분류 및 닮은꼴 매칭 시스템


📜 프로젝트 개요

MobileNetV2 기반 전이학습 모델을 이용해 16종류의 동물 얼굴 이미지를 분류하는 딥러닝 시스템을 구축했다.
수집된 이미지 데이터를 활용해 137,000+ 학습 샘플 전처리, 데이터셋 구성, 증강(Augmentation), Fine-Tuning을 통해 모델 성능을 향상시켰다.
Python 기반 환경(Pycharm)에서 PyQt5 Designer를 활용한 GUI 프로그램을 제작하여 사용자가 사진을 업로드하거나 웹캠을 통해 실시간으로 동물 분류 결과를 확인할 수 있도록 구현했다.
모델 출력은 Softmax 확률을 이용해 예측 결과와 Confidence Score를 직관적으로 보여주도록 구성했다.




👨‍🔧 기술적 핵심 요소 및 구현 내용
1. 데이터 수집 및 전처리
Bing/Google Image Crawler를 활용하여 동물별 데이터셋 생성
이미지 크기 통일(128×128), RGB 통합 변환, Label Encoding, Train/Test Split 처리
클래스 불균형이 존재해 class weighting 및 데이터 증강 적용

2. 전이학습(Transfer Learning) + Fine-Tuning
MobileNetV2의 Feature Extractor 부분을 Freeze하여 1차 학습
이후 상위 20~30개 레이어를 Unfreeze하여 Fine-Tuning 진행
EarlyStopping, ReduceLROnPlateau 등 콜백 사용으로 과적합(Overfitting) 방지

3. 모델 평가 및 최종 선택
Top-1 Accuracy, Confusion Matrix 분석
최초 학습 대비 Fine-Tuning 후 성능 개선 폭 확인
과적합이 발생하는 구간에서 BatchNormalization / Dropout 조정

4. GUI 기반 실시간 분류 프로그램 제작
PyQt5 Designer로 UI 개발 후 .ui → Python 코드 자동 변환
웹캠(OpenCV) 실시간 캡처 기능으로 즉석 분류 가능
Pillow, Numpy, MobileNetV2 모델(.h5) 로딩
결과 이미지를 GUI에 표시 + 예측 확률 시각화 구현




💡 프로젝트를 통해 새롭게 배운 점
데이터 품질이 모델 성능을 좌우한다는 것을 경험적으로 체감
→ 같은 모델 구조라도, 전처리/데이터 증강 전략에 따라 정확도가 크게 달라짐
전이학습 모델을 활용하면 학습 비용과 시간, 데이터 요구량이 크게 절감됨을 확인
Fine-Tuning 과정에서 얼마나 Freeze/Unfreeze할지에 따라 성능이 극적으로 변해, 모델 구조 이해가 중요함을 배움
PyQt5 Designer를 활용한 UI 구성으로, 딥러닝 모델의 실사용 인터페이스 제작 경험을 얻음



🔧 아쉬웠던 점 및 향후 보완 계획
모델 성능이 떨어졌을 때, 문제의 원인을 정확히 모델 내부에서 파악하기가 쉽지 않았다.
결국 데이터 전처리나 증강 방식을 조정하며 실험적으로 해결해야 했다.

이미지 기반 분류에서는 배경색, 조명, 촬영 각도 등 불필요한 요소들이 예측에 영향을 주는 현상을 확인
→ 배경 제거(Segmentation), Bounding Box Crop 등을 적용해보고 싶다.

Fine-Tuning 시 레이어 선택, Learning Rate 등 하이퍼파라미터 튜닝에 시간이 오래 걸림
→ Keras Tuner나 Optuna 기반 자동화 튜닝 시스템 도입 예정

현재 GUI는 단일 이미지 예측 중심이라
→ 여러 이미지를 일괄 분류하는 기능,
→ 예측 결과 리포트 자동 생성 기능 등을 추가할 수 있을 것 같다.

📈 프로젝트를 통해 성장한 역량

딥러닝 전체 파이프라인(Data → Model → Evaluation → Deployment)을 독자적으로 구성
오류 발생 시 디버깅(레이블 mismatch, 입력 shape 오류, one-hot 인코딩 문제 등) 해결 능력 향상
Python, Keras, PyQt5, OpenCV 등 다양한 기술 스택을 통합

모델을 실제 앱 형태로 배포하기 위한 UX/UI 고려 경험 축적
