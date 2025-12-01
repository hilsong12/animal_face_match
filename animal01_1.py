import os
import glob
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from PIL import Image

# ============================================================
# 0) 설정
# ============================================================
SAVE_DIR = './animal_dataset'
TARGET_COUNT_PER_SITE = 200       # 사이트당 200장 → 총 400장 예상
FINAL_TARGET_COUNT = 800          # 최종 목표치

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

extra_keywords = [
    "cute", "wildlife", "hd", "4k", "close up", "real photo"
]

# ============================================================
# 1) 폴더 생성
# ============================================================
os.makedirs(SAVE_DIR, exist_ok=True)
for c in categories:
    os.makedirs(os.path.join(SAVE_DIR, c), exist_ok=True)


# ============================================================
# 2) JPG 변환 함수
# ============================================================
def convert_to_jpg(folder):
    files = glob.glob(os.path.join(folder, '*'))
    converted = 0

    for f in files:
        try:
            img = Image.open(f).convert('RGB')
            new_name = os.path.splitext(f)[0] + ".jpg"
            img.save(new_name, 'JPEG')

            if f != new_name:
                os.remove(f)
            converted += 1

        except Exception:
            if os.path.exists(f):
                os.remove(f)

    print(f" → JPG 변환 완료: {converted}장")


# ============================================================
# 3) Bing + Google 다운로드 함수
# ============================================================
def download_from_sources(keyword, folder):

    # Bing
    print("  - Bing 크롤링...")
    bing = BingImageCrawler(storage={'root_dir': folder})
    bing.crawl(
        keyword=keyword,
        max_num=TARGET_COUNT_PER_SITE,
        min_size=(50, 50)
    )

    # Google
    print("  - Google 크롤링...")
    google = GoogleImageCrawler(storage={'root_dir': folder})
    google.crawl(
        keyword=keyword,
        max_num=TARGET_COUNT_PER_SITE,
        min_size=(50, 50)
    )


# ============================================================
# 4) 본격 다운로드
# ============================================================
for c in categories:

    print(f"\n======================================")
    print(f"   🦊 {c} 이미지 다운로드 시작")
    print("======================================")

    folder = os.path.join(SAVE_DIR, c)

    # 검색어 다양화 반복
    for k in extra_keywords:
        full_keyword = f"{c} {k} animal"
        print(f" 검색어: {full_keyword}")
        download_from_sources(full_keyword, folder)

    # JPG 변환
    convert_to_jpg(folder)

    # 개수 체크
    count = len(glob.glob(os.path.join(folder, "*.jpg")))
    print(f"현재 {c} 이미지 개수: {count}장\n")

    if count < FINAL_TARGET_COUNT:
        print(f"⚠ {c}는 이미지가 부족합니다 → 더 많은 검색어를 추가하거나 다른 사이트 필요")


print("\n🎉 모든 카테고리 다운로드 완료!")
