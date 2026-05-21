import random
import shutil
from pathlib import Path

from src.config import TRAIN_DIR, DATA_DIR


TEST_DIR = DATA_DIR / "test"
TEST_RATIO = 0.10
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_files(class_dir: Path):
    return [
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_test_split():
    random.seed(SEED)

    if TEST_DIR.exists() and any(TEST_DIR.iterdir()):
        raise RuntimeError(
            f"Test directory already exists and is not empty: {TEST_DIR}. "
            "Delete it manually if you want to recreate the split."
        )

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = [
        path for path in TRAIN_DIR.iterdir()
        if path.is_dir()
    ]

    if not class_dirs:
        raise RuntimeError(f"No class folders found in {TRAIN_DIR}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        test_class_dir = TEST_DIR / class_name
        test_class_dir.mkdir(parents=True, exist_ok=True)

        images = get_image_files(class_dir)
        random.shuffle(images)

        test_count = max(1, int(len(images) * TEST_RATIO))
        test_images = images[:test_count]

        print(f"{class_name}: moving {test_count} / {len(images)} images to test")

        for image_path in test_images:
            destination = test_class_dir / image_path.name
            shutil.move(str(image_path), str(destination))

    print("\nDone. Test split created successfully.")
    print(f"Test directory: {TEST_DIR}")


if __name__ == "__main__":
    create_test_split()