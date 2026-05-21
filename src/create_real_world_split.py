import random
import shutil
from pathlib import Path

from src import config


SOURCE_DIR = config.DATA_DIR / "real_world_unseen_data"
TRAIN_DIR = config.DATA_DIR / "real_world_train"
TEST_DIR = config.DATA_DIR / "real_world_test"

TRAIN_RATIO = 0.70
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder: Path):
    return [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_split():
    random.seed(SEED)

    if not SOURCE_DIR.exists():
        raise RuntimeError(f"Source folder not found: {SOURCE_DIR}")

    if TRAIN_DIR.exists() or TEST_DIR.exists():
        raise RuntimeError(
            "real_world_train or real_world_test already exists. "
            "Delete them manually if you want to recreate the split."
        )

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = [
        folder for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]

    for class_dir in class_dirs:
        class_name = class_dir.name

        train_class_dir = TRAIN_DIR / class_name
        test_class_dir = TEST_DIR / class_name

        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        images = get_images(class_dir)
        random.shuffle(images)

        train_count = int(len(images) * TRAIN_RATIO)

        train_images = images[:train_count]
        test_images = images[train_count:]

        print(
            f"{class_name}: "
            f"{len(train_images)} train, {len(test_images)} test"
        )

        for image_path in train_images:
            shutil.copy2(image_path, train_class_dir / image_path.name)

        for image_path in test_images:
            shutil.copy2(image_path, test_class_dir / image_path.name)

    print("\nDone.")
    print(f"Real-world train: {TRAIN_DIR}")
    print(f"Real-world test:  {TEST_DIR}")


if __name__ == "__main__":
    create_split()