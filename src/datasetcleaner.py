from pathlib import Path
from PIL import Image

ROOT = Path("dataset")  # from project root

def check_folder(folder: Path):
    for img_path in folder.rglob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"BAD IMAGE: {img_path}  --  {e}")
            # img_path.unlink()   # uncomment to delete automatically (dangerous!)

if __name__ == "__main__":
    check_folder(ROOT)
    print("Done checking images.")