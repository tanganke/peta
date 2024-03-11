"""
remove *.temp if * exists
"""
import os
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).parent
    for path in root.glob("*.temp"):
        basename = path.stem
        print(basename)
        if (root / basename).exists():
            print("remove", path)
            path.rename(root / ".trash" / f"{basename}.temp")
