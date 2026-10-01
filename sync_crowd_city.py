"""Copy the crowd-city analysis code into crowd_city/, changing only imports.

Usage: python sync_crowd_city.py /path/to/crowd-city

The listed files are copied unchanged except that imports of utils.* become
crowd_city.*, and crowd's helper_script import points at a stand-in. The
commit copied from is written to crowd_city/SOURCE_COMMIT. crowd_city/core/
holds hand-written stand-ins and is not touched.
"""

import os
import re
import subprocess
import sys

FILES = [
    "crossing/__init__.py",
    "crossing/detection.py",
    "crossing/metrics.py",
    "crossing/road_metrics.py",
    "crossing/road_crossing.py",
    "segmentation/__init__.py",
    "segmentation/constants.py",
    "segmentation/surface.py",
    "segmentation/segformer.py",
    "segmentation/pipeline.py",
    "segmentation/frames.py",
    "segmentation/store.py",
    "segmentation/crossing_pass.py",
    "analytics/csv_parallel.py",
]
REWRITES = [
    (re.compile(r"\butils\.(crossing|segmentation|analytics|core)\b"), r"crowd_city.\1"),
    (re.compile(r"^from helper_script import Youtube_Helper$", re.M),
     "from crowd_city.core.helper import Youtube_Helper"),
]


def main(source: str) -> None:
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crowd_city")
    for name in FILES:
        src = os.path.join(source, "utils", name)
        dst = os.path.join(target, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(src):
            # Upstream has no package __init__ for some folders.
            open(dst, "a").close()
            continue
        with open(src) as f:
            text = f.read()
        for pattern, replacement in REWRITES:
            text = pattern.sub(replacement, text)
        with open(dst, "w") as f:
            f.write(text)
    commit = subprocess.run(["git", "-C", source, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", source, "status", "--porcelain", "--", "utils"],
                           capture_output=True, text=True, check=True).stdout.strip()
    with open(os.path.join(target, "SOURCE_COMMIT"), "w") as f:
        f.write(commit + (" (with uncommitted changes)" if dirty else "") + "\n")
    print(f"Copied {len(FILES)} files from crowd-city {commit}{' (dirty)' if dirty else ''}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
