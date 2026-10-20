"""Export the offline occupancy grids as ROS map_server maps (PGM + YAML).

    python tools/export_maps.py

The exported maps are used by ``map_server`` / AMCL and by SLAM Toolbox as a
reference.  Occupancy semantics of PGM maps: 0 = occupied (black),
254 = free (white), 205 = unknown (grey).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from navfollow.maps import MAP_NAMES, load_map  # noqa: E402

OUT_DIR = REPO_ROOT / "ros2_ws" / "src" / "follow_robot" / "maps"


def export_map(name: str, out_dir: Path) -> None:
    grid = load_map(name)
    image = np.full(grid.occupancy.shape, 254, dtype=np.uint8)
    image[grid.occupancy] = 0
    # map_server expects (row 0 = top), our grid row 0 = bottom -> flip
    image = np.flipud(image)
    pgm_path = out_dir / f"{name}.pgm"
    with open(pgm_path, "wb") as fh:
        fh.write(b"P5\n")
        fh.write(f"# {name} exported from navfollow/{name}\n".encode())
        fh.write(f"{grid.width} {grid.height}\n255\n".encode())
        fh.write(image.tobytes())
    yaml_path = out_dir / f"{name}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"image: {name}.pgm",
                f"resolution: {grid.resolution}",
                f"origin: [{grid.origin[0]:.3f}, {grid.origin[1]:.3f}, 0.000]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"exported {pgm_path} ({grid.width}x{grid.height} @ {grid.resolution} m)")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in MAP_NAMES:
        export_map(name, OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
