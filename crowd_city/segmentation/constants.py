"""Cityscapes label constants and the surface categories CROWD derives from them.

SegFormer checkpoints finetuned on Cityscapes emit the 19 evaluation classes in
``trainId`` order. Only a handful of them describe the ground a pedestrian can
stand on; the rest are either scenery or dynamic objects that may occlude the
footpoint of a pedestrian bounding box.
"""

from __future__ import annotations

from typing import Dict, FrozenSet


# Cityscapes trainId -> label name, in the order SegFormer emits them.
CITYSCAPES_TRAIN_ID_TO_NAME: Dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

ROAD_TRAIN_ID = 0
SIDEWALK_TRAIN_ID = 1
TERRAIN_TRAIN_ID = 9

# Classes that can sit between the camera and the ground a pedestrian stands on.
# A footpoint sample that lands on one of these tells us nothing about the
# surface, so it is discarded rather than counted as "not road".
OCCLUDING_TRAIN_IDS: FrozenSet[int] = frozenset(
    {11, 12, 13, 14, 15, 16, 17, 18}
)

# Classes that are never ground. A sample landing here means the footpoint is
# above the ground plane, which usually indicates a truncated or badly scaled
# box rather than a real surface reading.
NON_GROUND_TRAIN_IDS: FrozenSet[int] = frozenset({2, 3, 5, 6, 7, 10})

# Surface categories used by the crossing metrics.
SURFACE_ROAD = "road"
SURFACE_FOOTPATH = "footpath"
SURFACE_OTHER = "other"
SURFACE_UNKNOWN = "unknown"

SURFACE_TO_CODE: Dict[str, int] = {
    SURFACE_UNKNOWN: 0,
    SURFACE_ROAD: 1,
    SURFACE_FOOTPATH: 2,
    SURFACE_OTHER: 3,
}

CODE_TO_SURFACE: Dict[int, str] = {
    code: name for name, code in SURFACE_TO_CODE.items()
}


def surface_for_train_id(train_id: int) -> str:
    """Map one Cityscapes trainId onto a CROWD surface category."""
    if train_id == ROAD_TRAIN_ID:
        return SURFACE_ROAD
    if train_id in (SIDEWALK_TRAIN_ID, TERRAIN_TRAIN_ID):
        # Terrain is grouped with the footpath because for the purpose of this
        # metric it is simply "off the carriageway": a pedestrian standing on a
        # verge has not begun to cross.
        return SURFACE_FOOTPATH
    return SURFACE_OTHER
