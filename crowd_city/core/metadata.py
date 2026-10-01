"""Stand-in for crowd-city's utils/core/metadata.py.

crowd-city looks every video up in its YouTube mapping.csv (country, city,
stature, fps, ...). The local dashcam videos are not in that mapping, so every
lookup misses, exactly as it does upstream for a video id absent from the
mapping: callers then fall back to the fps in the file name, no stature
correction (which is disabled upstream by default anyway) and no rider height.
"""


class MetaData:
    def find_values_with_video_id(self, df, key):
        return None
