"""Stand-in for crowd-city's utils/core/grouping.py.

Upstream, locality_country_wrapper regroups {video: {track: value}} by the
city and time of day found in mapping.csv, dropping videos it cannot find.
Here the city is known from the data folder, so values are passed through
under one outer key and analysis.py groups them by city itself.
"""

ALL = "all"


class Grouping:
    def locality_country_wrapper(self, input_dict, mapping=None, show_progress: bool = False):
        return {ALL: dict(input_dict)} if input_dict else {}
