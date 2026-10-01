"""Exact copy of the crowd-city analysis code (https://github.com/crowd-dataset/crowd-city).

Files under crossing/, segmentation/ and analytics/ are copied unchanged from
the crowd-city commit in SOURCE_COMMIT; only their imports were rewritten from
utils.* to crowd_city.*. core/ holds small stand-ins for crowd-city modules
that read its YouTube mapping. Refresh with: python sync_crowd_city.py <path>.
"""
