"""
Layer 5: Geospatial Engine Package
Exports:
  - GeospatialEngine (Five-stage Case A/B geotagger with PyProj WGS84 integration)
"""

from ai.geospatial.geotagger import GeospatialEngine

__all__ = ["GeospatialEngine"]
