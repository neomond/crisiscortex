"""Satellite imagery pipeline for Sentinel-2 and Landsat."""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio


def load_sentinel_tile(path: Path, bands: Optional[list[str]] = None) -> np.ndarray:
    """Load a Sentinel-2 tile and return stacked bands.
    
    Args:
        path: Path to .tif or .SAFE directory
        bands: List of band names (e.g., ["B02", "B03", "B04", "B08"])
               Defaults to RGB + NIR
    
    Returns:
        Array of shape (bands, height, width)
    """
    if bands is None:
        bands = ["B02", "B03", "B04", "B08"]  # RGB + NIR
    
    # TODO: Implement actual Sentinel-2 loading
    # This is a placeholder for the first commit
    raise NotImplementedError("Sentinel-2 loading not yet implemented")
