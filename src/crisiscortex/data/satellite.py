"""Satellite imagery pipeline for Sentinel-2 and Landsat."""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.mask import mask
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox,
    bbox_to_dimensions,
)


class SentinelDownloader:
    """Download and process Sentinel-2 imagery for crisis monitoring."""
    
    def __init__(self, client_id: str, client_secret: str):
        """Initialize with SentinelHub credentials.
        
        Args:
            client_id: SentinelHub OAuth client ID
            client_secret: SentinelHub OAuth client secret
        """
        self.config = SHConfig()
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret
        self.config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        
    def download_ndvi(
        self,
        bbox: Tuple[float, float, float, float],
        time_interval: Tuple[str, str],
        resolution: int = 60,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Download NDVI (Normalized Difference Vegetation Index) for a region.
        
        NDVI = (NIR - Red) / (NIR + Red)
        Values range from -1 to 1:
        - Negative: Water, clouds, snow
        - 0 to 0.2: Rock, sand, urban
        - 0.2 to 0.4: Sparse vegetation (stress indicator)
        - 0.4 to 0.6: Moderate vegetation
        - 0.6 to 0.9: Dense vegetation (healthy crops)
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            time_interval: ("YYYY-MM-DD", "YYYY-MM-DD")
            resolution: Pixel size in meters (10, 20, or 60)
            output_path: Where to save the .tif file
            
        Returns:
            NDVI array of shape (height, width)
        """
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B08", "SCL"],
                output: { bands: 1, sampleType: "FLOAT32" }
            };
        }
        function evaluatePixel(sample) {
            // SCL = Scene Classification. Filter out clouds (values 3, 8, 9)
            if (sample.SCL == 3 || sample.SCL == 8 || sample.SCL == 9) {
                return [NaN];
            }
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            return [ndvi];
        }
        """
        
        sentinel_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
        size = bbox_to_dimensions(sentinel_bbox, resolution=resolution)
        
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=sentinel_bbox,
            size=size,
            config=self.config,
        )
        
        data = request.get_data()[0]
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs="EPSG:4326",
                transform=rasterio.Affine.identity(),  # Simplified, should use proper transform
            ) as dst:
                dst.write(data, 1)
        
        return data
    
    def compute_ndvi_change(
        self,
        ndvi_before: np.ndarray,
        ndvi_after: np.ndarray,
    ) -> np.ndarray:
        """Compute NDVI change between two time periods.
        
        A significant drop in NDVI indicates:
        - Crop failure
        - Drought stress
        - Vegetation destruction (conflict, fire)
        
        Args:
            ndvi_before: Baseline NDVI array
            ndvi_after: Current NDVI array
            
        Returns:
            Change array (positive = improvement, negative = degradation)
        """
        # Handle NaN values (clouds)
        mask = np.isnan(ndvi_before) | np.isnan(ndvi_after)
        change = ndvi_after - ndvi_before
        change[mask] = np.nan
        return change


def load_local_tif(path: Path) -> np.ndarray:
    """Load a local GeoTIFF file.
    
    Args:
        path: Path to .tif file
        
    Returns:
        Numpy array of image data
    """
    with rasterio.open(path) as src:
        data = src.read(1)
    return data


def get_crisis_regions() -> dict:
    """Return predefined crisis-prone regions for monitoring.
    
    Returns:
        Dictionary of region_name: (min_lon, min_lat, max_lon, max_lat)
    """
    return {
        # Sahel region - frequent food insecurity
        "sahel_mali": (-5.0, 12.0, 15.0, 20.0),
        "sahel_niger": (0.0, 11.0, 16.0, 24.0),
        "sahel_chad": (13.0, 7.0, 24.0, 24.0),
        
        # Horn of Africa - drought + conflict
        "horn_ethiopia": (33.0, 3.0, 48.0, 18.0),
        "horn_somalia": (40.0, -2.0, 51.0, 12.0),
        
        # Yemen - conflict + famine
        "yemen": (42.0, 12.0, 54.0, 19.0),
        
        # South Sudan - conflict
        "south_sudan": (24.0, 3.0, 36.0, 13.0),
    }