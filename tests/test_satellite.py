"""Tests for satellite data pipeline."""

import numpy as np
import pytest

from crisiscortex.data.satellite import SentinelDownloader, get_crisis_regions


def test_get_crisis_regions():
    """Test that crisis regions are properly defined."""
    regions = get_crisis_regions()
    assert "sahel_mali" in regions
    assert len(regions["sahel_mali"]) == 4  # 4 coordinates
    
    # Check coordinates are valid
    for name, bbox in regions.items():
        min_lon, min_lat, max_lon, max_lat = bbox
        assert -180 <= min_lon <= 180
        assert -90 <= min_lat <= 90
        assert min_lon < max_lon
        assert min_lat < max_lat


def test_ndvi_change_computation():
    """Test NDVI change calculation."""
    # Create mock downloader
    downloader = SentinelDownloader("dummy_id", "dummy_secret")
    
    # Create synthetic NDVI arrays
    ndvi_before = np.array([[0.6, 0.5], [0.4, 0.3]])
    ndvi_after = np.array([[0.3, 0.4], [0.2, 0.1]])
    
    change = downloader.compute_ndvi_change(ndvi_before, ndvi_after)
    
    # Expected: all negative (degradation)
    expected = np.array([[-0.3, -0.1], [-0.2, -0.2]])
    np.testing.assert_array_almost_equal(change, expected)


def test_ndvi_change_with_nans():
    """Test that NaN values are handled correctly."""
    downloader = SentinelDownloader("dummy_id", "dummy_secret")
    
    ndvi_before = np.array([[0.6, np.nan], [0.4, 0.3]])
    ndvi_after = np.array([[0.3, 0.4], [np.nan, 0.1]])
    
    change = downloader.compute_ndvi_change(ndvi_before, ndvi_after)
    
    # (0,0): valid, (0,1): nan, (1,0): nan, (1,1): valid
    assert change[0, 0] == -0.3
    assert np.isnan(change[0, 1])
    assert np.isnan(change[1, 0])
    assert np.isclose(change[1, 1], -0.2)