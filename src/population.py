"""
Population Data Processor
Handles WorldPop population density data for urgency scoring
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import requests
from urllib.parse import urljoin

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.mask import mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. Some population features will be limited.")


# WorldPop data URLs (constrained resolution versions for faster download)
WORLDPOP_BASE_URL = "https://data.worldpop.org/GIS/Population/"


class PopulationDataProcessor:
    """
    Process population density data from WorldPop or similar sources.
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize processor.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'rapideye' / 'population'

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_worldpop(
        self,
        country_code: str = 'TUR',
        year: int = 2020,
        resolution: str = '1km'
    ) -> Optional[str]:
        """
        Download WorldPop population density data.

        Args:
            country_code: ISO3 country code (e.g., 'TUR' for Turkey)
            year: Year of data
            resolution: Resolution ('100m' or '1km')

        Returns:
            Path to downloaded file or None if failed
        """
        # Construct URL (this is a simplified example - actual URLs vary)
        filename = f"{country_code.lower()}_ppp_{year}_{resolution}_Constrained.tif"
        url = f"https://data.worldpop.org/GIS/Population/Global_{year}/{resolution}/Constrained/{country_code}/{filename}"

        output_path = self.cache_dir / filename

        if output_path.exists():
            print(f"Using cached file: {output_path}")
            return str(output_path)

        print(f"Downloading population data from WorldPop...")
        print(f"URL: {url}")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded to: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"Error downloading population data: {e}")
            print("You may need to manually download from: https://www.worldpop.org/geodata/listing?id=75")
            return None

    def load_population_raster(
        self,
        raster_path: str,
        bbox: Tuple[float, float, float, float] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load population raster, optionally cropping to bbox.

        Args:
            raster_path: Path to GeoTIFF file
            bbox: Optional bounding box (min_lon, min_lat, max_lon, max_lat)

        Returns:
            Tuple of (population array, metadata dict)
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required to load population rasters")

        with rasterio.open(raster_path) as src:
            if bbox is not None:
                # Create geometry for bbox
                from shapely.geometry import box
                geom = box(bbox[0], bbox[1], bbox[2], bbox[3])

                # Crop to bbox
                try:
                    data, transform = mask(src, [geom], crop=True)
                    data = data[0]  # Get first band
                except Exception as e:
                    print(f"Error cropping raster: {e}")
                    data = src.read(1)
                    transform = src.transform
            else:
                data = src.read(1)
                transform = src.transform

            # Handle nodata values
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, 0, data)

            # Clip negative values
            data = np.maximum(data, 0)

            metadata = {
                'crs': str(src.crs),
                'transform': transform,
                'shape': data.shape,
                'bounds': src.bounds
            }

        return data, metadata

    def resample_to_image(
        self,
        population: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resample population data to match image size.

        Args:
            population: Population density array
            target_size: Target (height, width)

        Returns:
            Resampled population array
        """
        from scipy.ndimage import zoom

        # Calculate zoom factors
        zoom_h = target_size[0] / population.shape[0]
        zoom_w = target_size[1] / population.shape[1]

        # Use bilinear interpolation
        resampled = zoom(population, (zoom_h, zoom_w), order=1)

        return resampled

    def estimate_affected_population(
        self,
        population: np.ndarray,
        damage_map: np.ndarray,
        pixel_area_km2: float = None
    ) -> Dict:
        """
        Estimate population affected by damage.

        Args:
            population: Population density array (people per pixel)
            damage_map: Damage predictions (H, W) with values 0-3
            pixel_area_km2: Area of each pixel in km² (for density conversion)

        Returns:
            Dictionary with affected population statistics
        """
        # Ensure same shape
        if population.shape != damage_map.shape:
            population = self.resample_to_image(population, damage_map.shape)

        stats = {
            'total_affected': 0,
            'by_damage_level': {},
            'density_stats': {}
        }

        damage_labels = {
            0: 'no_damage',
            1: 'minor_damage',
            2: 'major_damage',
            3: 'destroyed'
        }

        for level, label in damage_labels.items():
            mask = damage_map == level
            affected = population[mask].sum()
            stats['by_damage_level'][label] = int(affected)

            if level > 0:
                stats['total_affected'] += affected

        stats['total_affected'] = int(stats['total_affected'])

        # Density statistics in damaged areas
        damaged_mask = damage_map > 0
        if damaged_mask.any():
            damaged_pop = population[damaged_mask]
            stats['density_stats'] = {
                'mean': float(damaged_pop.mean()),
                'max': float(damaged_pop.max()),
                'total': float(damaged_pop.sum())
            }

        return stats


def create_synthetic_population(
    damage_map: np.ndarray,
    base_density: float = 100,
    urban_multiplier: float = 5
) -> np.ndarray:
    """
    Create synthetic population density when real data is unavailable.
    Uses building density from damage map as proxy.

    Args:
        damage_map: Damage predictions (H, W)
        base_density: Base population density (people per km²)
        urban_multiplier: Multiplier for areas with buildings

    Returns:
        Synthetic population density array
    """
    from scipy.ndimage import gaussian_filter

    H, W = damage_map.shape

    # Start with base density
    population = np.ones((H, W), dtype=np.float32) * base_density

    # Increase density where there are buildings (any damage class > 0 implies building)
    building_mask = damage_map > 0
    population[building_mask] *= urban_multiplier

    # Smooth to create more realistic distribution
    population = gaussian_filter(population, sigma=10)

    return population


def calculate_population_score(
    population: np.ndarray,
    damage_map: np.ndarray
) -> np.ndarray:
    """
    Calculate normalized population score for urgency calculation.

    Args:
        population: Population density array
        damage_map: Damage predictions

    Returns:
        Normalized population score (0-1)
    """
    # Only consider population in damaged areas
    score = population.copy()
    score[damage_map == 0] = 0

    # Normalize to 0-1
    if score.max() > 0:
        score = score / score.max()

    return score


# Turkey earthquake specific functions
def get_turkey_earthquake_bbox():
    """Get bounding box for Turkey earthquake affected area."""
    # Kahramanmaras province approximate bounds
    return {
        'name': 'Kahramanmaras',
        'bbox': (36.5, 37.3, 37.4, 37.9),  # (min_lon, min_lat, max_lon, max_lat)
        'center': (37.0, 37.58)  # (lon, lat)
    }


def load_turkey_population(
    data_path: str = None,
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Load or create population data for Turkey earthquake region.

    Args:
        data_path: Path to population raster (optional)
        target_size: Target image size

    Returns:
        Population density array
    """
    processor = PopulationDataProcessor()

    if data_path and Path(data_path).exists():
        try:
            bbox_info = get_turkey_earthquake_bbox()
            population, _ = processor.load_population_raster(
                data_path,
                bbox=bbox_info['bbox']
            )
            return processor.resample_to_image(population, target_size)
        except Exception as e:
            print(f"Error loading population data: {e}")

    # Create synthetic population data
    print("Using synthetic population data (WorldPop data not available)")
    # Create a realistic-looking synthetic population
    np.random.seed(42)

    H, W = target_size
    population = np.zeros((H, W), dtype=np.float32)

    # Add urban centers (higher density)
    centers = [
        (H // 3, W // 2, 50),     # City center 1
        (2 * H // 3, W // 3, 30), # City center 2
        (H // 2, 2 * W // 3, 40), # City center 3
    ]

    from scipy.ndimage import gaussian_filter

    for cy, cx, intensity in centers:
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        population += intensity * np.exp(-dist ** 2 / (2 * 50 ** 2))

    # Add some noise
    population += np.random.uniform(5, 15, (H, W))

    # Smooth
    population = gaussian_filter(population, sigma=5)

    return population


if __name__ == '__main__':
    print("Testing Population Data Processor...")

    # Test synthetic population
    print("\nCreating synthetic population data...")
    damage_map = np.random.randint(0, 4, (512, 512))
    population = create_synthetic_population(damage_map)
    print(f"Population shape: {population.shape}")
    print(f"Population range: {population.min():.1f} - {population.max():.1f}")

    # Test affected population calculation
    processor = PopulationDataProcessor()
    stats = processor.estimate_affected_population(population, damage_map)
    print(f"\nAffected population estimate: {stats['total_affected']:,}")
    print("By damage level:")
    for level, count in stats['by_damage_level'].items():
        print(f"  {level}: {count:,}")

    # Test Turkey-specific functions
    print("\nTurkey earthquake region:")
    bbox_info = get_turkey_earthquake_bbox()
    print(f"  Region: {bbox_info['name']}")
    print(f"  BBox: {bbox_info['bbox']}")
