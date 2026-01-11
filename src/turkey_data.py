"""
Turkey Earthquake Data Downloader
Downloads satellite imagery from Maxar Open Data via leafmap
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime

try:
    import leafmap
    LEAFMAP_AVAILABLE = True
except ImportError:
    LEAFMAP_AVAILABLE = False
    print("Warning: leafmap not available. Install with: pip install leafmap")

import numpy as np
from PIL import Image
import requests


# Turkey Earthquake Event Info
TURKEY_EARTHQUAKE = {
    'event_name': 'Kahramanmaras Turkey Earthquake',
    'date': '2023-02-06',
    'collection_id': 'Kahramanmaras-turkey-earthquake-23',
    'magnitude': 7.8,
    'center': {
        'lat': 37.174,
        'lon': 37.032
    },
    'bbox': {
        'min_lat': 36.5,
        'max_lat': 38.0,
        'min_lon': 35.5,
        'max_lon': 38.0
    }
}


class TurkeyEarthquakeDataLoader:
    """
    Load Turkey earthquake satellite imagery from Maxar Open Data.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize data loader.

        Args:
            output_dir: Directory to save downloaded images
        """
        if output_dir is None:
            output_dir = Path.home() / 'iitmhack' / 'data' / 'turkey'

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.event_date = datetime.strptime(TURKEY_EARTHQUAKE['date'], '%Y-%m-%d')

    def get_available_tiles(self) -> List[Dict]:
        """
        Get list of available tiles from Maxar Open Data.

        Returns:
            List of tile information dictionaries
        """
        if not LEAFMAP_AVAILABLE:
            print("leafmap required for tile discovery")
            return []

        try:
            # Get Maxar Open Data catalog
            gdf = leafmap.maxar_open_data(
                event=TURKEY_EARTHQUAKE['collection_id'],
                return_gdf=True
            )

            tiles = []
            for idx, row in gdf.iterrows():
                tile_info = {
                    'id': row.get('id', idx),
                    'datetime': row.get('datetime'),
                    'geometry': row.get('geometry'),
                    'assets': row.get('assets', {})
                }
                tiles.append(tile_info)

            return tiles

        except Exception as e:
            print(f"Error fetching tiles: {e}")
            return []

    def download_images(
        self,
        bbox: Tuple[float, float, float, float] = None,
        pre_date: str = '2023-02-05',
        post_date: str = '2023-02-07',
        output_format: str = 'tif'
    ) -> Dict[str, str]:
        """
        Download before and after images for the earthquake.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            pre_date: Date for pre-event image
            post_date: Date for post-event image
            output_format: Output format ('tif' or 'png')

        Returns:
            Dictionary with paths to downloaded images
        """
        if not LEAFMAP_AVAILABLE:
            print("leafmap not available. Using alternative method...")
            return self._download_sample_images()

        if bbox is None:
            # Default to city center area
            center = TURKEY_EARTHQUAKE['center']
            bbox = (
                center['lon'] - 0.1,
                center['lat'] - 0.1,
                center['lon'] + 0.1,
                center['lat'] + 0.1
            )

        try:
            # Download pre-event image
            pre_path = self.output_dir / f'turkey_before.{output_format}'

            # Use leafmap to download Maxar data
            # Note: This requires leafmap and proper API setup
            leafmap.maxar_open_data(
                event=TURKEY_EARTHQUAKE['collection_id'],
                end_date=pre_date,
                bbox=bbox,
                out_dir=str(self.output_dir),
                prefix='before_'
            )

            # Download post-event image
            post_path = self.output_dir / f'turkey_after.{output_format}'

            leafmap.maxar_open_data(
                event=TURKEY_EARTHQUAKE['collection_id'],
                start_date=post_date,
                bbox=bbox,
                out_dir=str(self.output_dir),
                prefix='after_'
            )

            return {
                'before': str(pre_path),
                'after': str(post_path)
            }

        except Exception as e:
            print(f"Error downloading via leafmap: {e}")
            return self._download_sample_images()

    def _download_sample_images(self) -> Dict[str, str]:
        """
        Download sample images from alternative sources or create synthetic.

        Returns:
            Dictionary with paths to images
        """
        print("Downloading sample Turkey earthquake images...")

        # Create sample images for demonstration
        # In actual use, you would download real satellite imagery

        H, W = 1024, 1024

        # Create synthetic before image (intact city)
        before = self._create_synthetic_city(H, W, damaged=False)
        before_path = self.output_dir / 'turkey_before.png'
        Image.fromarray(before).save(before_path)

        # Create synthetic after image (damaged city)
        after = self._create_synthetic_city(H, W, damaged=True)
        after_path = self.output_dir / 'turkey_after.png'
        Image.fromarray(after).save(after_path)

        print(f"Sample images saved to: {self.output_dir}")

        return {
            'before': str(before_path),
            'after': str(after_path)
        }

    def _create_synthetic_city(
        self,
        height: int,
        width: int,
        damaged: bool = False
    ) -> np.ndarray:
        """
        Create synthetic satellite-like city image for demonstration.

        Args:
            height: Image height
            width: Image width
            damaged: Whether to add damage artifacts

        Returns:
            RGB image array
        """
        np.random.seed(42 if not damaged else 43)

        # Base terrain (brownish)
        image = np.full((height, width, 3), [180, 160, 140], dtype=np.uint8)

        # Add some texture
        noise = np.random.randint(-20, 20, (height, width, 3))
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add buildings (gray/white blocks)
        n_buildings = 200

        for _ in range(n_buildings):
            x = np.random.randint(0, width - 50)
            y = np.random.randint(0, height - 50)
            w = np.random.randint(20, 60)
            h = np.random.randint(20, 60)

            # Building color
            brightness = np.random.randint(150, 220)
            color = [brightness, brightness, brightness]

            if damaged and np.random.random() < 0.3:
                # Add damage (darker, debris-like)
                color = [80, 70, 60]  # Rubble color
                # Add irregular edges
                for dy in range(h):
                    for dx in range(w):
                        if np.random.random() < 0.7:
                            py, px = y + dy, x + dx
                            if 0 <= py < height and 0 <= px < width:
                                image[py, px] = color

            else:
                image[y:y+h, x:x+w] = color

        # Add roads (dark lines)
        for i in range(5):
            y = np.random.randint(50, height - 50)
            image[y:y+10, :] = [60, 60, 60]

        for i in range(5):
            x = np.random.randint(50, width - 50)
            image[:, x:x+10] = [60, 60, 60]

        if damaged:
            # Add smoke/dust effect
            dust = np.random.randint(0, 30, (height, width, 3))
            image = np.clip(image.astype(np.int16) + dust, 0, 255).astype(np.uint8)

        return image

    def create_visualization_map(self, zoom: int = 12):
        """
        Create interactive map visualization.

        Args:
            zoom: Map zoom level
        """
        if not LEAFMAP_AVAILABLE:
            print("leafmap required for map visualization")
            return None

        try:
            m = leafmap.Map(
                center=[TURKEY_EARTHQUAKE['center']['lat'],
                        TURKEY_EARTHQUAKE['center']['lon']],
                zoom=zoom
            )

            # Add Maxar imagery layers
            m.add_basemap('Esri.WorldImagery')

            return m

        except Exception as e:
            print(f"Error creating map: {e}")
            return None


def download_turkey_earthquake_data(
    output_dir: str = None,
    use_leafmap: bool = True
) -> Dict[str, str]:
    """
    Convenience function to download Turkey earthquake data.

    Args:
        output_dir: Output directory
        use_leafmap: Whether to try using leafmap first

    Returns:
        Dictionary with paths to before/after images
    """
    if output_dir is None:
        output_dir = '/Users/mac/iitmhack/data/turkey'

    loader = TurkeyEarthquakeDataLoader(output_dir)

    if use_leafmap and LEAFMAP_AVAILABLE:
        return loader.download_images()
    else:
        return loader._download_sample_images()


def load_turkey_images(
    data_dir: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Turkey earthquake before/after images.

    Args:
        data_dir: Directory containing images

    Returns:
        Tuple of (before_img, after_img) arrays
    """
    if data_dir is None:
        data_dir = Path('/Users/mac/iitmhack/data/turkey')
    else:
        data_dir = Path(data_dir)

    before_path = data_dir / 'turkey_before.png'
    after_path = data_dir / 'turkey_after.png'

    # Check for TIFF files too
    if not before_path.exists():
        before_path = data_dir / 'turkey_before.tif'
    if not after_path.exists():
        after_path = data_dir / 'turkey_after.tif'

    if not before_path.exists() or not after_path.exists():
        print("Turkey images not found. Downloading...")
        download_turkey_earthquake_data(str(data_dir))

    before_img = np.array(Image.open(before_path).convert('RGB'))
    after_img = np.array(Image.open(after_path).convert('RGB'))

    return before_img, after_img


if __name__ == '__main__':
    print("Turkey Earthquake Data Loader")
    print("=" * 50)
    print(f"Event: {TURKEY_EARTHQUAKE['event_name']}")
    print(f"Date: {TURKEY_EARTHQUAKE['date']}")
    print(f"Magnitude: {TURKEY_EARTHQUAKE['magnitude']}")
    print(f"Location: {TURKEY_EARTHQUAKE['center']['lat']}, {TURKEY_EARTHQUAKE['center']['lon']}")
    print()

    # Download data
    print("Downloading/creating sample images...")
    paths = download_turkey_earthquake_data()

    print(f"\nBefore image: {paths['before']}")
    print(f"After image: {paths['after']}")

    # Load and verify
    before, after = load_turkey_images()
    print(f"\nLoaded images:")
    print(f"  Before shape: {before.shape}")
    print(f"  After shape: {after.shape}")
