"""
Infrastructure Data Fetcher
Fetches critical infrastructure data from OpenStreetMap via Overpass API
"""

import requests
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Infrastructure types to query
INFRASTRUCTURE_TYPES = {
    'hospitals': {
        'query': 'amenity=hospital',
        'priority': 1.0,
        'icon': 'hospital'
    },
    'fire_stations': {
        'query': 'amenity=fire_station',
        'priority': 0.9,
        'icon': 'fire'
    },
    'police': {
        'query': 'amenity=police',
        'priority': 0.8,
        'icon': 'shield'
    },
    'schools': {
        'query': 'amenity=school',
        'priority': 0.7,
        'icon': 'school'
    },
    'universities': {
        'query': 'amenity=university',
        'priority': 0.6,
        'icon': 'graduation'
    },
    'shelters': {
        'query': 'amenity=shelter',
        'priority': 0.85,
        'icon': 'home'
    },
    'pharmacies': {
        'query': 'amenity=pharmacy',
        'priority': 0.5,
        'icon': 'pharmacy'
    },
    'power_plants': {
        'query': 'power=plant',
        'priority': 0.95,
        'icon': 'bolt'
    },
    'water_towers': {
        'query': 'man_made=water_tower',
        'priority': 0.85,
        'icon': 'water'
    }
}


@dataclass
class InfrastructurePoint:
    """Represents a single infrastructure point."""
    lat: float
    lon: float
    type: str
    name: str
    priority: float
    osm_id: int


class InfrastructureFetcher:
    """
    Fetch critical infrastructure data from OpenStreetMap.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize fetcher.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.cache = {}

    def build_query(
        self,
        bbox: Tuple[float, float, float, float],
        infra_types: List[str] = None
    ) -> str:
        """
        Build Overpass QL query.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
            infra_types: List of infrastructure types to query (None for all)

        Returns:
            Overpass QL query string
        """
        if infra_types is None:
            infra_types = list(INFRASTRUCTURE_TYPES.keys())

        min_lat, min_lon, max_lat, max_lon = bbox
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        # Build query parts
        query_parts = []
        for infra_type in infra_types:
            if infra_type in INFRASTRUCTURE_TYPES:
                tag_query = INFRASTRUCTURE_TYPES[infra_type]['query']
                key, value = tag_query.split('=')
                query_parts.append(f'node["{key}"="{value}"]({bbox_str});')
                query_parts.append(f'way["{key}"="{value}"]({bbox_str});')
                query_parts.append(f'relation["{key}"="{value}"]({bbox_str});')

        query = f"""
        [out:json][timeout:{self.timeout}];
        (
            {' '.join(query_parts)}
        );
        out center;
        """

        return query

    def fetch(
        self,
        bbox: Tuple[float, float, float, float],
        infra_types: List[str] = None
    ) -> List[InfrastructurePoint]:
        """
        Fetch infrastructure data for a bounding box.

        Args:
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
            infra_types: List of infrastructure types (None for all)

        Returns:
            List of InfrastructurePoint objects
        """
        # Check cache
        cache_key = f"{bbox}_{infra_types}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = self.build_query(bbox, infra_types)

        try:
            response = requests.post(
                OVERPASS_URL,
                data={'data': query},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error fetching infrastructure data: {e}")
            return []

        # Parse results
        points = []
        for element in data.get('elements', []):
            # Get coordinates
            if element['type'] == 'node':
                lat, lon = element['lat'], element['lon']
            elif 'center' in element:
                lat, lon = element['center']['lat'], element['center']['lon']
            else:
                continue

            # Get name
            tags = element.get('tags', {})
            name = tags.get('name', 'Unknown')

            # Determine infrastructure type
            infra_type = None
            priority = 0.5

            for itype, info in INFRASTRUCTURE_TYPES.items():
                key, value = info['query'].split('=')
                if tags.get(key) == value:
                    infra_type = itype
                    priority = info['priority']
                    break

            if infra_type:
                points.append(InfrastructurePoint(
                    lat=lat,
                    lon=lon,
                    type=infra_type,
                    name=name,
                    priority=priority,
                    osm_id=element['id']
                ))

        # Cache results
        self.cache[cache_key] = points

        return points

    def to_pixel_coords(
        self,
        points: List[InfrastructurePoint],
        bbox: Tuple[float, float, float, float],
        img_size: Tuple[int, int]
    ) -> List[Tuple[int, int, float]]:
        """
        Convert infrastructure points to pixel coordinates.

        Args:
            points: List of InfrastructurePoint objects
            bbox: Geographic bounding box (min_lat, min_lon, max_lat, max_lon)
            img_size: Image size (height, width)

        Returns:
            List of (row, col, priority) tuples
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        height, width = img_size

        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        pixel_coords = []
        for point in points:
            # Convert to normalized coords
            norm_lat = (point.lat - min_lat) / lat_range if lat_range > 0 else 0.5
            norm_lon = (point.lon - min_lon) / lon_range if lon_range > 0 else 0.5

            # Convert to pixel coords (note: lat is inverted)
            row = int((1 - norm_lat) * height)
            col = int(norm_lon * width)

            # Clip to bounds
            row = max(0, min(height - 1, row))
            col = max(0, min(width - 1, col))

            pixel_coords.append((row, col, point.priority))

        return pixel_coords


def fetch_infrastructure_for_turkey_earthquake() -> List[InfrastructurePoint]:
    """
    Fetch infrastructure data for Turkey earthquake region (Kahramanmaras).

    Returns:
        List of infrastructure points
    """
    # Kahramanmaras region approximate bbox
    # Center: 37.58°N, 36.93°E
    bbox = (37.3, 36.5, 37.9, 37.4)  # (min_lat, min_lon, max_lat, max_lon)

    fetcher = InfrastructureFetcher(timeout=60)

    # Fetch critical infrastructure
    critical_types = ['hospitals', 'fire_stations', 'police', 'shelters', 'schools']
    points = fetcher.fetch(bbox, critical_types)

    print(f"Found {len(points)} infrastructure points in Kahramanmaras region")

    # Summary by type
    type_counts = {}
    for p in points:
        type_counts[p.type] = type_counts.get(p.type, 0) + 1

    for itype, count in sorted(type_counts.items()):
        print(f"  {itype}: {count}")

    return points


def create_infrastructure_raster(
    points: List[Tuple[int, int, float]],
    img_size: Tuple[int, int],
    influence_radius: int = 50
) -> np.ndarray:
    """
    Create a raster of infrastructure influence.

    Args:
        points: List of (row, col, priority) tuples
        img_size: Output raster size (height, width)
        influence_radius: Radius of influence for each point

    Returns:
        Infrastructure influence raster (H, W)
    """
    from scipy.ndimage import gaussian_filter

    height, width = img_size
    raster = np.zeros((height, width), dtype=np.float32)

    for row, col, priority in points:
        # Add point with priority weighting
        raster[row, col] = priority

    # Apply Gaussian blur to create influence zones
    raster = gaussian_filter(raster, sigma=influence_radius / 2)

    # Normalize
    if raster.max() > 0:
        raster = raster / raster.max()

    return raster


def get_affected_infrastructure(
    damage_map: np.ndarray,
    points: List[Tuple[int, int, float]],
    point_names: List[str] = None
) -> List[Dict]:
    """
    Identify infrastructure affected by damage.

    Args:
        damage_map: Damage predictions (H, W)
        points: List of (row, col, priority) pixel coordinates
        point_names: Optional list of infrastructure names

    Returns:
        List of affected infrastructure with damage levels
    """
    affected = []

    for i, (row, col, priority) in enumerate(points):
        # Check damage at infrastructure location (with small radius)
        H, W = damage_map.shape
        r_min, r_max = max(0, row - 5), min(H, row + 5)
        c_min, c_max = max(0, col - 5), min(W, col + 5)

        local_damage = damage_map[r_min:r_max, c_min:c_max]
        max_damage = np.max(local_damage) if local_damage.size > 0 else 0

        if max_damage > 0:
            damage_labels = {0: 'None', 1: 'Minor', 2: 'Major', 3: 'Destroyed'}
            affected.append({
                'index': i,
                'name': point_names[i] if point_names else f'Infrastructure {i}',
                'location': (row, col),
                'priority': priority,
                'damage_level': int(max_damage),
                'damage_label': damage_labels.get(max_damage, 'Unknown')
            })

    # Sort by priority (most critical first)
    affected.sort(key=lambda x: x['priority'], reverse=True)

    return affected


if __name__ == '__main__':
    print("Testing Infrastructure Fetcher...")
    print("Fetching data for Turkey earthquake region...")

    points = fetch_infrastructure_for_turkey_earthquake()

    if points:
        print(f"\nFirst 5 infrastructure points:")
        for p in points[:5]:
            print(f"  {p.name} ({p.type}) - Priority: {p.priority}")
