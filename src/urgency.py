"""
Urgency Scoring System for Disaster Response Prioritization
Key differentiator: "We don't just detect damage - we tell responders WHERE to help FIRST"

This module calculates urgency scores based on:
- Damage severity (40%)
- Population density (35%)
- Critical infrastructure proximity (25%)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt
from typing import Dict, Tuple, Optional, List
import cv2


# Urgency zone thresholds
URGENCY_ZONES = {
    'CRITICAL': {'min': 80, 'max': 100, 'color': (255, 0, 0)},      # Red
    'HIGH': {'min': 60, 'max': 80, 'color': (255, 128, 0)},         # Orange
    'MEDIUM': {'min': 40, 'max': 60, 'color': (255, 255, 0)},       # Yellow
    'LOW': {'min': 20, 'max': 40, 'color': (144, 238, 144)},        # Light Green
    'MINIMAL': {'min': 0, 'max': 20, 'color': (0, 128, 0)}          # Green
}

# Damage severity weights
DAMAGE_WEIGHTS = {
    0: 0.0,   # No damage
    1: 0.3,   # Minor damage
    2: 0.7,   # Major damage
    3: 1.0    # Destroyed
}


class UrgencyCalculator:
    """
    Calculate urgency scores for disaster response prioritization.
    """

    def __init__(
        self,
        damage_weight: float = 0.40,
        population_weight: float = 0.35,
        infrastructure_weight: float = 0.25,
        kernel_size: int = 31
    ):
        """
        Initialize urgency calculator.

        Args:
            damage_weight: Weight for damage severity (0-1)
            population_weight: Weight for population density (0-1)
            infrastructure_weight: Weight for infrastructure proximity (0-1)
            kernel_size: Size of smoothing kernel
        """
        self.damage_weight = damage_weight
        self.population_weight = population_weight
        self.infrastructure_weight = infrastructure_weight
        self.kernel_size = kernel_size

        # Validate weights sum to 1
        total = damage_weight + population_weight + infrastructure_weight
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1, got {total}"

    def calculate_damage_severity(self, damage_map: np.ndarray) -> np.ndarray:
        """
        Convert damage map to weighted severity scores.

        Args:
            damage_map: Damage predictions (H, W) with values 0-3

        Returns:
            Severity map (H, W) with values 0-1
        """
        severity = np.zeros_like(damage_map, dtype=np.float32)

        for class_id, weight in DAMAGE_WEIGHTS.items():
            severity[damage_map == class_id] = weight

        # Apply spatial smoothing to create continuous zones
        severity = gaussian_filter(severity, sigma=self.kernel_size // 4)

        return severity

    def calculate_damage_density(
        self,
        damage_map: np.ndarray,
        window_size: int = 64
    ) -> np.ndarray:
        """
        Calculate local damage density (concentration of damage).

        Args:
            damage_map: Damage predictions (H, W)
            window_size: Size of local window for density calculation

        Returns:
            Density map (H, W) with values 0-1
        """
        # Binary mask of any damage
        damage_binary = (damage_map > 0).astype(np.float32)

        # Calculate local density using uniform filter
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        density = ndimage.convolve(damage_binary, kernel, mode='constant')

        # Normalize to 0-1
        if density.max() > 0:
            density = density / density.max()

        return density

    def estimate_population_from_buildings(
        self,
        damage_map: np.ndarray,
        avg_people_per_building: int = 5
    ) -> Tuple[np.ndarray, int]:
        """
        Estimate population at risk from building damage map.
        Uses building count as proxy for population when population data unavailable.

        Args:
            damage_map: Damage predictions (H, W)
            avg_people_per_building: Estimated people per building

        Returns:
            population_proxy: Normalized population proxy map
            estimated_affected: Estimated number of affected people
        """
        # Count buildings (connected components in damage map)
        damage_binary = (damage_map > 0).astype(np.uint8)

        # Find connected components (approximate buildings)
        num_labels, labels = cv2.connectedComponents(damage_binary)
        building_count = num_labels - 1  # Exclude background

        # Estimate affected population
        estimated_affected = building_count * avg_people_per_building

        # Create density map from building locations
        building_density = gaussian_filter(damage_binary.astype(np.float32), sigma=20)
        if building_density.max() > 0:
            building_density = building_density / building_density.max()

        return building_density, estimated_affected

    def integrate_population_data(
        self,
        damage_map: np.ndarray,
        population_raster: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Integrate actual population density data (e.g., from WorldPop).

        Args:
            damage_map: Damage predictions (H, W)
            population_raster: Population density raster (H, W)

        Returns:
            population_score: Normalized population score map
            estimated_affected: Estimated affected population
        """
        # Resize population raster to match damage map if needed
        if population_raster.shape != damage_map.shape:
            population_raster = cv2.resize(
                population_raster,
                (damage_map.shape[1], damage_map.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Calculate affected population
        damage_mask = damage_map > 0
        estimated_affected = int(population_raster[damage_mask].sum())

        # Normalize population for scoring
        population_score = population_raster.copy()
        if population_score.max() > 0:
            population_score = population_score / population_score.max()

        return population_score, estimated_affected

    def calculate_infrastructure_score(
        self,
        damage_map: np.ndarray,
        infrastructure_points: List[Tuple[int, int]] = None,
        max_distance: int = 100
    ) -> np.ndarray:
        """
        Calculate infrastructure proximity score.

        Args:
            damage_map: Damage predictions (H, W)
            infrastructure_points: List of (row, col) for critical infrastructure
            max_distance: Maximum distance to consider (in pixels)

        Returns:
            Infrastructure score map (H, W) with values 0-1
        """
        H, W = damage_map.shape

        if infrastructure_points is None or len(infrastructure_points) == 0:
            # If no infrastructure data, use large buildings as proxy
            # Large connected components likely represent important structures
            damage_binary = (damage_map > 0).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                damage_binary, connectivity=8
            )

            # Find large structures (top 10% by area)
            areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
            if len(areas) > 0:
                threshold = np.percentile(areas, 90)
                large_indices = np.where(areas >= threshold)[0] + 1

                # Get centroids of large structures
                infrastructure_points = [
                    (int(centroids[i][1]), int(centroids[i][0]))
                    for i in large_indices
                ]

        if infrastructure_points is None or len(infrastructure_points) == 0:
            # No infrastructure detected, return uniform score
            return np.ones((H, W), dtype=np.float32) * 0.5

        # Create infrastructure mask
        infra_mask = np.zeros((H, W), dtype=np.uint8)
        for row, col in infrastructure_points:
            if 0 <= row < H and 0 <= col < W:
                infra_mask[row, col] = 1

        # Calculate distance transform
        distance = distance_transform_edt(1 - infra_mask)

        # Convert to proximity score (closer = higher)
        proximity = 1 - np.clip(distance / max_distance, 0, 1)

        # Only score damaged areas near infrastructure
        damage_mask = damage_map > 0
        proximity = proximity * damage_mask

        return proximity

    def calculate_urgency_score(
        self,
        damage_map: np.ndarray,
        population_raster: np.ndarray = None,
        infrastructure_points: List[Tuple[int, int]] = None
    ) -> Dict:
        """
        Calculate comprehensive urgency score.

        Args:
            damage_map: Damage predictions (H, W) with values 0-3
            population_raster: Optional population density data (H, W)
            infrastructure_points: Optional list of critical infrastructure locations

        Returns:
            Dictionary with:
                - urgency_map: Urgency scores 0-100 (H, W)
                - damage_score: Damage component (H, W)
                - population_score: Population component (H, W)
                - infrastructure_score: Infrastructure component (H, W)
                - estimated_affected: Estimated affected population
                - zone_stats: Statistics per urgency zone
        """
        H, W = damage_map.shape

        # 1. Calculate damage severity
        damage_score = self.calculate_damage_severity(damage_map)

        # Add damage density
        damage_density = self.calculate_damage_density(damage_map)
        damage_score = 0.7 * damage_score + 0.3 * damage_density

        # 2. Calculate population score
        if population_raster is not None:
            population_score, estimated_affected = self.integrate_population_data(
                damage_map, population_raster
            )
        else:
            # Use building density as proxy
            population_score, estimated_affected = self.estimate_population_from_buildings(
                damage_map
            )

        # 3. Calculate infrastructure proximity
        infrastructure_score = self.calculate_infrastructure_score(
            damage_map, infrastructure_points
        )

        # 4. Combine scores with weights
        urgency_score = (
            self.damage_weight * damage_score +
            self.population_weight * population_score +
            self.infrastructure_weight * infrastructure_score
        )

        # Scale to 0-100
        urgency_map = (urgency_score * 100).clip(0, 100).astype(np.float32)

        # 5. Calculate zone statistics
        zone_stats = self._calculate_zone_stats(urgency_map, damage_map)

        return {
            'urgency_map': urgency_map,
            'damage_score': damage_score,
            'population_score': population_score,
            'infrastructure_score': infrastructure_score,
            'estimated_affected': estimated_affected,
            'zone_stats': zone_stats
        }

    def _calculate_zone_stats(
        self,
        urgency_map: np.ndarray,
        damage_map: np.ndarray
    ) -> Dict:
        """Calculate statistics for each urgency zone."""
        stats = {}
        total_damaged = np.sum(damage_map > 0)

        for zone_name, zone_info in URGENCY_ZONES.items():
            mask = (urgency_map >= zone_info['min']) & (urgency_map < zone_info['max'])
            # Only count damaged areas
            mask = mask & (damage_map > 0)

            pixel_count = np.sum(mask)
            percentage = (pixel_count / total_damaged * 100) if total_damaged > 0 else 0

            stats[zone_name] = {
                'pixel_count': int(pixel_count),
                'percentage': round(percentage, 2),
                'color': zone_info['color']
            }

        return stats


def create_urgency_heatmap(urgency_map: np.ndarray) -> np.ndarray:
    """
    Create colored heatmap visualization of urgency scores.

    Args:
        urgency_map: Urgency scores 0-100 (H, W)

    Returns:
        Colored heatmap (H, W, 3)
    """
    # Normalize to 0-255
    normalized = (urgency_map / 100 * 255).astype(np.uint8)

    # Apply colormap (blue -> red)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


def create_zone_visualization(
    urgency_map: np.ndarray,
    damage_map: np.ndarray
) -> np.ndarray:
    """
    Create visualization with discrete urgency zones.

    Args:
        urgency_map: Urgency scores 0-100 (H, W)
        damage_map: Damage predictions (H, W)

    Returns:
        Zone visualization (H, W, 3)
    """
    H, W = urgency_map.shape
    visualization = np.zeros((H, W, 3), dtype=np.uint8)

    # Only color damaged areas
    damage_mask = damage_map > 0

    for zone_name, zone_info in URGENCY_ZONES.items():
        mask = (
            (urgency_map >= zone_info['min']) &
            (urgency_map < zone_info['max']) &
            damage_mask
        )
        visualization[mask] = zone_info['color']

    return visualization


def generate_response_priorities(
    urgency_results: Dict,
    top_n: int = 10,
    grid_size: int = 64
) -> List[Dict]:
    """
    Generate prioritized response locations from urgency map.

    Args:
        urgency_results: Output from UrgencyCalculator.calculate_urgency_score()
        top_n: Number of top priority locations to return
        grid_size: Size of grid cells for grouping

    Returns:
        List of priority locations with coordinates and scores
    """
    urgency_map = urgency_results['urgency_map']
    H, W = urgency_map.shape

    priorities = []

    # Divide into grid cells
    for i in range(0, H, grid_size):
        for j in range(0, W, grid_size):
            # Get cell
            cell = urgency_map[i:min(i+grid_size, H), j:min(j+grid_size, W)]

            if cell.size == 0:
                continue

            avg_urgency = np.mean(cell)
            max_urgency = np.max(cell)

            if avg_urgency > 20:  # Only include cells with significant urgency
                priorities.append({
                    'center_row': i + grid_size // 2,
                    'center_col': j + grid_size // 2,
                    'avg_urgency': round(avg_urgency, 1),
                    'max_urgency': round(max_urgency, 1),
                    'bounds': {
                        'min_row': i,
                        'max_row': min(i+grid_size, H),
                        'min_col': j,
                        'max_col': min(j+grid_size, W)
                    }
                })

    # Sort by average urgency
    priorities.sort(key=lambda x: x['avg_urgency'], reverse=True)

    # Assign priority ranks
    for rank, p in enumerate(priorities[:top_n], 1):
        p['priority_rank'] = rank
        # Determine zone
        for zone_name, zone_info in URGENCY_ZONES.items():
            if zone_info['min'] <= p['avg_urgency'] < zone_info['max']:
                p['zone'] = zone_name
                break

    return priorities[:top_n]


if __name__ == '__main__':
    # Test urgency calculator
    print("Testing Urgency Calculator...")

    # Create mock damage map
    H, W = 512, 512
    damage_map = np.zeros((H, W), dtype=np.uint8)

    # Add some damage regions
    damage_map[100:200, 100:200] = 3  # Destroyed area
    damage_map[150:250, 150:250] = 2  # Major damage
    damage_map[200:350, 200:350] = 1  # Minor damage
    damage_map[300:400, 50:150] = 3   # Another destroyed area

    # Calculate urgency
    calculator = UrgencyCalculator()
    results = calculator.calculate_urgency_score(damage_map)

    print(f"\nUrgency Map shape: {results['urgency_map'].shape}")
    print(f"Estimated affected population: {results['estimated_affected']}")
    print("\nZone Statistics:")
    for zone, stats in results['zone_stats'].items():
        print(f"  {zone}: {stats['percentage']:.1f}% of damaged area")

    # Get priorities
    priorities = generate_response_priorities(results, top_n=5)
    print("\nTop 5 Priority Response Locations:")
    for p in priorities:
        print(f"  Rank {p['priority_rank']}: ({p['center_row']}, {p['center_col']}) "
              f"- Urgency: {p['avg_urgency']} ({p.get('zone', 'N/A')})")
