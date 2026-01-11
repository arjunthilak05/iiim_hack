"""
Visualization Module for RapidEye
Creates compelling visualizations for disaster damage and urgency analysis
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


# Color schemes
DAMAGE_COLORS = {
    0: (46, 204, 113),    # Green - No damage
    1: (241, 196, 15),    # Yellow - Minor damage
    2: (230, 126, 34),    # Orange - Major damage
    3: (231, 76, 60)      # Red - Destroyed
}

DAMAGE_LABELS = {
    0: 'No Damage',
    1: 'Minor Damage',
    2: 'Major Damage',
    3: 'Destroyed'
}

URGENCY_COLORMAP = 'RdYlGn_r'  # Red-Yellow-Green reversed (red = high urgency)


class DisasterVisualizer:
    """
    Create visualizations for disaster damage analysis.
    """

    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def create_before_after_comparison(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
        damage_map: np.ndarray = None,
        title: str = "Before / After Comparison"
    ) -> plt.Figure:
        """
        Create side-by-side before/after comparison.

        Args:
            before_img: Pre-disaster image (H, W, 3)
            after_img: Post-disaster image (H, W, 3)
            damage_map: Optional damage predictions to overlay
            title: Figure title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3 if damage_map is not None else 2,
                                  figsize=(16, 6))

        axes[0].imshow(before_img)
        axes[0].set_title('Before Disaster', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(after_img)
        axes[1].set_title('After Disaster', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        if damage_map is not None:
            overlay = self.create_damage_overlay(after_img, damage_map, alpha=0.6)
            axes[2].imshow(overlay)
            axes[2].set_title('Damage Detection', fontsize=14, fontweight='bold')
            axes[2].axis('off')

            # Add legend
            legend_elements = [
                Patch(facecolor=np.array(DAMAGE_COLORS[i]) / 255, label=DAMAGE_LABELS[i])
                for i in range(4)
            ]
            axes[2].legend(handles=legend_elements, loc='lower right', fontsize=10)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig

    def create_damage_overlay(
        self,
        image: np.ndarray,
        damage_map: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create damage overlay on image.

        Args:
            image: Original image (H, W, 3)
            damage_map: Damage predictions (H, W)
            alpha: Overlay transparency

        Returns:
            Overlay image (H, W, 3)
        """
        if image.shape[:2] != damage_map.shape:
            damage_map = cv2.resize(
                damage_map.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        overlay = np.zeros_like(image)

        for class_id, color in DAMAGE_COLORS.items():
            mask = damage_map == class_id
            overlay[mask] = color

        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return result

    def create_urgency_heatmap(
        self,
        urgency_map: np.ndarray,
        overlay_image: np.ndarray = None,
        title: str = "Response Urgency Heatmap"
    ) -> plt.Figure:
        """
        Create urgency heatmap visualization.

        Args:
            urgency_map: Urgency scores 0-100 (H, W)
            overlay_image: Optional background image
            title: Figure title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        if overlay_image is not None:
            ax.imshow(overlay_image, alpha=0.3)

        im = ax.imshow(urgency_map, cmap=URGENCY_COLORMAP, alpha=0.7,
                       vmin=0, vmax=100)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Urgency Score', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

        # Add urgency zone labels
        self._add_urgency_legend(ax)

        plt.tight_layout()
        return fig

    def _add_urgency_legend(self, ax):
        """Add urgency zone legend to axis."""
        zones = [
            ('CRITICAL (80-100)', '#e74c3c'),
            ('HIGH (60-80)', '#e67e22'),
            ('MEDIUM (40-60)', '#f1c40f'),
            ('LOW (20-40)', '#7dcea0'),
            ('MINIMAL (0-20)', '#27ae60')
        ]

        legend_elements = [
            Patch(facecolor=color, label=label)
            for label, color in zones
        ]

        ax.legend(handles=legend_elements, loc='lower right',
                  title='Urgency Zones', fontsize=9, title_fontsize=10)

    def create_zone_map(
        self,
        urgency_map: np.ndarray,
        damage_map: np.ndarray
    ) -> np.ndarray:
        """
        Create discrete zone visualization.

        Args:
            urgency_map: Urgency scores 0-100
            damage_map: Damage predictions

        Returns:
            Zone visualization (H, W, 3)
        """
        H, W = urgency_map.shape
        zone_map = np.zeros((H, W, 3), dtype=np.uint8)

        zones = {
            'CRITICAL': {'range': (80, 100), 'color': (231, 76, 60)},
            'HIGH': {'range': (60, 80), 'color': (230, 126, 34)},
            'MEDIUM': {'range': (40, 60), 'color': (241, 196, 15)},
            'LOW': {'range': (20, 40), 'color': (125, 206, 160)},
            'MINIMAL': {'range': (0, 20), 'color': (39, 174, 96)}
        }

        damage_mask = damage_map > 0

        for zone_name, zone_info in zones.items():
            min_val, max_val = zone_info['range']
            mask = (urgency_map >= min_val) & (urgency_map < max_val) & damage_mask
            zone_map[mask] = zone_info['color']

        return zone_map

    def create_statistics_panel(
        self,
        damage_stats: Dict,
        urgency_stats: Dict,
        affected_population: int = None
    ) -> plt.Figure:
        """
        Create statistics visualization panel.

        Args:
            damage_stats: Damage statistics from inference
            urgency_stats: Urgency zone statistics
            affected_population: Estimated affected population

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 6))

        # Damage distribution pie chart
        ax1 = fig.add_subplot(131)
        damage_counts = [
            damage_stats.get('damage_percentages', {}).get(label, 0)
            for label in DAMAGE_LABELS.values()
        ]
        colors = [np.array(DAMAGE_COLORS[i]) / 255 for i in range(4)]

        wedges, texts, autotexts = ax1.pie(
            damage_counts,
            labels=list(DAMAGE_LABELS.values()),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('Damage Distribution', fontsize=12, fontweight='bold')

        # Urgency zone bar chart
        ax2 = fig.add_subplot(132)
        zone_names = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
        zone_values = [
            urgency_stats.get(zone, {}).get('percentage', 0)
            for zone in zone_names
        ]
        zone_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#7dcea0', '#27ae60']

        bars = ax2.barh(zone_names, zone_values, color=zone_colors)
        ax2.set_xlabel('Percentage of Damaged Area', fontsize=10)
        ax2.set_title('Urgency Zone Distribution', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

        # Key metrics
        ax3 = fig.add_subplot(133)
        ax3.axis('off')

        metrics_text = f"""
KEY METRICS

Total Damaged Area: {damage_stats.get('total_damaged_percentage', 0):.1f}%

Destroyed Buildings: {damage_stats.get('damage_percentages', {}).get('Destroyed', 0):.1f}%
Major Damage: {damage_stats.get('damage_percentages', {}).get('Major Damage', 0):.1f}%

"""
        if affected_population:
            metrics_text += f"Estimated Affected Population: ~{affected_population:,}\n\n"

        critical_pct = urgency_stats.get('CRITICAL', {}).get('percentage', 0)
        high_pct = urgency_stats.get('HIGH', {}).get('percentage', 0)

        metrics_text += f"""
URGENCY SUMMARY

Critical Priority Areas: {critical_pct:.1f}%
High Priority Areas: {high_pct:.1f}%

Areas Requiring Immediate Response:
{critical_pct + high_pct:.1f}% of damaged zone
"""

        ax3.text(0.1, 0.5, metrics_text, fontsize=11, va='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        plt.tight_layout()
        return fig

    def create_priority_locations_map(
        self,
        image: np.ndarray,
        priorities: List[Dict],
        urgency_map: np.ndarray = None
    ) -> np.ndarray:
        """
        Create map with priority response locations marked.

        Args:
            image: Base image
            priorities: List of priority locations from generate_response_priorities
            urgency_map: Optional urgency map for overlay

        Returns:
            Annotated image (H, W, 3)
        """
        result = image.copy()

        if urgency_map is not None:
            # Add urgency overlay
            heatmap = cv2.applyColorMap(
                (urgency_map / 100 * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            result = cv2.addWeighted(result, 0.6, heatmap, 0.4, 0)

        # Draw priority markers
        for p in priorities:
            center = (p['center_col'], p['center_row'])
            rank = p['priority_rank']
            urgency = p['avg_urgency']

            # Color based on urgency
            if urgency >= 80:
                color = (255, 0, 0)
            elif urgency >= 60:
                color = (255, 128, 0)
            elif urgency >= 40:
                color = (255, 255, 0)
            else:
                color = (0, 255, 0)

            # Draw marker
            cv2.circle(result, center, 20, color, 3)
            cv2.circle(result, center, 5, color, -1)

            # Draw rank number
            cv2.putText(
                result,
                str(rank),
                (center[0] - 8, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return result

    def create_full_report(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
        damage_map: np.ndarray,
        urgency_results: Dict,
        damage_stats: Dict,
        priorities: List[Dict] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Create comprehensive analysis report.

        Args:
            before_img: Pre-disaster image
            after_img: Post-disaster image
            damage_map: Damage predictions
            urgency_results: Output from UrgencyCalculator
            damage_stats: Damage statistics
            priorities: Priority response locations
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 16))

        # Before/After
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.imshow(before_img)
        ax1.set_title('Before Disaster', fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(3, 3, 2)
        ax2.imshow(after_img)
        ax2.set_title('After Disaster', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Damage overlay
        ax3 = fig.add_subplot(3, 3, 3)
        overlay = self.create_damage_overlay(after_img, damage_map, alpha=0.6)
        ax3.imshow(overlay)
        ax3.set_title('Damage Detection', fontsize=12, fontweight='bold')
        ax3.axis('off')
        legend_elements = [
            Patch(facecolor=np.array(DAMAGE_COLORS[i]) / 255, label=DAMAGE_LABELS[i])
            for i in range(4)
        ]
        ax3.legend(handles=legend_elements, loc='lower right', fontsize=8)

        # Urgency heatmap
        ax4 = fig.add_subplot(3, 3, 4)
        im = ax4.imshow(urgency_results['urgency_map'], cmap=URGENCY_COLORMAP,
                        vmin=0, vmax=100)
        ax4.set_title('Urgency Heatmap', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, fraction=0.046)

        # Zone map
        ax5 = fig.add_subplot(3, 3, 5)
        zone_map = self.create_zone_map(urgency_results['urgency_map'], damage_map)
        ax5.imshow(zone_map)
        ax5.set_title('Priority Zones', fontsize=12, fontweight='bold')
        ax5.axis('off')
        self._add_urgency_legend(ax5)

        # Priority locations
        ax6 = fig.add_subplot(3, 3, 6)
        if priorities:
            priority_map = self.create_priority_locations_map(
                after_img, priorities, urgency_results['urgency_map']
            )
            ax6.imshow(priority_map)
            ax6.set_title('Priority Response Locations', fontsize=12, fontweight='bold')
        else:
            ax6.imshow(after_img)
            ax6.set_title('Post-Disaster', fontsize=12, fontweight='bold')
        ax6.axis('off')

        # Statistics
        ax7 = fig.add_subplot(3, 3, 7)
        ax7.axis('off')

        stats_text = "DAMAGE STATISTICS\n\n"
        for level, label in DAMAGE_LABELS.items():
            pct = damage_stats.get('damage_percentages', {}).get(label, 0)
            stats_text += f"{label}: {pct:.1f}%\n"

        stats_text += f"\nTotal Damaged: {damage_stats.get('total_damaged_percentage', 0):.1f}%"

        ax7.text(0.1, 0.5, stats_text, fontsize=11, va='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

        # Urgency statistics
        ax8 = fig.add_subplot(3, 3, 8)
        zone_stats = urgency_results.get('zone_stats', {})
        zone_names = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
        zone_values = [zone_stats.get(z, {}).get('percentage', 0) for z in zone_names]
        zone_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#7dcea0', '#27ae60']

        ax8.barh(zone_names, zone_values, color=zone_colors)
        ax8.set_xlabel('% of Damaged Area')
        ax8.set_title('Urgency Distribution', fontsize=12, fontweight='bold')
        ax8.invert_yaxis()

        # Key metrics
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.axis('off')

        affected_pop = urgency_results.get('estimated_affected', 0)
        key_text = f"""
KEY FINDINGS

Estimated Affected Population:
~{affected_pop:,} people

Critical Response Areas:
{zone_stats.get('CRITICAL', {}).get('percentage', 0):.1f}% of damaged zone

Immediate Action Required:
{zone_stats.get('CRITICAL', {}).get('percentage', 0) + zone_stats.get('HIGH', {}).get('percentage', 0):.1f}% of damaged zone
"""
        ax9.text(0.1, 0.5, key_text, fontsize=11, va='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.2))

        fig.suptitle('RapidEye Disaster Analysis Report',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Report saved to: {save_path}")

        return fig


def save_visualization(fig: plt.Figure, path: str, dpi: int = 150):
    """Save matplotlib figure to file."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {path}")


if __name__ == '__main__':
    print("Testing Visualization Module...")

    # Create mock data
    H, W = 512, 512
    before_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    after_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    damage_map = np.random.randint(0, 4, (H, W), dtype=np.uint8)
    urgency_map = np.random.uniform(0, 100, (H, W))

    visualizer = DisasterVisualizer()

    # Test damage overlay
    overlay = visualizer.create_damage_overlay(after_img, damage_map)
    print(f"Damage overlay shape: {overlay.shape}")

    # Test zone map
    zone_map = visualizer.create_zone_map(urgency_map, damage_map)
    print(f"Zone map shape: {zone_map.shape}")

    print("\nVisualization module ready!")
