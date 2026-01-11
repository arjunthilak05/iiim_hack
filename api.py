"""
RapidEye Command Center - FastAPI Backend
Production-ready API for disaster damage detection
"""

import os
import io
import sys
import time
import base64
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from inference import DamagePredictor, create_damage_overlay, calculate_damage_stats, DAMAGE_CLASSES
from urgency import UrgencyCalculator, generate_response_priorities, create_urgency_heatmap, URGENCY_ZONES

# Initialize FastAPI app
app = FastAPI(
    title="RapidEye Command Center",
    description="AI-Powered Disaster Detection & Response Prioritization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (lazy loading)
predictor = None
MODEL_PATH = Path(__file__).parent / 'models' / 'best.pth'

def get_predictor():
    """Lazy load the model predictor."""
    global predictor
    if predictor is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail="Model checkpoint not found")
        predictor = DamagePredictor(str(MODEL_PATH), device='cpu')
    return predictor

# Pydantic models
class AnalysisResult(BaseModel):
    success: bool
    analysis_time: float
    damage_stats: dict
    urgency_stats: dict
    priorities: list
    estimated_affected: int
    images: dict

class HealthCheck(BaseModel):
    model_config = {'protected_namespaces': ()}
    status: str
    model_loaded: bool
    timestamp: str

# Helper functions
def image_to_base64(img: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    pil_img = Image.fromarray(img.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file."""
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    return np.array(img)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    frontend_path = Path(__file__).parent / 'frontend' / 'index.html'
    if frontend_path.exists():
        return frontend_path.read_text()
    return HTMLResponse("<h1>RapidEye API</h1><p>Frontend not found. Use /docs for API documentation.</p>")

@app.get("/api/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "operational",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze")
async def analyze_images(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...)
):
    """
    Analyze before/after satellite images for damage detection.
    Returns damage map, urgency scores, and priority response locations.
    """
    start_time = time.time()

    try:
        # Load images
        before_img = load_image_from_upload(before_image)
        after_img = load_image_from_upload(after_image)

        # Get predictor
        pred = get_predictor()

        # Run inference
        damage_map = pred.predict(before_img, after_img)

        # Calculate damage stats
        damage_stats = calculate_damage_stats(damage_map)

        # Calculate urgency scores
        calculator = UrgencyCalculator()
        urgency_results = calculator.calculate_urgency_score(damage_map)

        # Generate priorities
        priorities = generate_response_priorities(urgency_results, top_n=10)

        # Create visualizations
        damage_overlay = create_damage_overlay(after_img, damage_map, alpha=0.5)
        urgency_heatmap = create_urgency_heatmap(urgency_results['urgency_map'])

        analysis_time = time.time() - start_time

        return convert_numpy_types({
            "success": True,
            "analysis_time": round(analysis_time, 2),
            "damage_stats": damage_stats,
            "urgency_stats": urgency_results['zone_stats'],
            "priorities": priorities,
            "estimated_affected": urgency_results['estimated_affected'],
            "images": {
                "before": image_to_base64(before_img),
                "after": image_to_base64(after_img),
                "damage_overlay": image_to_base64(damage_overlay),
                "urgency_heatmap": image_to_base64(urgency_heatmap)
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sample")
async def get_sample_analysis():
    """
    Run analysis on sample xView2 data for demo purposes.
    """
    start_time = time.time()

    # Find sample images
    archive_dir = Path(__file__).parent / 'archive' / 'train' / 'train' / 'images'

    if not archive_dir.exists():
        # Try processed data
        processed_dir = Path(__file__).parent / 'data' / 'xview2_processed' / 'train' / 'images'
        if not processed_dir.exists():
            raise HTTPException(status_code=404, detail="No sample data found")

        # Load from processed
        samples = list(processed_dir.glob('*.npy'))[:10]

        best_sample = None
        best_damage = 0

        pred = get_predictor()

        for sample_path in samples:
            combined = np.load(sample_path)
            before_img = combined[:, :, :3]
            after_img = combined[:, :, 3:6]

            damage_map = pred.predict(before_img, after_img)
            damage_pct = (damage_map > 0).sum() / damage_map.size * 100

            if damage_pct > best_damage:
                best_damage = damage_pct
                best_sample = (before_img, after_img, damage_map, sample_path.stem)

        if best_sample is None:
            raise HTTPException(status_code=404, detail="No suitable sample found")

        before_img, after_img, damage_map, sample_id = best_sample

    else:
        # Load from raw archive
        samples = list(archive_dir.glob('*_pre_disaster.png'))[:20]

        best_sample = None
        best_damage = 0

        pred = get_predictor()

        for pre_path in samples:
            sample_id = pre_path.stem.replace('_pre_disaster', '')
            post_path = pre_path.parent / f'{sample_id}_post_disaster.png'

            if not post_path.exists():
                continue

            before_img = np.array(Image.open(pre_path).convert('RGB'))
            after_img = np.array(Image.open(post_path).convert('RGB'))

            damage_map = pred.predict(before_img, after_img)
            damage_pct = (damage_map > 0).sum() / damage_map.size * 100

            if damage_pct > best_damage:
                best_damage = damage_pct
                best_sample = (before_img, after_img, damage_map, sample_id)

        if best_sample is None:
            raise HTTPException(status_code=404, detail="No suitable sample found")

        before_img, after_img, damage_map, sample_id = best_sample

    # Calculate stats
    damage_stats = calculate_damage_stats(damage_map)

    calculator = UrgencyCalculator()
    urgency_results = calculator.calculate_urgency_score(damage_map)

    priorities = generate_response_priorities(urgency_results, top_n=10)

    damage_overlay = create_damage_overlay(after_img, damage_map, alpha=0.5)
    urgency_heatmap = create_urgency_heatmap(urgency_results['urgency_map'])

    analysis_time = time.time() - start_time

    return convert_numpy_types({
        "success": True,
        "sample_id": sample_id,
        "analysis_time": round(analysis_time, 2),
        "damage_stats": damage_stats,
        "urgency_stats": urgency_results['zone_stats'],
        "priorities": priorities,
        "estimated_affected": urgency_results['estimated_affected'],
        "images": {
            "before": image_to_base64(before_img),
            "after": image_to_base64(after_img),
            "damage_overlay": image_to_base64(damage_overlay),
            "urgency_heatmap": image_to_base64(urgency_heatmap)
        }
    })

@app.post("/api/export/csv")
async def export_csv(priorities: List[dict]):
    """Export priority queue as CSV."""
    import csv

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        'priority_rank', 'zone', 'avg_urgency', 'max_urgency',
        'center_row', 'center_col'
    ])
    writer.writeheader()

    for p in priorities:
        writer.writerow({
            'priority_rank': p.get('priority_rank', ''),
            'zone': p.get('zone', ''),
            'avg_urgency': p.get('avg_urgency', ''),
            'max_urgency': p.get('max_urgency', ''),
            'center_row': p.get('center_row', ''),
            'center_col': p.get('center_col', '')
        })

    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rapideye_priorities.csv"}
    )

@app.get("/api/config")
async def get_config():
    """Get configuration for frontend."""
    return {
        "damage_classes": {k: v['name'] for k, v in DAMAGE_CLASSES.items()},
        "damage_colors": {k: v['color'] for k, v in DAMAGE_CLASSES.items()},
        "urgency_zones": URGENCY_ZONES,
        "model_loaded": predictor is not None
    }

# Mount static files
frontend_dir = Path(__file__).parent / 'frontend'
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
