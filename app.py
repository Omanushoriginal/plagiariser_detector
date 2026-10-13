"""
app.py - FastAPI Application for YouTube Plagiarism & AI-Ripoff Detector
"""

import os
import sys
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import detector

app = FastAPI(
    title="YouTube Content Plagiarism & AI-Ripoff Detector",
    description="Multi-modal plagiarism detection engine for YouTube creators: thumbnails, scripts, and video metadata.",
    version="1.0.0",
)

# Enable CORS for local development and GitHub Pages integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ScanRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL or Video ID")
    max_candidates: int = Field(default=6, ge=1, le=12, description="Maximum YouTube candidates to scan")
    use_dinov2: bool = Field(default=True, description="Enable Meta DINOv2 Deep AI-modified analysis")


class CompareRequest(BaseModel):
    url1: str = Field(..., description="First YouTube video URL (e.g., your original)")
    url2: str = Field(..., description="Second YouTube video URL (e.g., suspected copy)")
    use_dinov2: bool = Field(default=True, description="Enable Meta DINOv2 Deep AI-modified analysis")


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "YouTube Plagiarism Detector",
        "author": "omanushoriginal",
        "email": "drisham4@gmail.com",
        "version": "1.0.0",
    }


@app.post("/api/scan")
def scan_video(req: ScanRequest):
    """
    Scans YouTube for any potential copycat or duplicate videos of the given URL.
    """
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube video URL or ID.")

    try:
        results = detector.scan_youtube_for_plagiarism(
            query_url=req.url.strip(),
            max_candidates=req.max_candidates,
            use_dinov2=req.use_dinov2,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plagiarism scan failed: {str(e)}")


@app.post("/api/compare")
def compare_videos(req: CompareRequest):
    """
    Compares two specific YouTube videos directly side-by-side.
    """
    if not req.url1 or not req.url2:
        raise HTTPException(status_code=400, detail="Please provide both video URLs.")

    try:
        results = detector.compare_two_videos(
            url1=req.url1.strip(),
            url2=req.url2.strip(),
            use_dinov2=req.use_dinov2,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video comparison failed: {str(e)}")


@app.get("/api/sample")
def get_sample_data():
    """
    Returns realistic sample scan data for instant preview and demo testing.
    """
    return {
        "status": "success",
        "scan_time_sec": 4.12,
        "source_video": {
            "id": "_uQrJ0TkZlc",
            "url": "https://www.youtube.com/watch?v=_uQrJ0TkZlc",
            "title": "Python Full Course for Beginners [Tutorial]",
            "channel": "Programming with Mosh",
            "channel_url": "https://www.youtube.com/@programmingwithmosh",
            "thumbnail": "https://img.youtube.com/vi/_uQrJ0TkZlc/maxresdefault.jpg",
            "duration_sec": 21600,
            "duration_str": "06:00:00",
            "view_count": 38400000,
            "upload_date": "2019-02-18",
            "description": "Python tutorial for beginners - Learn Python for machine learning and web development.",
            "transcript_stats": {"word_count": 48200, "available": True}
        },
        "candidates_count": 4,
        "highest_plagiarism_risk_pct": 94.2,
        "matches": [
            {
                "candidate": {
                    "id": "SAMPLE_COPY_1",
                    "url": "https://www.youtube.com/watch?v=sample1",
                    "title": "Python Tutorial For Beginners (Full Course 2024)",
                    "channel": "Code Academy Hub Reuploads",
                    "thumbnail": "https://img.youtube.com/vi/_uQrJ0TkZlc/hqdefault.jpg",
                    "duration_sec": 21600,
                    "duration_str": "06:00:00",
                    "view_count": 142000,
                    "upload_date": "2024-05-12",
                    "transcript_stats": {"word_count": 48100, "available": True}
                },
                "visual_comparison": {
                    "success": True,
                    "visual_similarity_pct": 98.4,
                    "verdict": "NEAR_DUPLICATE_THUMBNAIL",
                    "phash_distance": 2,
                    "dhash_distance": 1,
                    "dinov2_pct": 99.1,
                    "sift_inliers": 84,
                    "geometric_match": True,
                    "findings": [
                        "Near-duplicate thumbnail detected (pHash distance: 2).",
                        "Candidate copied the entire original thumbnail artwork with slight contrast adjustment."
                    ]
                },
                "transcript_comparison": {
                    "transcripts_available": True,
                    "script_similarity_pct": 96.5,
                    "sequence_similarity": 0.965,
                    "verdict": "BLATANT_SCRIPT_COPY",
                    "findings": [
                        "Blatant script plagiarism detected (96.5% sequence overlap).",
                        "Candidate video audio is a direct re-upload of original spoken audio."
                    ]
                },
                "title_comparison": {
                    "title_similarity_pct": 82.0,
                    "is_exact_title": False
                },
                "verdict": {
                    "risk_score_pct": 94.2,
                    "threat_level": "CRITICAL",
                    "threat_color": "red",
                    "threat_badge": "High Probability of Plagiarism",
                    "evidence": [
                        "Near-duplicate thumbnail detected (pHash distance: 2).",
                        "Blatant script plagiarism detected (96.5% sequence overlap).",
                        "Video length matches original within 0.1% duration margin."
                    ]
                }
            },
            {
                "candidate": {
                    "id": "SAMPLE_AI_COPY",
                    "url": "https://www.youtube.com/watch?v=sample2",
                    "title": "Learn Python in 6 Hours | Complete Roadmap",
                    "channel": "Tech Synth AI",
                    "thumbnail": "https://img.youtube.com/vi/_uQrJ0TkZlc/mqdefault.jpg",
                    "duration_sec": 19400,
                    "duration_str": "05:23:20",
                    "view_count": 89000,
                    "upload_date": "2023-11-04",
                    "transcript_stats": {"word_count": 41000, "available": True}
                },
                "visual_comparison": {
                    "success": True,
                    "visual_similarity_pct": 86.2,
                    "verdict": "AI_MODIFIED_OR_STYLED_THUMBNAIL",
                    "phash_distance": 11,
                    "dhash_distance": 14,
                    "dinov2_pct": 88.4,
                    "sift_inliers": 29,
                    "geometric_match": True,
                    "findings": [
                        "AI-stylized or filter-modified duplicate detected (DINOv2 similarity: 88.4%).",
                        "Layout and keypoint geometry match the original thumbnail layout."
                    ]
                },
                "transcript_comparison": {
                    "transcripts_available": True,
                    "script_similarity_pct": 74.0,
                    "sequence_similarity": 0.740,
                    "verdict": "HEAVY_SCRIPT_PARAPHRASE",
                    "findings": [
                        "Substantial script paraphrase detected (74.0% alignment). Sentences follow original structure with synonyms."
                    ]
                },
                "title_comparison": {
                    "title_similarity_pct": 65.0,
                    "is_exact_title": False
                },
                "verdict": {
                    "risk_score_pct": 78.5,
                    "threat_level": "CRITICAL",
                    "threat_color": "red",
                    "threat_badge": "AI-Modified Duplicate / Stolen Structure",
                    "evidence": [
                        "DINOv2 deep vision detected AI restyling/inpainting of original thumbnail.",
                        "Spoken transcript follows identical chapter progression and structure."
                    ]
                }
            },
            {
                "candidate": {
                    "id": "SAMPLE_DERIVATIVE",
                    "url": "https://www.youtube.com/watch?v=sample3",
                    "title": "Python for Absolute Beginners - Quick Overview",
                    "channel": "DevTips Daily",
                    "thumbnail": "https://img.youtube.com/vi/rfscVS0vtbw/hqdefault.jpg",
                    "duration_sec": 1200,
                    "duration_str": "20:00",
                    "view_count": 312000,
                    "upload_date": "2022-08-19",
                    "transcript_stats": {"word_count": 3200, "available": True}
                },
                "visual_comparison": {
                    "success": True,
                    "visual_similarity_pct": 42.0,
                    "verdict": "ORIGINAL_THUMBNAIL",
                    "phash_distance": 22,
                    "dhash_distance": 25,
                    "dinov2_pct": 45.2,
                    "sift_inliers": 4,
                    "geometric_match": False,
                    "findings": [
                        "Thumbnail is visually distinct."
                    ]
                },
                "transcript_comparison": {
                    "transcripts_available": True,
                    "script_similarity_pct": 38.0,
                    "sequence_similarity": 0.380,
                    "verdict": "MODERATE_TOPIC_OVERLAP",
                    "findings": [
                        "General Python programming keyword overlap. Independent narration."
                    ]
                },
                "title_comparison": {
                    "title_similarity_pct": 52.0,
                    "is_exact_title": False
                },
                "verdict": {
                    "risk_score_pct": 39.5,
                    "threat_level": "MODERATE",
                    "threat_color": "yellow",
                    "threat_badge": "Topic Overlap / Independent Work",
                    "evidence": [
                        "Similar instructional subject matter.",
                        "No direct visual or verbatim script copying detected."
                    ]
                }
            }
        ]
    }


# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to YouTube Plagiarism Detector API. Static frontend is building..."}


if __name__ == "__main__":
    import uvicorn
    print("[app.py] Starting server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
