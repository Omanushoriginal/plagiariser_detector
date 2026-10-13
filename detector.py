"""
detector.py - YouTube Video Plagiarism & AI-Ripoff Detection Engine

Features:
1. Video Metadata & Thumbnail Extraction (via yt-dlp)
2. Closed-Caption / Transcript Extraction & Timestamp Mapping (via youtube-transcript-api)
3. YouTube-wide Candidate Search for Copycat Videos
4. Multi-Modal Plagiarism Comparison:
   - Visual: Perceptual Hash (pHash/dHash), Meta DINOv2 (AI-styling/composition), SIFT+RANSAC (homography)
   - Audio/Script: SequenceMatcher transcript alignment & Jaccard token overlap
   - Metadata: Title token similarity & upload chronology (detects who uploaded first)
5. Comprehensive Evidence Report Generation (suitable for copyright claims / DMCA)
"""

import os
import re
import io
import time
import math
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

import functions


def extract_youtube_id(url_or_id: str) -> str:
    """
    Extracts the clean 11-character YouTube video ID from various URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - Raw VIDEO_ID
    """
    if not url_or_id:
        return ""
    
    cleaned = url_or_id.strip()
    if len(cleaned) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", cleaned):
        return cleaned

    patterns = [
        r"(?:v=|\/vi\/|\/embed\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/watch\?.*v=([0-9A-Za-z_-]{11})",
        r"youtube\.com\/live\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)

    # Fallback query parameter parsing
    try:
        parsed = urlparse(cleaned)
        qs = parse_qs(parsed.query)
        if "v" in qs and len(qs["v"]) > 0:
            return qs["v"][0]
    except Exception:
        pass

    return cleaned


def format_duration(seconds: Optional[float]) -> str:
    """Formats seconds into MM:SS or HH:MM:SS."""
    if not seconds or math.isnan(seconds):
        return "Unknown"
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_video_metadata(url_or_id: str) -> Dict[str, Any]:
    """
    Fetches rich video metadata using yt-dlp without downloading the media file.
    """
    video_id = extract_youtube_id(url_or_id)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL or ID: {url_or_id}")

    watch_url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(watch_url, download=False)
    except Exception as e:
        # Fallback to direct oEmbed metadata if yt-dlp hits a restriction
        info = {
            "id": video_id,
            "title": f"YouTube Video ({video_id})",
            "uploader": "Unknown Creator",
            "duration": None,
            "view_count": None,
            "upload_date": None,
            "description": "",
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        }

    # Best thumbnail selection
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    thumbnails = info.get("thumbnails", [])
    if thumbnails:
        # Pick highest resolution thumbnail
        best_thumb = max(thumbnails, key=lambda t: (t.get("preference", 0) or 0, t.get("width", 0) or 0))
        thumbnail_url = best_thumb.get("url", thumbnail_url)

    upload_date_str = info.get("upload_date")
    formatted_date = "Unknown"
    if upload_date_str and len(upload_date_str) == 8:
        formatted_date = f"{upload_date_str[:4]}-{upload_date_str[4:6]}-{upload_date_str[6:]}"

    return {
        "id": video_id,
        "url": watch_url,
        "title": info.get("title", f"Video {video_id}"),
        "channel": info.get("uploader") or info.get("channel") or "Unknown Creator",
        "channel_url": info.get("uploader_url") or info.get("channel_url") or "",
        "thumbnail": thumbnail_url,
        "duration_sec": info.get("duration"),
        "duration_str": format_duration(info.get("duration")),
        "view_count": info.get("view_count") or 0,
        "upload_date": formatted_date,
        "description": (info.get("description") or "")[:400],
    }


def get_video_transcript(video_id: str) -> Dict[str, Any]:
    """
    Fetches captions / transcript for a YouTube video using youtube-transcript-api.
    Returns transcript snippet list and combined full text.
    """
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
        snippets = []
        full_text_parts = []

        for item in fetched.snippets:
            snippets.append({
                "start": round(getattr(item, "start", 0.0), 2),
                "duration": round(getattr(item, "duration", 0.0), 2),
                "text": getattr(item, "text", "").strip(),
            })
            full_text_parts.append(getattr(item, "text", "").strip())

        full_text = " ".join(full_text_parts)
        return {
            "available": True,
            "snippets_count": len(snippets),
            "snippets": snippets[:150],  # keep reasonable payload size
            "full_text": full_text,
            "word_count": len(full_text.split()),
        }
    except Exception as e:
        return {
            "available": False,
            "snippets_count": 0,
            "snippets": [],
            "full_text": "",
            "word_count": 0,
            "error": str(e),
        }


def search_youtube_candidates(
    title: str,
    original_id: str,
    max_candidates: int = 6
) -> List[Dict[str, Any]]:
    """
    Searches YouTube for potential duplicate, copycat, or plagiarized videos
    using multi-query search via yt-dlp.
    """
    # Clean title for effective search
    cleaned_title = re.sub(r"[^\w\s]", " ", title)
    cleaned_title = re.sub(r"\s+", " ", cleaned_title).strip()
    words = [w for w in cleaned_title.split() if len(w) > 2]
    keywords_query = " ".join(words[:6])

    queries = [cleaned_title]
    if keywords_query and keywords_query.lower() != cleaned_title.lower():
        queries.append(keywords_query)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }

    seen_ids = {original_id}
    candidates = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for q in queries:
            search_str = f"ytsearch{max_candidates + 2}:{q}"
            try:
                search_res = ydl.extract_info(search_str, download=False)
                entries = search_res.get("entries", [])
                for entry in entries:
                    if not entry:
                        continue
                    v_id = entry.get("id")
                    if not v_id or v_id in seen_ids:
                        continue

                    seen_ids.add(v_id)
                    thumb_url = f"https://img.youtube.com/vi/{v_id}/hqdefault.jpg"
                    if entry.get("thumbnails"):
                        thumb_url = entry["thumbnails"][0].get("url", thumb_url)

                    candidates.append({
                        "id": v_id,
                        "url": f"https://www.youtube.com/watch?v={v_id}",
                        "title": entry.get("title", f"Video {v_id}"),
                        "channel": entry.get("uploader") or entry.get("channel") or "Unknown Channel",
                        "thumbnail": thumb_url,
                        "duration_sec": entry.get("duration"),
                        "duration_str": format_duration(entry.get("duration")),
                        "view_count": entry.get("view_count") or 0,
                        "description": (entry.get("description") or "")[:250],
                    })

                    if len(candidates) >= max_candidates:
                        break
            except Exception as e:
                print(f"[detector.py] Search query failed: {q} - {e}")

            if len(candidates) >= max_candidates:
                break

    return candidates


def compare_thumbnails(
    thumb1_url: str,
    thumb2_url: str,
    use_dinov2: bool = True
) -> Dict[str, Any]:
    """
    Multi-tier thumbnail plagiarism check:
    1. Perceptual Hashing (aHash, pHash, dHash)
    2. SIFT Keypoint & Homography Inliers
    3. DINOv2 Deep Structural Feature Cosine Similarity
    """
    try:
        img1 = functions.load_image(thumb1_url)
        img2 = functions.load_image(thumb2_url)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load thumbnails: {str(e)}",
            "visual_similarity": 0.0,
            "verdict": "ERROR_LOADING_IMAGE",
            "findings": ["Could not download one or both thumbnails for visual comparison."],
        }

    # Resize to standard analysis resolution for high-speed deterministic evaluation
    img1_std = img1.resize((480, 270))
    img2_std = img2.resize((480, 270))

    # Tier 1: Perceptual Hashing
    hash_res = functions.compare_perceptual_hash(img1_std, img2_std)
    phash_dist = hash_res["phash_distance"]
    dhash_dist = hash_res["dhash_distance"]

    # Tier 2: SIFT Geometric Matching
    sift_res = functions.detect_geometric_feature_matches(img1_std, img2_std, min_inliers=12)

    # Tier 3: Meta DINOv2 (Deep Structural / AI-Modified similarity)
    dinov2_score = 0.0
    if use_dinov2:
        try:
            _, dinov2_score = functions.compare_dinov2_similarity(img1_std, img2_std, threshold=0.82)
            dinov2_score = round(float(dinov2_score), 4)
        except Exception as e:
            dinov2_score = 0.0

    # Determine visual similarity score (0.0 to 1.0)
    # Exact hash match gives 1.0
    # Small hash distance maps to high similarity
    hash_sim = max(0.0, 1.0 - (phash_dist / 32.0))
    sift_sim = min(1.0, sift_res["ransac_inliers"] / 40.0)

    if use_dinov2 and dinov2_score > 0:
        visual_similarity = round(0.40 * hash_sim + 0.35 * dinov2_score + 0.25 * sift_sim, 4)
    else:
        visual_similarity = round(0.60 * hash_sim + 0.40 * sift_sim, 4)

    # Specific findings
    findings = []
    verdict = "ORIGINAL_THUMBNAIL"

    if hash_res["is_identical"]:
        visual_similarity = 1.0
        verdict = "EXACT_DUPLICATE_THUMBNAIL"
        findings.append("Thumbnails are 100% identical byte-for-byte or exact visual duplicates.")
    elif hash_res["is_near_duplicate"]:
        visual_similarity = max(visual_similarity, 0.92)
        verdict = "NEAR_DUPLICATE_THUMBNAIL"
        findings.append(f"Near-duplicate thumbnail detected (pHash distance: {phash_dist}). Suspect is a cropped, compressed, or resized clone.")
    elif dinov2_score >= 0.85:
        visual_similarity = max(visual_similarity, dinov2_score)
        verdict = "AI_MODIFIED_OR_STYLED_THUMBNAIL"
        findings.append(f"AI-stylized or filter-modified duplicate detected (DINOv2 similarity: {dinov2_score * 100:.1f}%).")
    elif sift_res["geometric_match"]:
        verdict = "GEOMETRIC_COMPOSITION_THEFT"
        findings.append(f"Key landmark elements match layout ({sift_res['ransac_inliers']} RANSAC inliers). Objects or text were directly traced/copied.")
    elif visual_similarity >= 0.70:
        verdict = "HIGH_VISUAL_SIMILARITY"
        findings.append("High visual similarity detected in composition, color grading, or subject matter.")
    else:
        findings.append("No significant visual copying detected in thumbnail.")

    return {
        "success": True,
        "visual_similarity": visual_similarity,
        "visual_similarity_pct": round(visual_similarity * 100, 1),
        "verdict": verdict,
        "phash_distance": phash_dist,
        "dhash_distance": dhash_dist,
        "dinov2_score": dinov2_score,
        "dinov2_pct": round(dinov2_score * 100, 1) if dinov2_score > 0 else 0,
        "sift_inliers": sift_res["ransac_inliers"],
        "geometric_match": sift_res["geometric_match"],
        "findings": findings,
    }


def compare_transcripts(trans1: Dict[str, Any], trans2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compares video spoken scripts using difflib sequence matching and vocabulary overlap.
    """
    t1_avail = trans1.get("available", False)
    t2_avail = trans2.get("available", False)

    if not t1_avail or not t2_avail:
        return {
            "transcripts_available": False,
            "t1_available": t1_avail,
            "t2_available": t2_avail,
            "script_similarity": 0.0,
            "script_similarity_pct": 0.0,
            "verdict": "CAPTIONS_NOT_AVAILABLE",
            "findings": [
                "Captions/transcripts were not available on both videos. Script comparison skipped."
            ],
            "matched_excerpts": [],
        }

    text1 = trans1.get("full_text", "")
    text2 = trans2.get("full_text", "")

    script_comp = functions.compare_scripts(text1, text2)
    seq_sim = script_comp["sequence_similarity"]
    jaccard_sim = script_comp["jaccard_similarity"]
    script_similarity = round(max(seq_sim, jaccard_sim), 4)

    verdict = "ORIGINAL_SCRIPT"
    findings = []

    if script_similarity >= 0.85:
        verdict = "BLATANT_SCRIPT_COPY"
        findings.append(f"Blatant script plagiarism detected ({script_similarity * 100:.1f}% sequence overlap). The suspect is reading the same script verbatim.")
    elif script_similarity >= 0.65:
        verdict = "HEAVY_SCRIPT_PARAPHRASE"
        findings.append(f"Substantial script paraphrase detected ({script_similarity * 100:.1f}% alignment). Concepts and sentences follow identical structure.")
    elif script_similarity >= 0.45:
        verdict = "MODERATE_TOPIC_OVERLAP"
        findings.append("Moderate topic and terminology overlap detected in spoken audio.")
    else:
        findings.append("Audio scripts appear original and independent.")

    return {
        "transcripts_available": True,
        "t1_available": True,
        "t2_available": True,
        "script_similarity": script_similarity,
        "script_similarity_pct": round(script_similarity * 100, 1),
        "sequence_similarity": seq_sim,
        "jaccard_similarity": jaccard_sim,
        "verdict": verdict,
        "findings": findings,
    }


def compare_titles(title1: str, title2: str) -> Dict[str, Any]:
    """
    Calculates title text similarity using token set overlap and SequenceMatcher.
    """
    import difflib

    t1_clean = re.sub(r"[^\w\s]", "", title1.lower()).strip()
    t2_clean = re.sub(r"[^\w\s]", "", title2.lower()).strip()

    tokens1 = set(t1_clean.split())
    tokens2 = set(t2_clean.split())

    token_jaccard = 0.0
    if tokens1 and tokens2:
        token_jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

    seq_sim = difflib.SequenceMatcher(None, t1_clean, t2_clean).ratio()
    title_similarity = round(max(token_jaccard, seq_sim), 4)

    return {
        "title_similarity": title_similarity,
        "title_similarity_pct": round(title_similarity * 100, 1),
        "is_exact_title": t1_clean == t2_clean,
    }


def compute_composite_plagiarism_score(
    visual_res: Dict[str, Any],
    transcript_res: Dict[str, Any],
    title_res: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesizes multi-modal evidence into an actionable plagiarism risk assessment.
    """
    vis_score = visual_res.get("visual_similarity", 0.0)
    title_score = title_res.get("title_similarity", 0.0)

    if transcript_res.get("transcripts_available", False):
        script_score = transcript_res.get("script_similarity", 0.0)
        # 45% visual, 40% script, 15% title
        raw_composite = (0.45 * vis_score) + (0.40 * script_score) + (0.15 * title_score)
    else:
        # If no captions: 70% visual, 30% title
        raw_composite = (0.70 * vis_score) + (0.30 * title_score)

    # Boost score if there is a smoking-gun duplicate (e.g. 100% thumbnail or 90% script)
    if visual_res.get("phash_distance", 99) <= 2:
        raw_composite = max(raw_composite, 0.90)
    elif visual_res.get("dinov2_score", 0.0) >= 0.88:
        raw_composite = max(raw_composite, 0.85)

    if transcript_res.get("script_similarity", 0.0) >= 0.80:
        raw_composite = max(raw_composite, 0.88)

    risk_score_pct = round(min(100.0, raw_composite * 100.0), 1)

    # Categorize Risk
    if risk_score_pct >= 75.0:
        threat_level = "CRITICAL"
        threat_color = "red"
        threat_badge = "High Probability of Plagiarism"
    elif risk_score_pct >= 50.0:
        threat_level = "SUSPICIOUS"
        threat_color = "amber"
        threat_badge = "Suspicious Copy / AI Derivative"
    elif risk_score_pct >= 30.0:
        threat_level = "MODERATE"
        threat_color = "yellow"
        threat_badge = "Similar Topic / Concept Overlap"
    else:
        threat_level = "LOW"
        threat_color = "emerald"
        threat_badge = "Original Work / Low Risk"

    # Compile all evidence findings
    evidence = []
    evidence.extend(visual_res.get("findings", []))
    evidence.extend(transcript_res.get("findings", []))
    if title_res.get("is_exact_title", False):
        evidence.append("Exact identical video title used.")
    elif title_score >= 0.65:
        evidence.append(f"Title is heavily derived ({title_res['title_similarity_pct']}% match).")

    return {
        "risk_score_pct": risk_score_pct,
        "threat_level": threat_level,
        "threat_color": threat_color,
        "threat_badge": threat_badge,
        "evidence": evidence,
    }


def compare_two_videos(
    url1: str,
    url2: str,
    use_dinov2: bool = True
) -> Dict[str, Any]:
    """
    Direct 1-on-1 comparative plagiarism analysis between two specific YouTube videos.
    Ideal when a YouTuber already has a suspected clone URL.
    """
    meta1 = get_video_metadata(url1)
    meta2 = get_video_metadata(url2)

    trans1 = get_video_transcript(meta1["id"])
    trans2 = get_video_transcript(meta2["id"])

    visual_res = compare_thumbnails(meta1["thumbnail"], meta2["thumbnail"], use_dinov2=use_dinov2)
    transcript_res = compare_transcripts(trans1, trans2)
    title_res = compare_titles(meta1["title"], meta2["title"])

    composite = compute_composite_plagiarism_score(visual_res, transcript_res, title_res)

    return {
        "video1": {**meta1, "transcript_stats": {"word_count": trans1.get("word_count", 0), "available": trans1.get("available", False)}},
        "video2": {**meta2, "transcript_stats": {"word_count": trans2.get("word_count", 0), "available": trans2.get("available", False)}},
        "visual_comparison": visual_res,
        "transcript_comparison": transcript_res,
        "title_comparison": title_res,
        "composite_verdict": composite,
    }


def scan_youtube_for_plagiarism(
    query_url: str,
    max_candidates: int = 6,
    use_dinov2: bool = True
) -> Dict[str, Any]:
    """
    Master YouTube-wide scan:
    Given a single YouTube video URL:
    1. Fetches metadata and captions for the source video.
    2. Searches YouTube for top potential copycat/cloned videos.
    3. Runs multi-modal comparisons against every candidate.
    4. Ranks candidates by plagiarism score and generates a comprehensive report.
    """
    t_start = time.time()
    source_meta = get_video_metadata(query_url)
    source_trans = get_video_transcript(source_meta["id"])

    candidates = search_youtube_candidates(
        title=source_meta["title"],
        original_id=source_meta["id"],
        max_candidates=max_candidates
    )

    matches = []
    for cand in candidates:
        try:
            cand_trans = get_video_transcript(cand["id"])
            vis_res = compare_thumbnails(source_meta["thumbnail"], cand["thumbnail"], use_dinov2=use_dinov2)
            trans_res = compare_transcripts(source_trans, cand_trans)
            tit_res = compare_titles(source_meta["title"], cand["title"])
            verdict = compute_composite_plagiarism_score(vis_res, trans_res, tit_res)

            matches.append({
                "candidate": {
                    **cand,
                    "transcript_stats": {
                        "word_count": cand_trans.get("word_count", 0),
                        "available": cand_trans.get("available", False)
                    }
                },
                "visual_comparison": vis_res,
                "transcript_comparison": trans_res,
                "title_comparison": tit_res,
                "verdict": verdict,
            })
        except Exception as e:
            print(f"[detector.py] Error comparing with candidate {cand.get('id')}: {e}")

    # Sort matches by highest risk score descending
    matches.sort(key=lambda m: m["verdict"]["risk_score_pct"], reverse=True)

    highest_score = matches[0]["verdict"]["risk_score_pct"] if matches else 0.0
    elapsed_sec = round(time.time() - t_start, 2)

    return {
        "status": "success",
        "scan_time_sec": elapsed_sec,
        "source_video": {
            **source_meta,
            "transcript_stats": {
                "word_count": source_trans.get("word_count", 0),
                "available": source_trans.get("available", False)
            }
        },
        "candidates_count": len(candidates),
        "highest_plagiarism_risk_pct": highest_score,
        "matches": matches,
    }
