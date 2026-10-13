"""
functions.py - Comprehensive Content Plagiarism and AI-Modification Detection Suite.

Contains algorithms for:
1. Thumbnail / Image Analysis:
   - Perceptual Hashing (aHash, pHash, dHash) for exact and near-duplicate detection.
   - Meta DINOv2 self-supervised embeddings for AI-modified, stylized, or inpainted images.
   - OpenAI CLIP embeddings for semantic concept comparison.
   - SIFT + RANSAC geometric homography for structural keypoint matching.
   - Multi-tiered unified plagiarism analyzer.

2. Video Content Copy Detection (VVCD):
   - Keyframe / uniform temporal frame sampling via OpenCV.
   - Frame sequence embedding generation.
   - Temporal segment alignment & sequence matching (locates exact timestamps copied).

3. Audio / Script Plagiarism:
   - Automatic speech-to-text transcription via Whisper.
   - Lexical & semantic sequence alignment for plagiarized scripts / voiceovers.
"""

import os
import io
import difflib
import re
from typing import Union, List, Tuple, Dict, Any, Optional
import urllib.request

import numpy as np
from PIL import Image
import cv2
import imagehash
import torch
import torchvision.transforms as T

# Global lazy cache for heavy deep learning models
_CACHED_MODELS: Dict[str, Any] = {
    "dinov2_model": None,
    "dinov2_transform": None,
    "clip_model": None,
    "clip_processor": None,
    "whisper_model": None,
}


def _get_device() -> str:
    """Returns CUDA if available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _extract_tensor(features: Any) -> torch.Tensor:
    """
    Robust tensor extractor across different versions of Hugging Face transformers.
    In transformers 4.x it returns a Tensor directly, while in 5.x it returns
    BaseModelOutputWithPooling.
    """
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if isinstance(features, (list, tuple)):
        return features[0]
    return torch.as_tensor(features)


# ============================================================================
# Model Getters (Lazy-loaded to keep imports instantaneous)
# ============================================================================

def get_dinov2():
    """
    Loads Meta's DINOv2 ViT-S/14 model.
    DINOv2 is the state-of-the-art self-supervised vision model for detecting
    visual structure, spatial layout, and AI-modified images.
    """
    if _CACHED_MODELS["dinov2_model"] is None:
        device = _get_device()
        print(f"[functions.py] Loading DINOv2 on {device}...")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.to(device)
        model.eval()

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        _CACHED_MODELS["dinov2_model"] = model
        _CACHED_MODELS["dinov2_transform"] = transform

    return _CACHED_MODELS["dinov2_model"], _CACHED_MODELS["dinov2_transform"]


def get_clip(model_id: str = "openai/clip-vit-base-patch32"):
    """
    Loads OpenAI's CLIP model and processor.
    CLIP excels at semantic / conceptual visual similarity.
    """
    if _CACHED_MODELS["clip_model"] is None:
        from transformers import CLIPModel, CLIPProcessor
        device = _get_device()
        print(f"[functions.py] Loading CLIP ({model_id}) on {device}...")
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.eval()

        _CACHED_MODELS["clip_model"] = model
        _CACHED_MODELS["clip_processor"] = processor

    return _CACHED_MODELS["clip_model"], _CACHED_MODELS["clip_processor"]


def get_whisper(model_size: str = "base"):
    """
    Loads OpenAI's Whisper model for speech-to-text transcription.
    """
    if _CACHED_MODELS["whisper_model"] is None:
        import whisper
        device = _get_device()
        print(f"[functions.py] Loading Whisper ({model_size}) on {device}...")
        _CACHED_MODELS["whisper_model"] = whisper.load_model(model_size, device=device)

    return _CACHED_MODELS["whisper_model"]


# ============================================================================
# Image Utility Functions
# ============================================================================

def load_image(img_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Flexible image loader. Accepts:
    - Local file path (e.g. 'thumbnails/img1.jpg')
    - Remote HTTP/HTTPS URL (e.g. YouTube thumbnail direct URL)
    - Existing PIL Image
    - NumPy image array (from OpenCV)
    """
    if isinstance(img_input, Image.Image):
        return img_input.convert("RGB")
    elif isinstance(img_input, np.ndarray):
        if len(img_input.shape) == 3 and img_input.shape[2] == 3:
            # OpenCV BGR to RGB
            return Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        return Image.fromarray(img_input).convert("RGB")
    elif isinstance(img_input, str):
        # Support direct YouTube video URLs
        if "youtube.com" in img_input or "youtu.be" in img_input:
            if not img_input.endswith((".jpg", ".png", ".webp")):
                return get_youtube_thumbnail(img_input)

        if img_input.startswith("http://") or img_input.startswith("https://"):
            req = urllib.request.Request(
                img_input,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as resp:
                image_data = resp.read()
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Image not found at path: {img_input}")
            return Image.open(img_input).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(img_input)}")


def extract_youtube_video_id(url_or_id: str) -> str:
    """
    Extracts the 11-character YouTube video ID from various URL formats
    (standard, shorts, youtu.be, or raw ID).
    """
    cleaned = url_or_id.strip()
    if len(cleaned) == 11 and not ("/" in cleaned or "?" in cleaned or "=" in cleaned):
        return cleaned

    patterns = [
        r"(?:v=|\/vi\/|\/embed\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/watch\?.*v=([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return cleaned


def get_youtube_thumbnail(url_or_id: str) -> Image.Image:
    """
    Fetches the highest available resolution thumbnail for a YouTube video URL or ID.
    Attempts: maxresdefault -> hqdefault -> mqdefault -> default.
    """
    video_id = extract_youtube_video_id(url_or_id)
    resolutions = ["maxresdefault", "hqdefault", "mqdefault", "default"]

    for res in resolutions:
        thumb_url = f"https://img.youtube.com/vi/{video_id}/{res}.jpg"
        try:
            req = urllib.request.Request(
                thumb_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            # YouTube returns a 120x90 placeholder if maxresdefault doesn't exist
            if res == "maxresdefault" and img.size == (120, 90):
                continue
            return img
        except Exception:
            continue

    raise ValueError(f"Could not retrieve thumbnail for YouTube video ID: {video_id}")



# ============================================================================
# Algorithm 1: Perceptual Hashing (Tier 1 - Fast exact / near-duplicate filter)
# ============================================================================

def compare_perceptual_hash(
    img1: Union[str, Image.Image],
    img2: Union[str, Image.Image]
) -> Dict[str, Any]:
    """
    Computes perceptual hashes (pHash, dHash, aHash).
    Fast baseline check:
    - Hamming distance = 0 -> Identical images
    - Hamming distance <= 5 -> Minor recompression / resize / slight crop
    - Hamming distance > 10 -> Visually altered (AI styling, inpainting, or different)
    """
    image_a = load_image(img1)
    image_b = load_image(img2)

    phash1 = imagehash.phash(image_a)
    phash2 = imagehash.phash(image_b)
    phash_dist = phash1 - phash2

    dhash1 = imagehash.dhash(image_a)
    dhash2 = imagehash.dhash(image_b)
    dhash_dist = dhash1 - dhash2

    ahash1 = imagehash.average_hash(image_a)
    ahash2 = imagehash.average_hash(image_b)
    ahash_dist = ahash1 - ahash2

    is_identical = (phash_dist == 0 and dhash_dist == 0)
    is_near_duplicate = (phash_dist <= 5 or dhash_dist <= 5)

    return {
        "phash_distance": int(phash_dist),
        "dhash_distance": int(dhash_dist),
        "ahash_distance": int(ahash_dist),
        "is_identical": is_identical,
        "is_near_duplicate": is_near_duplicate,
    }


# Backwards compatibility functions from original functions.py
def check_image_similarity(image_path_1: str, image_path_2: str) -> None:
    res = compare_perceptual_hash(image_path_1, image_path_2)
    print(f"Hamming Distance (pHash): {res['phash_distance']}")
    if res["is_identical"]:
        print("Result: The images are identical.")
    elif res["is_near_duplicate"]:
        print("Result: The images are very similar (likely duplicates with minor compression or resizing).")
    else:
        print("Result: The images are different.")


def check_image_similarity_from_imgs(img1: Image.Image, img2: Image.Image) -> None:
    res = compare_perceptual_hash(img1, img2)
    print(f"Hamming Distance (pHash): {res['phash_distance']}")
    if res["is_identical"]:
        print("Result: The images are identical.")
    elif res["is_near_duplicate"]:
        print("Result: The images are very similar (likely duplicates with minor compression or resizing).")
    else:
        print("Result: The images are different.")


# ============================================================================
# Algorithm 2: DINOv2 Deep Structural Similarity (Tier 2 - Best for AI Remixing)
# ============================================================================

def compare_dinov2_similarity(
    img1: Union[str, Image.Image],
    img2: Union[str, Image.Image],
    threshold: float = 0.85
) -> Tuple[bool, float]:
    """
    Extracts Meta DINOv2 self-supervised visual embeddings and calculates Cosine Similarity.
    
    Why DINOv2 is superior for AI-modified images:
    - Retains geometric composition, spatial layout, and fine-grained patches.
    - Robust against artistic filters, Stable Diffusion / Midjourney img2img restyling,
      lighting adjustments, and background modifications.
    
    Returns:
        (is_match: bool, similarity_score: float)
    """
    model, transform = get_dinov2()
    device = _get_device()

    image_a = load_image(img1)
    image_b = load_image(img2)

    tensor_a = transform(image_a).unsqueeze(0).to(device)
    tensor_b = transform(image_b).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_a = model(tensor_a)
        feat_b = model(tensor_b)

    # L2 Normalization
    feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
    feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)

    similarity = torch.nn.functional.cosine_similarity(feat_a, feat_b).item()
    similarity = float(similarity)

    return (similarity >= threshold, similarity)


# ============================================================================
# Algorithm 3: CLIP Semantic Similarity
# ============================================================================

def compare_clip_similarity(
    img1: Union[str, Image.Image],
    img2: Union[str, Image.Image],
    threshold: float = 0.85
) -> Tuple[bool, float]:
    """
    Compares two images using OpenAI CLIP embeddings.
    Measures semantic conceptual alignment between two images.
    """
    model, processor = get_clip()
    device = _get_device()

    image_a = load_image(img1)
    image_b = load_image(img2)

    inputs1 = processor(images=image_a, return_tensors="pt").to(device)
    inputs2 = processor(images=image_b, return_tensors="pt").to(device)

    with torch.no_grad():
        features1 = _extract_tensor(model.get_image_features(**inputs1))
        features2 = _extract_tensor(model.get_image_features(**inputs2))

    features1 = features1 / features1.norm(dim=-1, keepdim=True)
    features2 = features2 / features2.norm(dim=-1, keepdim=True)

    similarity_score = (features1 @ features2.T).item()
    similarity_score = float(similarity_score)

    return (similarity_score >= threshold, similarity_score)


# Preserves original signature
def check_semantic_similarity(
    image_path_1: str,
    image_path_2: str,
    threshold: float = 0.85
) -> Tuple[bool, float]:
    return compare_clip_similarity(image_path_1, image_path_2, threshold=threshold)


# ============================================================================
# Algorithm 4: SIFT + RANSAC Homography (Tier 3 - Geometric Anchor Points)
# ============================================================================

def detect_geometric_feature_matches(
    img1: Union[str, Image.Image],
    img2: Union[str, Image.Image],
    min_inliers: int = 15
) -> Dict[str, Any]:
    """
    Uses OpenCV SIFT (Scale-Invariant Feature Transform) with FLANN matching and
    RANSAC Homography estimation.
    
    Why this detects AI modifications:
    Even when an AI regenerates textures, key invariant landmarks (such as facial
    features, logo positions, text outlines, character poses, or silhouette contours)
    often retain identical relative geometry. RANSAC finds points that fit a consistent
    geometric transformation.
    """
    image_a = np.array(load_image(img1))
    image_b = np.array(load_image(img2))

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_a, None)
    kp2, des2 = sift.detectAndCompute(gray_b, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {
            "keypoints_img1": len(kp1) if kp1 else 0,
            "keypoints_img2": len(kp2) if kp2 else 0,
            "good_matches": 0,
            "ransac_inliers": 0,
            "inlier_ratio": 0.0,
            "geometric_match": False,
        }

    # FLANN matcher with Lowe's ratio test
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    inliers_count = 0
    inlier_ratio = 0.0
    geometric_match = False

    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers_count = int(np.sum(mask))
            inlier_ratio = float(inliers_count / len(good_matches))
            geometric_match = inliers_count >= min_inliers

    return {
        "keypoints_img1": len(kp1),
        "keypoints_img2": len(kp2),
        "good_matches": len(good_matches),
        "ransac_inliers": inliers_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "geometric_match": geometric_match,
    }


# ============================================================================
# Algorithm 5: Unified Thumbnail Plagiarism Analyzer
# ============================================================================

def analyze_thumbnail_plagiarism(
    img1: Union[str, Image.Image],
    img2: Union[str, Image.Image],
    dinov2_threshold: float = 0.85,
    clip_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Comprehensive multi-stage plagiarism detection for YouTube thumbnails:
    Combines:
    1. Perceptual Hashing (Exact / Resized matches)
    2. DINOv2 (Self-supervised structural / AI-modified matching)
    3. CLIP (Semantic match)
    4. SIFT + RANSAC (Geometric landmark correspondence)

    Returns a structured verdict with detailed score breakdown and human-readable explanation.
    """
    hash_result = compare_perceptual_hash(img1, img2)
    _, dinov2_score = compare_dinov2_similarity(img1, img2, threshold=dinov2_threshold)
    _, clip_score = compare_clip_similarity(img1, img2, threshold=clip_threshold)
    sift_result = detect_geometric_feature_matches(img1, img2)

    # Determine Verdict
    verdict = "ORIGINAL_OR_DIFFERENT"
    explanation = "The thumbnails appear to be distinct with no significant structural or conceptual copying detected."

    if hash_result["is_identical"]:
        verdict = "EXACT_DUPLICATE"
        explanation = "The thumbnails are 100% identical byte-for-byte or exact visual replicas."
    elif hash_result["is_near_duplicate"]:
        verdict = "NEAR_DUPLICATE"
        explanation = "The suspect thumbnail is an exact crop, minor resize, or recompressed version of the original."
    elif dinov2_score >= dinov2_threshold or (dinov2_score >= 0.80 and sift_result["geometric_match"]):
        verdict = "AI_MODIFIED_OR_STYLED_COPY"
        explanation = (
            f"Strong evidence of AI-generation / style transfer or direct tracing. "
            f"DINOv2 similarity is high ({dinov2_score * 100:.1f}%) and "
            f"{sift_result['ransac_inliers']} geometric anchor inliers match the original layout."
        )
    elif clip_score >= clip_threshold:
        verdict = "HIGH_SEMANTIC_SIMILARITY"
        explanation = (
            f"The thumbnails share very similar concepts/subject matter (CLIP: {clip_score * 100:.1f}%), "
            f"though the visual composition or assets may have been recreated."
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "metrics": {
            "phash_distance": hash_result["phash_distance"],
            "dinov2_similarity": round(dinov2_score, 4),
            "clip_similarity": round(clip_score, 4),
            "sift_inliers": sift_result["ransac_inliers"],
            "sift_inlier_ratio": sift_result["inlier_ratio"],
        },
        "raw_results": {
            "hash": hash_result,
            "sift": sift_result,
        }
    }


# ============================================================================
# Algorithm 6: Video Copy Detection (Visual Video Copy Detection - VVCD)
# ============================================================================

def extract_video_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int = 600
) -> List[Tuple[float, Image.Image]]:
    """
    Extracts frames from a local video file at a specified sampling rate (default 1 frame per second).
    Returns a list of tuples: (timestamp_in_seconds, PIL.Image).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    frame_interval = max(1, int(round(video_fps / fps)))
    extracted = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or len(extracted) >= max_frames:
            break

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / video_fps
            # Convert BGR to RGB PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            extracted.append((round(timestamp_sec, 2), pil_img))

        frame_idx += 1

    cap.release()
    return extracted


def compute_video_embeddings(
    video_path: str,
    model_type: str = "dinov2",
    fps: float = 1.0,
    max_frames: int = 600
) -> Tuple[List[float], torch.Tensor]:
    """
    Samples frames uniformly and encodes each frame into a normalized embedding vector.
    Returns:
        (timestamps: List[float], embeddings: torch.Tensor of shape [num_frames, embedding_dim])
    """
    frames = extract_video_frames(video_path, fps=fps, max_frames=max_frames)
    if not frames:
        raise ValueError(f"No frames could be extracted from {video_path}")

    timestamps = [t for t, _ in frames]
    pil_images = [img for _, img in frames]
    device = _get_device()

    if model_type.lower() == "dinov2":
        model, transform = get_dinov2()
        # Process in batches of 16 to avoid GPU OOM
        batch_size = 16
        all_embeddings = []

        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            tensors = torch.stack([transform(img) for img in batch]).to(device)
            with torch.no_grad():
                feats = model(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embeddings.append(feats.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
    else:
        # Fallback to CLIP
        model, processor = get_clip()
        batch_size = 16
        all_embeddings = []

        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            with torch.no_grad():
                raw_feats = model.get_image_features(**inputs)
                feats = _extract_tensor(raw_feats)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embeddings.append(feats.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)

    return timestamps, embeddings


def detect_video_copy(
    video_1_path: str,
    video_2_path: str,
    model_type: str = "dinov2",
    similarity_threshold: float = 0.82,
    min_consecutive_sec: int = 3,
    fps: float = 1.0
) -> Dict[str, Any]:
    """
    Detects if segments of Video 1 were copied and pasted into Video 2.
    
    Pipeline:
    1. Uniformly sample frames at `fps` from both videos.
    2. Compute dense frame embeddings (DINOv2 or CLIP).
    3. Generate the 2D cross-video similarity matrix: S[i, j] = cosine(frame1_i, frame2_j).
    4. Run temporal sequence matching to find contiguous blocks where frames match.
    
    Returns:
        A structured report containing:
        - List of matched segments (timestamps in video 1 and video 2)
        - Total copied duration
        - Copied percentage for both videos
        - Final verdict
    """
    times1, emb1 = compute_video_embeddings(video_1_path, model_type=model_type, fps=fps)
    times2, emb2 = compute_video_embeddings(video_2_path, model_type=model_type, fps=fps)

    # Compute pairwise similarity matrix: shape [N1, N2]
    sim_matrix = (emb1 @ emb2.T).numpy()

    n1, n2 = sim_matrix.shape
    matched_segments = []
    min_frames = max(1, int(round(min_consecutive_sec * fps)))

    # Scan for diagonal runs (clips played at standard speed)
    visited_matrix = np.zeros_like(sim_matrix, dtype=bool)

    for i in range(n1):
        for j in range(n2):
            if visited_matrix[i, j]:
                continue

            if sim_matrix[i, j] >= similarity_threshold:
                # Trace diagonal run
                run_len = 0
                total_sim = 0.0
                curr_i, curr_j = i, j

                while (
                    curr_i < n1
                    and curr_j < n2
                    and sim_matrix[curr_i, curr_j] >= similarity_threshold
                ):
                    visited_matrix[curr_i, curr_j] = True
                    total_sim += sim_matrix[curr_i, curr_j]
                    run_len += 1
                    curr_i += 1
                    curr_j += 1

                if run_len >= min_frames:
                    v1_start = times1[i]
                    v1_end = times1[curr_i - 1]
                    v2_start = times2[j]
                    v2_end = times2[curr_j - 1]
                    duration = round(run_len / fps, 2)
                    avg_sim = round(float(total_sim / run_len), 4)

                    matched_segments.append({
                        "v1_indices": (i, curr_i),
                        "v2_indices": (j, curr_j),
                        "video_1_segment": {"start_sec": v1_start, "end_sec": v1_end},
                        "video_2_segment": {"start_sec": v2_start, "end_sec": v2_end},
                        "duration_sec": duration,
                        "avg_similarity": avg_sim,
                    })

    # Sort segments by duration descending and filter redundant overlaps
    matched_segments.sort(key=lambda s: s["duration_sec"], reverse=True)
    filtered_segments = []
    v1_mask = np.zeros(n1, dtype=bool)
    v2_mask = np.zeros(n2, dtype=bool)

    for seg in matched_segments:
        s1, e1 = seg["v1_indices"]
        s2, e2 = seg["v2_indices"]
        seg_v1 = np.zeros(n1, dtype=bool)
        seg_v1[s1:e1] = True
        
        # Check if already mostly covered
        overlap = np.sum(seg_v1 & v1_mask)
        if overlap / max(1, (e1 - s1)) < 0.6:
            v1_mask[s1:e1] = True
            v2_mask[s2:e2] = True
            del seg["v1_indices"]
            del seg["v2_indices"]
            filtered_segments.append(seg)

    # Sort final segments chronologically by video 1 start time
    filtered_segments.sort(key=lambda s: s["video_1_segment"]["start_sec"])

    # Accurate non-overlapping durations and percentages
    v1_total_duration = times1[-1] if times1 else 0.0
    v2_total_duration = times2[-1] if times2 else 0.0
    total_copied_sec = round(float(np.sum(v1_mask) / fps), 2)

    v1_copied_pct = min(100.0, round((total_copied_sec / max(v1_total_duration, 1e-5) * 100), 2)) if v1_total_duration > 0 else 0.0
    v2_copied_pct = min(100.0, round((total_copied_sec / max(v2_total_duration, 1e-5) * 100), 2)) if v2_total_duration > 0 else 0.0

    is_plagiarized = len(filtered_segments) > 0 and total_copied_sec >= min_consecutive_sec

    return {
        "is_plagiarized": is_plagiarized,
        "matched_segments_count": len(filtered_segments),
        "total_copied_duration_sec": total_copied_sec,
        "video_1_total_duration_sec": v1_total_duration,
        "video_2_total_duration_sec": v2_total_duration,
        "video_1_copied_percentage": v1_copied_pct,
        "video_2_copied_percentage": v2_copied_pct,
        "matched_segments": filtered_segments,
    }


# ============================================================================
# Algorithm 7: Audio & Transcript Plagiarism Detection
# ============================================================================

def transcribe_media(media_path: str, model_size: str = "base") -> Dict[str, Any]:
    """
    Transcribes audio from a video or audio file using OpenAI Whisper.
    Returns:
        {
            "full_text": str,
            "segments": List[{"start": float, "end": float, "text": str}]
        }
    """
    whisper_model = get_whisper(model_size=model_size)
    result = whisper_model.transcribe(media_path)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    return {
        "full_text": result["text"].strip(),
        "segments": segments,
    }


def compare_scripts(
    text1: str,
    text2: str,
    similarity_threshold: float = 0.70
) -> Dict[str, Any]:
    """
    Compares two video scripts/transcripts for narrative plagiarism.
    Uses SequenceMatcher token matching and word-level Jaccard similarity.
    """
    words1 = [w.lower() for w in text1.split() if w.isalnum()]
    words2 = [w.lower() for w in text2.split() if w.isalnum()]

    if not words1 or not words2:
        return {
            "sequence_similarity": 0.0,
            "jaccard_similarity": 0.0,
            "is_script_plagiarized": False,
        }

    # Sequence alignment
    matcher = difflib.SequenceMatcher(None, words1, words2)
    seq_ratio = matcher.ratio()

    # Vocabulary overlap (Jaccard)
    set1, set2 = set(words1), set(words2)
    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))

    is_plagiarized = seq_ratio >= similarity_threshold or jaccard >= 0.75

    return {
        "sequence_similarity": round(seq_ratio, 4),
        "jaccard_similarity": round(jaccard, 4),
        "is_script_plagiarized": is_plagiarized,
    }
