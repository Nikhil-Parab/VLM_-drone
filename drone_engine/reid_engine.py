"""
reid_engine.py
===============
Periodic re-identification layer — Phase A/B/C optimized.

Architecture:
  - Phase C: ReID now uses ORB feature descriptors + HSV color histogram for
    embedding-based similarity (~5ms CPU) instead of SmolVLM text-tag Jaccard
    matching (~10s CPU). REID_INTERVAL_SEC drops from 8s → 0.5s — can now
    run frequently without blocking anything.
  - Phase B: generate_tag() still exists for HUD human-readable labels but
    is now greedy (do_sample=False) and trimmed to max_new_tokens=15.
  - torch.set_num_threads(2) caps intra-op threads to leave cores for the
    main tracking loop / camera thread.

Shared-model support (Phase 5/6):
  Pass shared_model and shared_processor to __init__ to reuse the
  SmolVLM instance loaded in drone_tracking_engine.py. This saves ~987 MB
  of RAM. SmolVLM is now only called for HUD label generation (rare, async),
  never for frame-to-frame ReID decisions.
"""

import re
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLMModel

# Phase B: cap intra-op threads — leave cores for camera/main loop
torch.set_num_threads(2)

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"  # <-- CHANGE VLM MODEL HERE

# PSTG-style prompt: only used for HUD label generation now (not ReID decisions)
REID_PROMPT = (
    "Summarize this person's appearance as a single compact tag inside "
    "<REID></REID>. Include clothing color, clothing type, and any "
    "distinctive item, in 5 words or fewer. "
    "Example: <REID>red jacket, blue jeans, backpack</REID>"
)  # <-- CHANGE REID PROMPT HERE

# Phase C: embedding-based ReID — interval can be much shorter now
REID_INTERVAL_SEC = 0.5     # <-- fast ORB+histogram verify cadence (was 8.0s with VLM)
SIMILARITY_THRESHOLD = 0.35  # <-- CHANGE re-ID match strictness (0-1, higher = stricter)

# H2 fix: HUD label (SmolVLM generate) has its OWN much-longer cooldown,
# completely decoupled from the fast 0.5s verify() cadence.
# The HUD label only needs occasional refresh, not real-time updates.
HUD_LABEL_INTERVAL_SEC = 30.0  # <-- CHANGE: how often to refresh the appearance HUD tag

# Weights for combined embedding score (ORB structural + HSV color)
ORB_WEIGHT   = 0.5   # <-- CHANGE structural match contribution
COLOR_WEIGHT = 0.5   # <-- CHANGE color histogram match contribution


# -----------------------------------------------------------------------
# Phase C — Embedding helpers (ORB + HSV histogram, zero new deps)
# -----------------------------------------------------------------------

_orb = cv2.ORB_create(nfeatures=200)  # module-level; created once


def _compute_embedding(crop_bgr: np.ndarray) -> Optional[dict]:
    """
    Compute a lightweight visual embedding for a BGR crop:
      - ORB descriptors  (structural / shape features)
      - HSV histogram    (color distribution features)

    Returns None if the crop is too small to describe reliably.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 16 or w < 16:
        return None

    # Resize to a fixed canonical size so descriptor counts are comparable
    canonical = cv2.resize(crop_bgr, (64, 128))

    # ORB descriptors
    gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
    _, descriptors = _orb.detectAndCompute(gray, None)

    # HSV color histogram (8 bins per channel → 512-dim vector)
    hsv = cv2.cvtColor(canonical, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    return {"descriptors": descriptors, "hist": hist.flatten()}


def _embedding_similarity(emb_a: dict, emb_b: dict) -> float:
    """
    Combined similarity score (0–1) between two embeddings.
    Combines ORB Hamming match ratio and HSV histogram correlation.
    """
    # --- Color histogram similarity ---
    hist_sim = float(np.dot(emb_a["hist"], emb_b["hist"]) /
                     (np.linalg.norm(emb_a["hist"]) * np.linalg.norm(emb_b["hist"]) + 1e-8))
    hist_sim = max(0.0, min(1.0, hist_sim))

    # --- ORB descriptor match ratio ---
    da, db = emb_a["descriptors"], emb_b["descriptors"]
    if da is None or db is None or len(da) == 0 or len(db) == 0:
        # No keypoints found — fall back to color-only
        return hist_sim

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(da, db, k=2)
    except cv2.error:
        return hist_sim

    # Lowe's ratio test
    good = 0
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good += 1

    orb_sim = good / max(min(len(da), len(db)), 1)
    orb_sim = max(0.0, min(1.0, orb_sim))

    return ORB_WEIGHT * orb_sim + COLOR_WEIGHT * hist_sim


# -----------------------------------------------------------------------
# Legacy text-tag helpers (kept for HUD label extraction only)
# -----------------------------------------------------------------------

def _extract_reid_tag(raw_text: str) -> str:
    """Pull out the <REID>...</REID> content; fall back to the raw text."""
    match = re.search(r"<reid>(.*?)</reid>", raw_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().lower()
    return raw_text.strip().lower()


class ReIDEngine:
    def __init__(self, model_id=MODEL_ID, shared_model=None, shared_processor=None,
                 shared_inference_lock=None):
        """
        Parameters
        ----------
        model_id              : HuggingFace model ID to load (if not using shared instance)
        shared_model          : pre-loaded AutoVLMModel — pass this to skip a second model load
        shared_processor      : pre-loaded AutoProcessor — must be provided with shared_model
        shared_inference_lock : threading.Lock shared with SmolObjectClassifier to prevent
                                concurrent model.generate() calls from cls_executor and
                                reid_executor threads simultaneously. Without this, both
                                callers run on the same model object at once -> starvation.
        """
        # H2 fix: shared inference lock
        self._inference_lock = shared_inference_lock or __import__('threading').Lock()

        if shared_model is not None and shared_processor is not None:
            print("[REID] Using shared SmolVLM model instance (no second load).")
            self.model = shared_model
            self.processor = shared_processor
            self.device = next(shared_model.parameters()).device
        else:
            print(f"[REID] Loading {model_id} ...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoVLMModel.from_pretrained(
                model_id, torch_dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            print(f"[REID] Loaded on {self.device}.")

        self.target_tag: Optional[str] = None        # human-readable HUD label
        self.target_embedding: Optional[dict] = None  # Phase C: visual embedding
        self.last_check_time: float = 0.0

        # H2 fix: separate cadence for HUD label generation (SmolVLM) vs verify (ORB)
        self.last_hud_label_time: float = 0.0
        # H2 diagnosis: call counter
        self._hud_call_count: int = 0
        self._hud_last_wall: float = 0.0

    # ------------------------------------------------------------------
    # Phase C: Embedding-based ReID (primary — fast, ~5ms)
    # ------------------------------------------------------------------

    def lock_target(self, cropped_frame_bgr: np.ndarray) -> str:
        """
        Called once when the target is first acquired.
        Stores the visual embedding for fast subsequent verification.
        Also generates a VLM HUD label in the background (caller's responsibility
        to submit generate_hud_label() async if desired).
        Returns the embedding-based class string for immediate use.
        """
        self.target_embedding = _compute_embedding(cropped_frame_bgr)
        # Use a plain label until VLM label arrives async
        label = self.target_tag or "target"
        print(f"[REID] Target embedding locked. HUD label: '{label}'")
        return label

    def verify(self, cropped_frame_bgr: np.ndarray) -> Tuple[bool, float, Optional[str]]:
        """
        Phase C: Fast embedding-based verification (~5ms).
        Returns (is_match: bool, similarity: float, tag: None)
        tag is None because we no longer generate a VLM tag per-verify call.
        """
        self.last_check_time = time.time()

        if self.target_embedding is None:
            return True, 1.0, None  # nothing locked yet

        candidate_emb = _compute_embedding(cropped_frame_bgr)
        if candidate_emb is None:
            return True, 1.0, None  # crop too small to judge

        similarity = _embedding_similarity(self.target_embedding, candidate_emb)
        is_match = similarity >= SIMILARITY_THRESHOLD

        print(f"[REID] embedding sim={similarity:.3f} match={is_match}")
        return is_match, similarity, None

    def should_check_now(self) -> bool:
        return (time.time() - self.last_check_time) >= REID_INTERVAL_SEC

    def should_generate_hud_now(self) -> bool:
        """H2 fix: gate SmolVLM HUD label generation on its own long-interval cooldown.
        This is INDEPENDENT of should_check_now() (ORB verify cadence).
        Only returns True once per HUD_LABEL_INTERVAL_SEC (default 30s).
        """
        return (time.time() - self.last_hud_label_time) >= HUD_LABEL_INTERVAL_SEC

    # ------------------------------------------------------------------
    # Phase B: VLM HUD label generation (demoted — async, infrequent)
    # do_sample=False (greedy), max_new_tokens=15 (was 25 + sampling)
    # ------------------------------------------------------------------

    def generate_hud_label(self, cropped_frame_bgr: np.ndarray) -> str:
        """
        Run SmolVLM to produce a compact human-readable appearance tag.
        Used ONLY for HUD display and voice-command target matching.
        NOT called in the per-frame ReID decision path.

        H2 fix:
          - Gated externally by should_generate_hud_now() (30s cooldown).
          - Uses shared_inference_lock to prevent concurrent execution with classify().
          - Logs call number, wall time, duration, gap from previous call.
        """
        # H2 diagnosis: log every call so frequency is visible in console
        self._hud_call_count += 1
        call_n = self._hud_call_count
        now_wall = time.time()
        gap = now_wall - self._hud_last_wall if self._hud_last_wall > 0 else -1.0
        self._hud_last_wall = now_wall
        self.last_hud_label_time = now_wall   # record so should_generate_hud_now() gates future calls
        if gap >= 0:
            print(f"[HUD_LABEL] call #{call_n} start  wall={now_wall:.3f}  gap_from_prev={gap:.2f}s")
        else:
            print(f"[HUD_LABEL] call #{call_n} start  wall={now_wall:.3f}  (first call)")
        _t0 = time.perf_counter()

        rgb = cv2.cvtColor(cropped_frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": REID_PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)

        # H2 fix: shared lock ensures classify() and generate_hud_label() never overlap
        with self._inference_lock:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=False,
                )

        dt = time.perf_counter() - _t0
        print(f"[HUD_LABEL] call #{call_n} done   dt={dt:.2f}s")

        new_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
        decoded = self.processor.decode(new_ids, skip_special_tokens=True)
        answer = decoded.lower()
        if "assistant" in answer:
            answer = answer.split("assistant")[-1]

        tag = _extract_reid_tag(answer)
        self.target_tag = tag
        return tag

    # Backwards-compat alias used by drone_tracking_engine.py
    def generate_tag(self, cropped_frame_bgr: np.ndarray) -> str:
        return self.generate_hud_label(cropped_frame_bgr)


# ------------------------------------------------------------------
# UPGRADE PATH NOTE:
# The ORB+histogram embedding here is a large improvement over Jaccard
# text matching (~5ms vs ~10s, plus runs every 0.5s vs every 8s).
# If re-ID accuracy needs further improvement (e.g. crowded scenes with
# similar clothing), the natural next step is MobileCLIP-S0 — a proper
# visual embedding model that runs ~10ms on CPU and gives much stronger
# discriminability. The call signature of lock_target/verify stays the same.
# ------------------------------------------------------------------
