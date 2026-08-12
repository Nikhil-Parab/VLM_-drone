"""
SmolVLM Natural Language Command Engine for DroneEye
=====================================================
Architecture: Idefics3ForConditionalGeneration (SmolVLM)
Tokenizer:    Idefics3Processor / GPT2Tokenizer
EOS token:    <end_of_utterance>

Converts free-form natural language drone commands into structured
action dicts that MissionEngine.execute_mission_command() understands.

Supported actions: ARM, DISARM, TAKEOFF, LAND, RTL, HOLD,
                   SEARCH, TRACK, SCAN_GEO, FLY_TO, UNKNOWN

Example prompts handled:
  "fly up to 20 meters and look for a red car"
  "arm the motors then take off to 15m"
  "find the person near the gate"
  "go home"
  "follow that blue backpack"
  "hold position and scan the terrain below"
  "jarvis, track the red bicycle"
"""

import os
import re
import json
import logging
import threading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fast-path regex — catches dead-simple single-keyword commands instantly
# without loading the heavy model (saves ~2–3 seconds and RAM)
# ---------------------------------------------------------------------------
_FAST_PATH_RULES = [
    (r"^(arm|arm motors?|start motors?)$",              {"action": "ARM",      "params": {}}),
    (r"^(disarm|stop motors?|disarm motors?)$",          {"action": "DISARM",   "params": {}}),
    (r"^(land|touch ?down)$",                            {"action": "LAND",     "params": {}}),
    (r"^(rtl|return home|go home|return to launch)$",    {"action": "RTL",      "params": {}}),
    (r"^(hold|loiter|pause|hover|hold position)$",       {"action": "HOLD",     "params": {}}),
    (r"^(scan ?geo|scan terrain|scan environment)$",     {"action": "SCAN_GEO", "params": {}}),
]

# ---------------------------------------------------------------------------
# System / user prompt injected into the model chat template
# ---------------------------------------------------------------------------
_SYSTEM_CONTENT = (
    "You are the flight command parser for a hexacopter drone. "
    "Convert natural language instructions into ONE JSON object with NO extra text.\n\n"
    "Schema: {\"action\": \"<ACTION>\", \"params\": {<PARAMS>}}\n\n"
    "Actions and params:\n"
    "  ARM          -> {}\n"
    "  DISARM       -> {}\n"
    "  TAKEOFF      -> {\"altitude\": <float, default 10.0>}\n"
    "  LAND         -> {}\n"
    "  RTL          -> {}\n"
    "  HOLD         -> {}\n"
    "  SCAN_GEO     -> {}\n"
    "  SEARCH       -> {\"target\": \"<coco class>\", \"color\": \"<color or null>\", \"text_query\": \"<text on object or null>\"}\n"
    "  TRACK        -> {\"target\": \"<coco class or null>\", \"target_id\": <int or null>, \"color\": \"<color or null>\"}\n"
    "  FLY_TO       -> {\"lat\": <float>, \"lon\": <float>, \"alt\": <float, default 10.0>}\n"
    "  UNKNOWN      -> {}\n\n"
    "Rules:\n"
    "- Output ONLY the raw JSON object, nothing else.\n"
    "- If the prompt has multiple steps, return only the FIRST actionable command.\n"
    "- 'target' must be a COCO class: person, car, truck, bus, bicycle, motorcycle, boat, backpack, cell phone, bottle, book, dog, cat, etc.\n"
    "- If no valid action can be determined, use UNKNOWN.\n"
)


class SmolVLMEngine:
    """
    Lazy-loading SmolVLM (Idefics3) inference engine for NL drone commands.

    The 500 MB model is only loaded from disk on the first parse_command()
    call — startup is instant, and memory is only used when needed.
    """

    def __init__(self, model_dir: str, device: str = "cpu", max_new_tokens: int = 128):
        self.model_dir = model_dir
        self.device = device
        self.max_new_tokens = max_new_tokens

        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._loaded = False
        self._load_failed = False

        logger.info(f"SmolVLMEngine initialised. Model will lazy-load from: {model_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_command(self, text: str) -> dict:
        """
        Convert natural language string -> mission command dict.

        Returns:
            {
              "action": str,          # ARM / TAKEOFF / SEARCH / etc.
              "params": dict,         # action-specific parameters
              "raw":    str,          # original input text
              "source": str           # "smolvlm_fastpath" | "smolvlm" | "smolvlm_unavailable"
            }
        """
        if not text or not isinstance(text, str):
            return {"action": "UNKNOWN", "params": {}, "raw": str(text), "source": "smolvlm"}

        raw   = text.strip()
        clean = raw.lower().strip()

        # Strip wake word prefix (e.g. "jarvis, ...")
        clean = re.sub(r"^jarvis[,\s]+", "", clean).strip()

        # --- Fast path -------------------------------------------------------
        for pattern, result in _FAST_PATH_RULES:
            if re.match(pattern, clean):
                logger.info(f"[SmolVLM fast-path] '{clean}' -> {result['action']}")
                return {**result, "raw": raw, "source": "smolvlm_fastpath"}

        # --- Model path ------------------------------------------------------
        if self._load_failed:
            logger.warning("SmolVLM unavailable (load failed) — returning UNKNOWN.")
            return {"action": "UNKNOWN", "params": {}, "raw": raw, "source": "smolvlm_unavailable"}

        self._ensure_loaded()
        if self._load_failed:
            return {"action": "UNKNOWN", "params": {}, "raw": raw, "source": "smolvlm_unavailable"}

        try:
            json_str = self._run_inference(raw)
            cmd = self._parse_json_response(json_str)
            cmd["raw"]    = raw
            cmd["source"] = "smolvlm"
            logger.info(f"[SmolVLM] '{raw}' -> action={cmd['action']} params={cmd['params']}")
            return cmd
        except Exception as e:
            logger.error(f"SmolVLM inference error: {e}", exc_info=True)
            return {"action": "UNKNOWN", "params": {}, "raw": raw, "source": "smolvlm_error"}

    def is_ready(self) -> bool:
        """True once the model has been successfully loaded."""
        return self._loaded and not self._load_failed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        """Thread-safe lazy load."""
        with self._lock:
            if self._loaded or self._load_failed:
                return
            self._load_model()

    def _load_model(self):
        logger.info(f"Loading SmolVLM (Idefics3) from {self.model_dir} ...")
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch

            self._processor = AutoProcessor.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )

            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_dir,
                local_files_only=True,
                torch_dtype=torch.bfloat16,   # model was saved as bfloat16
                low_cpu_mem_usage=True,
                device_map=self.device,
            )
            self._model.eval()
            self._loaded = True
            logger.info("SmolVLM model loaded successfully ✓")

        except Exception as e:
            self._load_failed = True
            logger.error(f"SmolVLM load failed: {e}", exc_info=True)

    def _run_inference(self, user_text: str) -> str:
        """
        Build a chat-style prompt using the Idefics3 chat template,
        run greedy decoding, return the generated text.
        """
        import torch

        # Build messages in the Idefics3 / SmolVLM chat format (text only)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _SYSTEM_CONTENT},
                    {"type": "text", "text": f"User command: {user_text}"},
                ],
            }
        ]

        # Apply chat template
        prompt = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # EOS token for Idefics3 is <end_of_utterance> (token id from tokenizer)
        eos_id = self._processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=self._processor.tokenizer.pad_token_id or eos_id,
            )

        # Decode only the newly generated tokens
        input_len  = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _parse_json_response(self, text: str) -> dict:
        """Extract and validate the JSON action dict from raw model output."""
        # Find the first {...} block (model may add prose around it)
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not match:
            logger.warning(f"SmolVLM returned no JSON block: '{text}'")
            return {"action": "UNKNOWN", "params": {}}

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            # Try to extract a larger block in case of nested braces
            match2 = re.search(r'\{.*\}', text, re.DOTALL)
            try:
                data = json.loads(match2.group(0)) if match2 else {}
            except Exception:
                logger.warning(f"SmolVLM JSON parse failed: '{text}'")
                return {"action": "UNKNOWN", "params": {}}

        action = str(data.get("action", "UNKNOWN")).upper().strip()
        params = data.get("params", {})
        if not isinstance(params, dict):
            params = {}

        # ---- Validate / normalise per action --------------------------------
        if action == "TAKEOFF":
            try:
                params["altitude"] = float(params.get("altitude", 10.0))
            except (TypeError, ValueError):
                params["altitude"] = 10.0

        elif action == "SEARCH":
            params["target"]     = params.get("target") or None
            params["color"]      = params.get("color") or None
            params["text_query"] = params.get("text_query") or None

        elif action == "TRACK":
            params["target"]     = params.get("target") or None
            params["color"]      = params.get("color") or None
            raw_id = params.get("target_id")
            try:
                params["target_id"] = int(raw_id) if raw_id is not None else None
            except (TypeError, ValueError):
                params["target_id"] = None

        elif action == "FLY_TO":
            try:
                params["lat"] = float(params["lat"])
                params["lon"] = float(params["lon"])
                params["alt"] = float(params.get("alt", 10.0))
            except (KeyError, TypeError, ValueError):
                action = "UNKNOWN"
                params = {}

        allowed = {"ARM","DISARM","TAKEOFF","LAND","RTL","HOLD","SCAN_GEO",
                   "SEARCH","TRACK","FLY_TO","UNKNOWN"}
        if action not in allowed:
            action = "UNKNOWN"
            params = {}

        return {"action": action, "params": params}
