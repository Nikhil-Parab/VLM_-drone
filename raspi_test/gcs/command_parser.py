import re
import logging

logger = logging.getLogger(__name__)

SYNONYM_MAP = {
    "bottle": ["bottles", "water bottle", "flask", "canteen", "container"],
    "book": ["books", "notebook", "notebooks", "textbook", "novel", "magazine", "pad", "diary", "journal"],
    "pen": ["pens", "pencil", "pencils", "marker", "stylus", "ballpoint"],
    "person": ["people", "human", "humans", "man", "woman", "guy", "person", "pedestrian", "individual", "tshirt", "t-shirt", "shirt"],
    "car": ["cars", "vehicle", "vehicles", "automobile", "automobiles"],
    "truck": ["trucks", "lorry", "pickup"],
    "bus": ["busses", "buses"],
    "boat": ["boats", "ship", "vessel", "watercraft"],
    "cell phone": ["phone", "phones", "mobile", "cellphone", "smartphone"],
    "backpack": ["bag", "bags", "backpacks", "rucksack", "handbag"],
    "chair": ["chairs", "seat", "stool"],
    "dog": ["dogs", "puppy", "canine"],
    "cat": ["cats", "kitten", "feline"],
    "bicycle": ["bicycles", "bike", "bikes", "cycle"],
    "laptop": ["laptops", "computer", "macbook"]
}

COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white", "gray", "brown", "cyan", "dark"]

# Simple single-keyword commands that don't need NL understanding
_SIMPLE_KEYWORDS = [
    "arm", "disarm", "land", "rtl", "hold", "loiter", "pause", "hover",
    "return home", "go home", "return to launch", "touch down",
    "scan geo", "scan_geo", "scan terrain", "scan environment", "scangeo",
    "takeoff", "fly up", "ascend",
]


class CommandParser:
    """
    Hybrid command parser.

    Fast-path: dead-simple keyword commands go through regex instantly.
    Smart-path: complex natural language prompts ("fly to 20m and look for
                the red car near the gate") go through the SmolVLM engine.

    Usage:
        parser = CommandParser(wake_word="jarvis", vlm_engine=smolvlm_engine)
        cmd = parser.parse_command("find the person wearing a red shirt")
    """

    def __init__(self, wake_word="jarvis", vlm_engine=None):
        self.wake_word   = wake_word.lower()
        self.vlm_engine  = vlm_engine   # SmolVLMEngine instance (optional)

    def parse_command(self, text_input):
        if not text_input or not isinstance(text_input, str):
            return {"action": "UNKNOWN", "params": {}, "raw": str(text_input)}

        raw_text   = text_input.strip()
        clean_text = raw_text.lower()

        # Strip wake word
        is_voice = self.wake_word in clean_text
        if is_voice:
            clean_text = clean_text.replace(self.wake_word, "").strip()
        clean_text = re.sub(r'^[^\w\s]+', '', clean_text).strip()

        # ------------------------------------------------------------------ #
        # Fast path — regex for simple/known single commands                  #
        # ------------------------------------------------------------------ #
        fast_result = self._fast_parse(clean_text, raw_text, is_voice)
        if fast_result is not None:
            return fast_result

        # ------------------------------------------------------------------ #
        # Smart path — route to SmolVLM for natural language understanding    #
        # ------------------------------------------------------------------ #
        if self.vlm_engine is not None:
            logger.info(f"[CommandParser] Routing to SmolVLM: '{raw_text}'")
            cmd = self.vlm_engine.parse_command(raw_text)
            cmd["source"] = cmd.get("source", "smolvlm")
            if is_voice:
                cmd["source"] += "_voice"
            return cmd

        # ------------------------------------------------------------------ #
        # Fallback — original regex SEARCH/TRACK parsing                      #
        # ------------------------------------------------------------------ #
        return self._regex_search_track(clean_text, raw_text, is_voice)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fast_parse(self, clean_text, raw_text, is_voice):
        """Returns a command dict for known simple commands, or None."""
        src = "voice" if is_voice else "text"

        if any(clean_text == kw or clean_text.startswith(kw)
               for kw in ["arm", "arm motors", "start motors"]):
            return {"action": "ARM", "params": {}, "raw": raw_text, "source": src}

        if any(clean_text == kw or clean_text.startswith(kw)
               for kw in ["disarm", "stop motors", "disarm motors"]):
            return {"action": "DISARM", "params": {}, "raw": raw_text, "source": src}

        takeoff_match = re.search(
            r"(?:takeoff|fly up|ascend)(?:\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*m(?:eters)?)?",
            clean_text
        )
        if takeoff_match and not any(
            kw in clean_text for kw in ["find", "search", "look", "track", "follow"]
        ):
            alt = float(takeoff_match.group(1)) if takeoff_match.group(1) else 10.0
            return {"action": "TAKEOFF", "params": {"altitude": alt}, "raw": raw_text, "source": src}

        if any(kw in clean_text for kw in ["land", "touch down"]) and len(clean_text.split()) <= 3:
            return {"action": "LAND", "params": {}, "raw": raw_text, "source": src}

        if any(kw in clean_text for kw in ["rtl", "return home", "return to launch", "go home"]):
            return {"action": "RTL", "params": {}, "raw": raw_text, "source": src}

        if any(clean_text == kw for kw in ["hold", "loiter", "pause", "hover"]):
            return {"action": "HOLD", "params": {}, "raw": raw_text, "source": src}

        if any(kw in clean_text for kw in ["scan_geo", "scan geo", "scan geography",
                                            "scan environment", "scan terrain", "scangeo"]):
            return {"action": "SCAN_GEO", "params": {"terrain_type": "all"}, "raw": raw_text, "source": src}

        # Track by numeric ID only ("track 3", "follow id 5")
        track_id_match = re.match(r"^(?:track|follow|lock)\s+(?:id\s*)?(\d+)$", clean_text)
        if track_id_match:
            return {"action": "TRACK",
                    "params": {"target_id": int(track_id_match.group(1))},
                    "raw": raw_text, "source": src}

        return None  # not a fast-path command — go to SmolVLM

    def _regex_search_track(self, clean_text, raw_text, is_voice):
        """Original regex SEARCH/TRACK parser used as final fallback."""
        src = "voice" if is_voice else "text"

        if clean_text.startswith(("track", "follow", "lock")):
            requested_action = "TRACK"
            query_body = re.sub(r'^(?:track|follow|lock on|lock)\s+(?:id\s*)?', '', clean_text).strip()
        else:
            requested_action = "SEARCH"
            query_body = re.sub(
                r'^(?:search|find|scan for|look for|locate|detect)\s+(?:a|the|specific|any)?\s*',
                '', clean_text
            ).strip()

        ocr_text_query = None
        text_clause_match = re.search(
            r"(?:with|having|written|text|named|saying|printed)\s+(?:text|word|name)?\s*['\"]?([a-zA-Z0-9]+)['\"]?",
            query_body
        )
        if text_clause_match:
            ocr_text_query = text_clause_match.group(1).strip()

        detected_color = None
        for color in COLOR_WORDS:
            if color in query_body:
                detected_color = color
                break

        normalized_target = query_body
        for base_class, synonyms in SYNONYM_MAP.items():
            if any(syn in query_body for syn in synonyms):
                normalized_target = base_class
                break

        return {
            "action": requested_action,
            "params": {
                "query": query_body,
                "target": normalized_target,
                "color": detected_color,
                "text_query": ocr_text_query
            },
            "raw": raw_text,
            "source": src
        }