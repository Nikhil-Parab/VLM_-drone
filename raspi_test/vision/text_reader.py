

import os
import sys
import re
import logging
import cv2

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

logger = logging.getLogger(__name__)


try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

class TextReader:
    def __init__(self):
        self.available = HAS_PYTESSERACT

    def read_text(self, crop_bgr):

        if not self.available or crop_bgr is None or crop_bgr.size == 0:
            return ""

        try:

            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)


            h, w = gray.shape[:2]
            if h < 100 or w < 100:
                gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


            gray = cv2.equalizeHist(gray)


            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(gray, config=custom_config)
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', extracted_text).strip()
            return clean_text
        except Exception as e:
            logger.debug(f"OCR Error: {e}")
            return ""

    def matches_target_text(self, crop_bgr, target_text):

        if not target_text:
            return True, ""

        extracted = self.read_text(crop_bgr).lower()
        target_lower = target_text.lower().strip()


        if target_lower in extracted:
            return True, extracted


        target_words = target_lower.split()
        extracted_words = extracted.split()

        for tw in target_words:
            if len(tw) >= 3 and any(tw in ew or ew in tw for ew in extracted_words):
                return True, extracted

        return False, extracted