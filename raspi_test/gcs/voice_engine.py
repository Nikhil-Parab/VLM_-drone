import time
import threading
import logging
from gcs.command_parser import CommandParser

logger = logging.getLogger(__name__)

class VoiceEngine:
    def __init__(self, wake_word="jarvis", command_callback=None):
        self.wake_word = wake_word
        self.command_parser = CommandParser(wake_word=wake_word)
        self.command_callback = command_callback
        self.running = False
        self.thread = None
        self.backend = "simulated"

        self._init_backend()

    def _init_backend(self):
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.backend = "speech_recognition"
            logger.info("SpeechRecognition microphone backend initialized.")
        except Exception as e:
            logger.info(f"Microphone offline voice backend not loaded ({e}). Operating in GCS Voice Command Pipeline mode.")
            self.backend = "simulated"

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info(f"Voice engine started in [{self.backend}] mode.")

    def _listen_loop(self):
        while self.running:
            if self.backend == "speech_recognition":
                try:
                    import speech_recognition as sr
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = self.recognizer.listen(source, timeout=3.0, phrase_time_limit=5.0)
                    text = self.recognizer.recognize_google(audio)
                    logger.info(f"Voice Audio Heard: '{text}'")
                    self.process_voice_phrase(text)
                except Exception:
                    pass
            else:
                time.sleep(1.0)

    def process_voice_phrase(self, phrase_text):

        logger.info(f"Processing Voice Command Phrase: '{phrase_text}'")
        cmd = self.command_parser.parse_command(phrase_text)
        if cmd["action"] != "UNKNOWN":
            logger.info(f"Voice Command Recognized -> Action: {cmd['action']} Params: {cmd['params']}")
            if self.command_callback:
                self.command_callback(cmd)
            return cmd
        else:
            logger.warning(f"Voice phrase could not be parsed: '{phrase_text}'")
            return cmd

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Voice engine stopped.")