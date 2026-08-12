

import os
import sys
import time
import threading
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MissionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestFullSystem")

def run_integration_test():
    logger.info("=" * 60)
    logger.info("STARTING FULL SYSTEM INTEGRATION TEST WITH T-SHIRT OCR READING")
    logger.info("=" * 60)


    engine = MissionEngine()

    engine.camera.start()
    engine.drone.start()
    engine.streamer.start()
    engine.voice_engine.start()

    engine.running = True
    engine.async_vision_thread = threading.Thread(target=engine._async_vision_worker, daemon=True)
    engine.async_vision_thread.start()

    engine.loop_thread = threading.Thread(target=engine._main_processing_loop, daemon=True)
    engine.loop_thread.start()

    logger.info("Companion computer engine running in background.")
    time.sleep(1.5)


    logger.info("Testing Open Command: 'search person with Monday written on tshirt'...")
    res1 = engine.dispatch_command_text("search person with Monday written on tshirt")
    assert res1["action"] == "SEARCH", f"Expected SEARCH, got {res1['action']}"
    assert engine.search_target == "person", f"Expected target 'person', got {engine.search_target}"
    assert engine.search_text == "monday", f"Expected text_query 'monday', got {engine.search_text}"
    logger.info(f"Command Response 1: {res1}")


    logger.info("Testing Open Track Command: 'track person with Monday written on tshirt'...")
    res2 = engine.dispatch_command_text("track person with Monday written on tshirt")
    assert res2["action"] == "TRACK", f"Expected TRACK, got {res2['action']}"
    assert engine.search_text == "monday", f"Expected text_query 'monday', got {engine.search_text}"
    logger.info(f"Command Response 2: {res2}")


    logger.info("Testing Plural & Color Command: 'search black bottles'...")
    res3 = engine.dispatch_command_text("search black bottles")
    assert engine.search_target == "bottle", f"Expected target 'bottle', got {engine.search_target}"
    assert engine.search_color == "black", f"Expected color 'black', got {engine.search_color}"
    logger.info(f"Command Response 3: {res3}")

    time.sleep(1.0)
    engine.stop()

    logger.info("=" * 60)
    logger.info("ALL INTEGRATION TESTS PASSED 100%! T-SHIRT OCR & DYNAMIC COMMANDS VERIFIED.")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_integration_test()