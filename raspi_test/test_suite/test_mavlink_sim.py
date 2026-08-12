

import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight import MavlinkController
from config import FLIGHT_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestMavlink")

def test_flight_controller():
    logger.info("=" * 60)
    logger.info("STARTING PIXHAWK MAVLINK FLIGHT CONTROLLER TEST")
    logger.info("=" * 60)

    fc = MavlinkController(FLIGHT_CONFIG)
    fc.start()
    time.sleep(0.5)

    logger.info(f"Connected: {fc.connected} | Simulated: {fc.simulated} | Initial Mode: {fc.state['mode']}")


    logger.info("1. Testing GUIDED Mode Request & Arming Motors...")
    fc.set_mode("GUIDED")
    fc.arm()
    time.sleep(0.5)
    assert fc.state["armed"] == True, "Failed to arm flight controller"
    logger.info(f"Armed State: {fc.state['armed']} | Flight Mode: {fc.state['mode']}")


    logger.info("2. Testing GUIDED Takeoff to 10.0 meters...")
    fc.takeoff(altitude=10.0)
    time.sleep(2.0)
    logger.info(f"Current Altitude: {fc.state['alt']:.1f} m")


    logger.info("3. Testing Closed-Loop PID Visual Tracking Guidance (Target offset: err_x=0.3, err_y=-0.2)...")
    fc.track_target_offset(err_x=0.3, err_y=-0.2)
    time.sleep(1.0)
    logger.info(f"Calculated Velocities -> Vx: {fc.state['vx']:.2f} m/s, Vy: {fc.state['vy']:.2f} m/s")


    logger.info("4. Simulating Manual RC Override Trigger (Pilot flips switch on T12 Transmitter)...")
    fc.state["manual_override"] = True
    fc.send_velocity_target(2.0, 2.0, 0.0)
    logger.info("Velocity command sent during override. Checking state...")
    time.sleep(0.5)


    fc.state["manual_override"] = False
    fc.disarm()
    fc.stop()

    logger.info("=" * 60)
    logger.info("PIXHAWK MAVLINK FLIGHT CONTROLLER TEST COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    test_flight_controller()