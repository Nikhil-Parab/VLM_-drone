import os
import sys
import time
import subprocess
import logging

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

import psutil

logger = logging.getLogger(__name__)

def get_cpu_temperature():


    try:
        thermal_file = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_file):
            with open(thermal_file, "r") as f:
                temp_raw = float(f.read().strip())
                return round(temp_raw / 1000.0, 1)
    except Exception:
        pass


    try:
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=1.0)
        if res.returncode == 0 and "temp=" in res.stdout:

            temp_str = res.stdout.strip().split("=")[1].split("'")[0]
            return round(float(temp_str), 1)
    except Exception:
        pass

    return 42.0

def get_system_health():

    cpu_temp = get_cpu_temperature()
    cpu_usage = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    try:
        load_avg = [round(x, 2) for x in os.getloadavg()]
    except Exception:
        load_avg = [0.0, 0.0, 0.0]


    if cpu_temp > 80.0 or cpu_usage > 92.0 or mem.percent > 90.0:
        health_status = "CRITICAL / OVERLOAD"
        status_color = "var(--accent-red)"
    elif cpu_temp > 70.0 or cpu_usage > 80.0 or mem.percent > 80.0:
        health_status = "WARM / ELEVATED"
        status_color = "var(--accent-orange)"
    else:
        health_status = "HEALTHY (OPTIMAL)"
        status_color = "var(--accent-green)"

    return {
        "cpu_temp": cpu_temp,
        "cpu_usage": round(cpu_usage, 1),
        "ram_used_mb": round(mem.used / (1024 * 1024), 1),
        "ram_total_gb": round(mem.total / (1024 * 1024 * 1024), 2),
        "ram_pct": round(mem.percent, 1),
        "disk_pct": round(disk.percent, 1),
        "load_1m": load_avg[0],
        "status": health_status,
        "status_color": status_color
    }