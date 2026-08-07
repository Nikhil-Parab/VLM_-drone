# Drone Tracking Engine — Pi Deployment Package

Runtime files only (dev/test scripts are in `dev_tools/`, not needed on
the Pi). Full development history: `CHANGELOG.md`.

## Setup on the Pi

```bash
pip install -r requirements.txt --break-system-packages
```

If `torch` fails to install, use piwheels (prebuilt ARM wheels):
```bash
pip install torch --index-url https://www.piwheels.org/simple
```

For pyaudio (microphone support):
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

First run downloads `yolov8n.pt` (~6MB) and `SmolVLM-256M-Instruct`
(~987MB) — needs a 32GB+ SD card and internet access once.

## Run

**Legacy YOLO path (default):**
```bash
python drone_tracking_engine.py --skip-activation
python voice_command_interface.py --text "follow the phone"
```

**SVLM master plan path (open-vocabulary + SVLM control):**
```bash
python drone_tracking_engine.py --detector grounding --svlm-control --skip-activation --device cpu
python voice_command_interface.py --text "follow the red mug handle"
python voice_command_interface.py --text "move right then up slowly to 5 meters"
```

**Offline evaluation (Phase 4):**
```bash
python eval_harness.py --video clip.mp4 --phrase "person" --svlm --device cpu
```

**Production Pi budget (Phase 8 — revisit after sim validation):**
```bash
python drone_tracking_engine.py --max-cores 2 --device cpu
```

## Keyboard controls (engine window)
`q` quit · `l` lock nearest target · `u` unlock/hover · `h` hover ·
`s` search · `r` return-to-launch · `p` toggle timing HUD

## If lag reappears
Set `DEBUG = True` at the top of `reid_engine.py` and/or
`small_object_classifier.py` to re-enable per-call VLM timing/frequency
logs, then follow the same measure-first diagnostic approach documented
in `CHANGELOG.md` (Hypotheses 1 & 2) before changing anything.

## Dev tools (not needed on the Pi)
`dev_tools/_opt_validate.py` and `dev_tools/_smoke_test.py` — logic checks
that run without a camera/model. `dev_tools/_bench_diagnostic.py` —
benchmarking script. Run these on a dev machine when modifying the code,
not as part of normal Pi operation.
