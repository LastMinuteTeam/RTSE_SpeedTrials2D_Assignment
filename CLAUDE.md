# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coursework for SECJ 4423 Real-Time Software Engineering (RTSE) Assignment 2. The
deliverable is an **autonomous driver** for the *SpeedTrials2D* Unity game: a Python
program (`sample_drive.py`) that connects to the game over TCP sockets, receives front
and back camera frames, runs OpenCV vision to decide steering/throttle, and streams
control packets back. The accompanying IEEE technical paper frames the same code as a
uC/OS-II-style real-time system.

`SpeedTrials2D/`, `SpeedTrials2D_V2.5/`, and `SpeedTrials2D_V3.0/` are **prebuilt Unity
game binaries** (the simulator), not source you edit. `V3.0` is the latest; develop
against it unless told otherwise. `test_communication.py` is the instructor-provided
manual-driving reference (WASD keyboard control) — read it to understand the wire
protocol; do not build on it.

## Run / develop

There is no build step and no test suite. Workflow is: launch the game, then run the
driver against it.

```sh
# 1. Launch the simulator (it opens camera servers on 8080/8082 and waits for a control client)
./SpeedTrials2D_V3.0/SpeedTrials2D.exe

# 2. Run the autonomous driver (uses the project venv)
./.venv/Scripts/python.exe sample_drive.py

# Syntax-check without running the game:
python -m py_compile sample_drive.py
```

Dependencies (`requirements.txt`): `opencv-python`, `numpy`, `keyboard`, `pytesseract`.
`pytesseract` is imported defensively (`try/except`) and is only used for golden-lane OCR;
the code falls back gracefully if Tesseract isn't installed. Press **`q`** in any window to
quit cleanly.

## Wire protocol (do not change — fixed by the game)

- **Front camera**: TCP `127.0.0.1:8080`, **Back camera**: `8082`. The driver is the
  *client* and connects to these. Each frame = 4-byte little-endian length prefix, then a
  JPEG-encoded image (decode with `cv2.imdecode`).
- **Control**: the driver is the *server* on `127.0.0.1:8081`; the game connects to it.
  Each control packet is `struct.pack('ff', steering_input, acceleration_input)`.
  - `steering_input`: -1.0 = left, +1.0 = right, 0.0 = straight.
  - `acceleration_input`: +1.0 = forward, -1.0 = reverse.

The setup functions and the camera read loop in `sample_drive.py` are marked
"Do not change this in your code" — treat the networking layer as fixed and put logic in
the vision/decision functions.

## Architecture of `sample_drive.py`

A single ~2500-line module. The grading rubric rewards demonstrating real-time concepts,
so the structure deliberately mirrors an RTOS (the IEEE paper maps this to uC/OS-II).

### Real-time task model
`RTTask(threading.Thread)` runs an `execute_func` on a fixed `period`, sleeping the
remainder of each cycle, and sets a Windows thread priority via `ctypes`. Four tasks are
started in `__main__`, all at period `0.005s` (200 Hz):

| Task | Priority | Role |
|------|----------|------|
| `ReadFrontCamera` (`read_front_camera_task`) | HIGH | drain front socket → `latest_front_frame` |
| `ReadBackCamera` (`read_back_camera_task`)   | LOW  | drain back socket → `latest_back_frame` |
| `Processing` (`processing_task`)             | HIGH | run `analyse_drive`, write steering/accel/mode |
| `SendControls` (`send_controls_task`)        | HIGH | turn desired steering into tapped pulses, send packet |

This is a **Perceive → Compute → Actuate** pipeline. All cross-task state lives in the
global `shared_data` dict, guarded by the single `data_lock` mutex — any read or write of
`shared_data` must hold `data_lock`. Camera read tasks also do the `cv2.imshow` display
(overlaying `debug_frame` / `back_debug_frame` when present).

`send_controls_task` does not send `steering_input` continuously: it converts a desired
direction into short **taps** (cooldowns + a confirm-cycle + release-pending state machine)
to emulate keyboard presses, because the game expects discrete steering taps.

### Decision core: `analyse_drive(front_frame, back_frame)`
The brain. Returns `(steering, acceleration, drive_mode, debug_frame, back_debug_frame,
is_edge_escape)`. It detects each game challenge and chooses a target via a strict
**precedence ladder** (highest first):

1. **edge escape** — emergency, car drifting off-road (`check_edge_lane_escape`)
2. **golden lane** — an OCR'd lane number you must occupy (`detect_golden_lane_number`)
3. **police dodge** — avoid colliding with the police car (collision = game over)
4. **chasing car** — a car appears behind you; dodge it (back camera)
5. **red collect** — *only while a police event is active*, drive to a RED token
6. **golden lane** (secondary scoring path) / **hazard avoid** / **green collect** / hazard fallback

Persistent per-call state is stashed as function attributes (e.g.
`analyse_drive.left_lane_poly`, `analyse_drive.police_event_until`,
`analyse_drive.golden_lane_target`) instead of globals — same pattern in `detect_low_light`
and `send_controls_task`. **Low light** (`detect_low_light`) short-circuits most detectors
and reverses (`acceleration = -1.0`) to recover.

### Tuning constants
Lines ~45–200 are a large block of `UPPERCASE` tuning constants grouped by subsystem
(`CHASE_BACK_*`, `POLICE_*`, `LOW_LIGHT_*`, `YELLOW_*`, `RED_*`, `GREEN_*`, etc.). Most
behavior tuning happens here rather than in logic. When adjusting a challenge's sensitivity,
look for its prefixed constants first.

## Domain rules that aren't in the code

The game's challenge mechanics (low-light recovery, the two chasing-car waves with 10s/3s
windows, the police-car red-token rule and its red/blue-light signature) define what
"correct" behaviour is but are not derivable from the source. The **red token has a dual
role**: avoid it normally, but *collect* it during a police event — this is wired through
`choose_hazard_avoidance(..., avoid_red=not police_active)`. See the persisted memory notes
(`game-challenges`, `technical-paper`) before changing token or challenge logic.

## Conventions

- Per-version game folders are never edited; only the Python at the repo root is source.
- New detectors follow the existing shape: return `None` or a dict with keys like
  `target_x`, `norm_x`, `strength`, `mode`, `front_priority_score`; feed into the
  precedence ladder in `analyse_drive`.
- Use function attributes for detector state that must persist across frames; never block
  on `data_lock` longer than the dict access.
