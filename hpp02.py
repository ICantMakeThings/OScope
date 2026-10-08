#!/usr/bin/env python3

from collections import deque
import argparse
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import types

import cv2
import numpy as np
from web_controls import WebControls

MACHINE = platform.machine().lower()
IS_ARM = MACHINE.startswith(("arm", "aarch64"))
def board_identity():
    """Return safe-to-read id for a-d."""
    parts = [platform.platform(), platform.node()]
    for name in ("/proc/device-tree/model", "/proc/device-tree/compatible"):
        try:
            with open(name, "rb") as source:
                parts.append(source.read().decode("utf-8", "ignore"))
        except OSError:
            pass
    return " ".join(parts).lower()


def select_profile():
    requested = os.environ.get("HPP_PROFILE", "auto").lower()
    valid = {"auto", "pc", "radxa-a7s", "legacy-arm"}
    if requested not in valid:
        raise SystemExit(f"Invalid HPP_PROFILE={requested!r}; use one of {sorted(valid)}")
    if requested != "auto":
        return requested
    identity = board_identity()
    if "cubie a7s" in identity or "a733" in identity:
        return "radxa-a7s"
    return "legacy-arm" if IS_ARM else "pc"


PROFILE = select_profile()
IS_RADXA_A7S = PROFILE == "radxa-a7s"
DETECTOR_BACKEND = "opencv_hog" if PROFILE == "legacy-arm" else "mediapipe_pose"
mp = None
if DETECTOR_BACKEND == "mediapipe_pose":
    sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

    try:
        import mediapipe as mp

        # This program uses the legacy MediaPipe Solutions API.
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
            raise ImportError(
                "installed MediaPipe does not provide the legacy "
                "mp.solutions.pose API"
            )

    except (ImportError, OSError, AttributeError) as error:
        if PROFILE == "pc":
            raise

        print(f"MediaPipe unavailable ({error}); using OpenCV HOG fallback.")
        DETECTOR_BACKEND = "opencv_hog"
        mp = None


WINDOW_NAME = "OScope UI"
CONTROLS_WINDOW_NAME = "OScope Controls"
SIDEBAR_WIDTH = 300
CONTROLS_HEIGHT = 900
AIM_RADIUS = 20
ALIGNMENT_LOSS_GRACE = .15
MODEL_FILE = "yolo11n.pt"
MODEL_FILES = {
    "General YOLO": "yolo11n.pt",
    "Lite gun": "gunnknife.pt",
    "Drone": "Drone.pt",
}
NPU_MODEL_FILES = {
    "General YOLO": "npu_models/yolo11n_6_uint8_a733.nb",
    "Lite gun": "npu_models/gunnknife_6_uint8_a733.nb",
    "Drone": "npu_models/Drone_6_uint8_a733.nb",
}
YOLO_BACKEND_REQUEST = os.environ.get("HPP_YOLO_BACKEND", "auto").lower()
if YOLO_BACKEND_REQUEST not in {"auto", "cpu", "npu"}:
    raise SystemExit("Invalid HPP_YOLO_BACKEND; use auto, cpu, or npu")
if YOLO_BACKEND_REQUEST == "npu" and not IS_RADXA_A7S:
    raise SystemExit("HPP_YOLO_BACKEND=npu is only supported on a detected Radxa Cubie A7S")
YOLO_BACKEND = "npu" if (YOLO_BACKEND_REQUEST == "npu" or
                           YOLO_BACKEND_REQUEST == "auto" and IS_RADXA_A7S) else "cpu"
COCO_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
NPU_CLASS_NAMES = {
    "General YOLO": COCO_CLASS_NAMES,
    "Lite gun": ("gun", "knife"),
    "Drone": ("drone",),
}
POSE_INFERENCE_MAX_WIDTH = 512 if IS_RADXA_A7S else 640
HOG_INFERENCE_MAX_WIDTH = 320

# 1 = full; 0 = lite
POSE_MODEL_COMPLEXITY = 1

YOLO_IMAGE_SIZE = int(os.environ.get("HPP_YOLO_IMGSZ", "416" if IS_RADXA_A7S else "640"))
YOLO_DETECTION_HZ = float(os.environ.get("HPP_YOLO_HZ", "8" if IS_RADXA_A7S else "10"))
ANIMAL_LABELS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}


class MotionPredictor:
    """Constant Velocity prediction."""
    def __init__(self, history_size=12):
        self.history = deque(maxlen=history_size)
        self.label = None

    def reset(self):
        self.history.clear()
        self.label = None

    def add(self, label, position, timestamp):
        if self.label not in (None, label):
            self.reset()
        elif self.history and np.linalg.norm(position - self.history[-1][0]) > 180:
            self.reset()
        self.label = label
        self.history.append((position.astype(float), timestamp))

    def future(self, seconds):
        if len(self.history) < 2:
            return None
        positions = np.array([item[0] for item in self.history])
        elapsed = self.history[-1][1] - self.history[0][1]
        if elapsed <= 0:
            return None
        return positions[-1] + (positions[-1] - positions[0]) / elapsed * seconds


class LedOutput:
    PIN = 27

    def __init__(self):
        self.enabled = self.available = self.state = False
        self.wiringpi = None
        self.message = "LED off"
        try:
            import wiringpi
            wiringpi.wiringPiSetup()
            wiringpi.pinMode(self.PIN, 1)
            wiringpi.digitalWrite(self.PIN, 0)
            self.wiringpi, self.available = wiringpi, True
            self.message = "LED ready"
        except (ImportError, OSError, RuntimeError):
            pass

    def set(self, active):
        active = bool(active and self.enabled)
        if active == self.state:
            return
        self.state = active
        if self.available:
            self.wiringpi.digitalWrite(self.PIN, int(active))
            self.message = f"LED {'HIGH' if active else 'LOW'}"
        else:
            self.message = f"LED ({'ON' if active else 'off'})"

    def close(self):
        self.enabled = False
        self.set(False)
        if self.available:
            self.wiringpi.digitalWrite(self.PIN, 0)


class ToneOutput:
    SAMPLE_RATE = 44100

    def __init__(self):
        self.player = shutil.which("aplay") or shutil.which("paplay")
        self.solid_process = None
        self.solid_stop_event = None

    def player_command(self):
        if not self.player:
            return None
        if os.path.basename(self.player) == "paplay":
            return [self.player, "--raw", "--format=s16le", "--rate=44100", "--channels=1"]
        return [self.player, "-q", "-t", "raw", "-f", "S16_LE", "-r", "44100", "-c", "1"]

    @classmethod
    def samples(cls, frequency, duration, start_sample=0, fade=True):
        count = round(cls.SAMPLE_RATE * duration)
        samples = np.arange(count)
        times = (samples + start_sample) / cls.SAMPLE_RATE
        signal = np.sin(2 * np.pi * frequency * times) * 0.2
        if fade:
            envelope = np.minimum(1, np.minimum(times * 100, (duration - samples / cls.SAMPLE_RATE) * 100))
            signal *= envelope
        return (signal * 32767).astype("<i2").tobytes()

    def play(self, frequency, duration=.12):
        command = self.player_command()
        if command is None:
            return
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
        except OSError:
            return

        def write_tone():
            try:
                process.communicate(input=self.samples(frequency, duration))
            except (BrokenPipeError, OSError):
                pass

        threading.Thread(target=write_tone, daemon=True).start()

    def set_solid(self, active, frequency=740):
        if active and (self.solid_process is None or self.solid_process.poll() is not None):
            self.solid_process = None
            command = self.player_command()
            if command is None:
                return
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE)
            except OSError:
                return
            self.solid_process = process
            stop_event = threading.Event()
            self.solid_stop_event = stop_event

            def stream_tone():
                sample_offset = 0
                chunk_samples = round(self.SAMPLE_RATE * .1)
                try:
                    while not stop_event.is_set() and process.poll() is None:
                        tone = self.samples(frequency, .1, sample_offset, fade=False)
                        process.stdin.write(tone)
                        process.stdin.flush()
                        sample_offset += chunk_samples
                except (BrokenPipeError, OSError, AttributeError, ValueError):
                    pass
                finally:
                    try:
                        process.stdin.close()
                    except (AttributeError, OSError, ValueError):
                        pass
                    process.wait()

            threading.Thread(target=stream_tone, daemon=True).start()
        elif not active and self.solid_process is not None:
            self.solid_process = None
            stop_event = self.solid_stop_event
            self.solid_stop_event = None
            if stop_event is not None:
                stop_event.set()


class Controls:
    def __init__(self, camera_modes, camera_controls):
        self.toggles = {"Humans": True, "Animals": False, "Balls": False,
                        "Cars / vehicles": False, "Other objects": False,
                        "Drones": False, "Lite gun": False,
                        "Monochrome display": True,
                        "Calibration mode": False,
                        "Enable GPIO LED": False}
        self.future_seconds = .5
        self.consecutive_frames = 1
        self.confidence_threshold = .40
        self.crosshair_value = 255
        self.alignment_tone_delay = .2
        self.tab = "Detection"
        self.toggle_rows, self.slider_rect = {}, None
        self.aim_position = None
        self.camera_modes = camera_modes or [("MJPG", 1920, 1080, 30)]
        self.mode_index = next((i for i, mode in enumerate(self.camera_modes)
                                if mode[1:3] == (1920, 1080)), 0)
        self.camera_controls = camera_controls
        self.camera_rows, self.camera_slider_rows = {}, {}
        self.fps_slider_rect = None
        self.mode_option_rows = {}
        self.mode_menu_open = False
        self.camera_changed = False
        self.tab_rows = {}
        self.alert_slider_rows = {}

    def fps_options(self):
        current = self.camera_modes[self.mode_index]
        return sorted({mode[3] for mode in self.camera_modes
                       if mode[:3] == current[:3]})

    def set_fps_ratio(self, ratio):
        options = self.fps_options()
        if not options:
            return
        fps = options[round(np.clip(ratio, 0, 1) * (len(options) - 1))]
        current = self.camera_modes[self.mode_index]
        selected = next((i for i, mode in enumerate(self.camera_modes)
                         if mode[:3] == current[:3] and mode[3] == fps), self.mode_index)
        if selected != self.mode_index:
            self.mode_index = selected
            self.camera_changed = True

    def draw_fps_slider(self, panel, y):
        options = self.fps_options()
        current = self.camera_modes[self.mode_index][3]
        self.fps_slider_rect = (125, y, 122, 25)
        cv2.putText(panel, "Frame rate", (18, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (225, 225, 225), 1)
        left, right = 125, 247
        cv2.line(panel, (left, y + 13), (right, y + 13), (105, 105, 105), 3)
        index = options.index(current)
        ratio = .5 if len(options) == 1 else index / (len(options) - 1)
        knob = int(left + ratio * (right - left))
        cv2.circle(panel, (knob, y + 13), 5, (50, 180, 255), -1)
        cv2.putText(panel, f"{current:g}", (255, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1)
        if len(options) > 1:
            cv2.putText(panel, f"{options[0]:g}", (left, y + 31), cv2.FONT_HERSHEY_SIMPLEX, .37, (170, 170, 170), 1)
            cv2.putText(panel, f"{options[-1]:g}", (right - 15, y + 31), cv2.FONT_HERSHEY_SIMPLEX, .37, (170, 170, 170), 1)

    def draw_toggle(self, panel, name, y, disabled=False):
        self.toggle_rows[name] = (y, y + 28)
        checked = self.toggles[name]
        color = (80, 80, 80) if disabled else ((60, 190, 90) if checked else (55, 55, 55))
        cv2.rectangle(panel, (18, y), (40, y + 22), color, -1)
        cv2.rectangle(panel, (18, y), (40, y + 22), (210, 210, 210), 1)
        if checked:
            cv2.putText(panel, "X", (23, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        cv2.putText(panel, name, (50, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .52,
                    (135, 135, 135) if disabled else (235, 235, 235), 1)

    def draw_camera_button(self, panel, label, value, y):
        self.camera_rows[label] = (y, y + 29)
        cv2.putText(panel, label, (18, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 220, 220), 1)
        cv2.rectangle(panel, (125, y), (SIDEBAR_WIDTH - 18, y + 27), (57, 61, 76), -1)
        cv2.putText(panel, value, (133, y + 19), cv2.FONT_HERSHEY_SIMPLEX, .47, (245, 245, 245), 1)
        cv2.putText(panel, "v", (SIDEBAR_WIDTH - 34, y + 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 230, 230), 1)

    def draw_camera_slider(self, panel, name, y, control):
        minimum, maximum, value = control["min"], control["max"], control["value"]
        self.camera_slider_rows[name] = (y, y + 25)
        cv2.putText(panel, name.replace("_", " ").title(), (18, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (225, 225, 225), 1)
        left, right = 125, 247
        cv2.line(panel, (left, y + 13), (right, y + 13), (105, 105, 105), 3)
        ratio = 0 if maximum == minimum else (value - minimum) / (maximum - minimum)
        knob = int(left + np.clip(ratio, 0, 1) * (right - left))
        cv2.circle(panel, (knob, y + 13), 5, (245, 245, 245), -1)
        cv2.putText(panel, str(int(value)), (255, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1)

    def draw_camera_panel(self, height):
        panel = np.full((height, SIDEBAR_WIDTH, 3), 28, dtype=np.uint8)
        cv2.putText(panel, "CAMERA SETTINGS", (18, 29), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)
        tab_width = (SIDEBAR_WIDTH - 36) // 3
        for index, name in enumerate(("Camera", "Detection", "Alert")):
            left = 18 + index * tab_width
            top = 43
            self.tab_rows[name] = (left, top, left + tab_width - 3, top + 27)
            selected = name == self.tab
            cv2.rectangle(panel, (left, top), (left + tab_width - 3, top + 27),
                          (65, 105, 155) if selected else (48, 51, 61), -1)
            cv2.putText(panel, name, (left + 10, top + 18), cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (255, 255, 255), 1)
        mode = self.camera_modes[self.mode_index]
        self.draw_camera_button(panel, "Video format", "Motion-JPEG" if mode[0] == "MJPG" else mode[0], 83)
        self.draw_camera_button(panel, "Resolution", f"{mode[1]}x{mode[2]} @ {mode[3]:g} fps", 117)
        y = 157
        if self.mode_menu_open:
            # A real dropdown: selection is explicit, not a hidden cycle.
            for index, option in enumerate(self.camera_modes):
                self.mode_option_rows[index] = (y, y + 27)
                selected = index == self.mode_index
                cv2.rectangle(panel, (125, y), (SIDEBAR_WIDTH - 18, y + 26),
                              (78, 105, 155) if selected else (47, 50, 61), -1)
                cv2.putText(panel, f"{option[1]}x{option[2]} @ {option[3]:g}", (133, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                y += 28
        self.draw_fps_slider(panel, y)
        y += 39
        for name in ("brightness", "contrast", "saturation", "hue", "gamma", "white_balance_temperature", "sharpness", "backlight_compensation"):
            if name in self.camera_controls:
                self.draw_camera_slider(panel, name, y, self.camera_controls[name])
                y += 29
        if "white_balance_automatic" in self.camera_controls:
            control = self.camera_controls["white_balance_automatic"]
            self.camera_rows["white_balance_automatic"] = (y, y + 25)
            checked = bool(control["value"])
            cv2.rectangle(panel, (125, y + 2), (141, y + 18), (60, 190, 90) if checked else (55, 55, 55), -1)
            cv2.putText(panel, "X" if checked else "", (128, y + 15), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.putText(panel, "White balance automatic", (148, y + 16), cv2.FONT_HERSHEY_SIMPLEX, .43, (225, 225, 225), 1)
            y += 31
        y += 10
        self.draw_toggle(panel, "Monochrome display", y)
        cv2.putText(panel, "q / Esc: quit", (18, height - 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 180, 180), 1)
        return panel

    def draw(self, height, led):
        self.tab_rows = {}
        self.mode_option_rows = {}
        self.camera_rows = {}
        self.camera_slider_rows = {}
        self.alert_slider_rows = {}
        self.toggle_rows = {}
        self.fps_slider_rect = None
        self.slider_rect = None
        if self.tab == "Camera":
            return self.draw_camera_panel(height)
        panel = np.full((height, SIDEBAR_WIDTH, 3), 28, dtype=np.uint8)
        self.tab_rows = {}
        tab_width = (SIDEBAR_WIDTH - 36) // 3
        for index, name in enumerate(("Camera", "Detection", "Alert")):
            left = 18 + index * tab_width
            self.tab_rows[name] = (left, 43, left + tab_width - 3, 70)
            cv2.rectangle(panel, (left, 43), (left + tab_width - 3, 70),
                          (65, 105, 155) if name == self.tab else (48, 51, 61), -1)
            cv2.putText(panel, name, (left + 10, 61), cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (255, 255, 255), 1)
        y = 94
        if self.tab == "Detection":
            cv2.putText(panel, "DETECTION", (18, y), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)
            y += 26
            for name in ("Humans", "Animals", "Balls", "Cars / vehicles", "Other objects",
                         "Drones", "Lite gun"):
                self.draw_toggle(panel, name, y)
                y += 31
            y += 14
            cv2.putText(panel, f"PREDICTION: {self.future_seconds:.1f} s", (18, y), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1)
            slider_y = y + 22
            self.slider_rect = (18, slider_y, SIDEBAR_WIDTH - 36, 18)
            cv2.rectangle(panel, (18, slider_y), (SIDEBAR_WIDTH - 18, slider_y + 6), (95, 95, 95), -1)
            x = int(18 + (self.future_seconds - .1) / 4.9 * (SIDEBAR_WIDTH - 36))
            cv2.circle(panel, (x, slider_y + 3), 9, (50, 180, 255), -1)
            y = slider_y + 53
            self.draw_toggle(panel, "Monochrome display", y)
            y += 34
        else:
            cv2.putText(panel, "ALERT", (18, y), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2)
            y += 30
            self.draw_alert_slider(panel, "Consecutive detections", y, self.consecutive_frames, 1, 10, "frames")
            y += 48
            self.draw_alert_slider(panel, "Confidence to trigger", y, self.confidence_threshold, .01, 1.00, "%")
            y += 48
            self.draw_alert_slider(panel, "Crosshair level", y, self.crosshair_value, 0, 255, "level")
            y += 48
            self.draw_alert_slider(panel, "Alignment tone delay", y, self.alignment_tone_delay, .1, 2.0, "s")
            y += 60
            cv2.putText(panel, "LED requires alignment, confidence,", (18, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (180, 180, 180), 1)
            cv2.putText(panel, "and the selected consecutive count.", (18, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (180, 180, 180), 1)
            y += 48
        self.draw_toggle(panel, "Enable GPIO LED", y)
        cv2.putText(panel, led.message, (18, y + 58), cv2.FONT_HERSHEY_SIMPLEX, .43,
                    (90, 235, 140) if led.state else (180, 180, 180), 1)
        cv2.putText(panel, "q / Esc: quit", (18, height - 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 180, 180), 1)
        return panel

    def draw_alert_slider(self, panel, name, y, value, minimum, maximum, suffix):
        self.alert_slider_rows[name] = (y, y + 25)
        cv2.putText(panel, name, (18, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (225, 225, 225), 1)
        left, right = 125, 247
        cv2.line(panel, (left, y + 13), (right, y + 13), (105, 105, 105), 3)
        ratio = (value - minimum) / (maximum - minimum)
        knob = int(left + np.clip(ratio, 0, 1) * (right - left))
        cv2.circle(panel, (knob, y + 13), 5, (50, 180, 255), -1)
        if suffix == "frames":
            label = f"{value:g} {suffix}"
        elif suffix == "level":
            label = f"{value:g}"
        else:
            label = f"{value:.1f}" if suffix == "s" else f"{value:.2f}"
        cv2.putText(panel, label, (255, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
        cv2.putText(panel, f"{minimum:g}", (left, y + 31), cv2.FONT_HERSHEY_SIMPLEX, .37, (170, 170, 170), 1)
        cv2.putText(panel, f"{maximum:g}", (right - 15, y + 31), cv2.FONT_HERSHEY_SIMPLEX, .37, (170, 170, 170), 1)

    def click(self, x, y):
        for name, (left, top, right, bottom) in self.tab_rows.items():
            if left <= x <= right and top <= y <= bottom:
                self.tab = name
                self.mode_menu_open = False
                return True
        if self.tab == "Alert":
            for name, (top, bottom) in self.alert_slider_rows.items():
                if top - 8 <= y <= bottom + 8 and 115 <= x <= SIDEBAR_WIDTH - 18:
                    ratio = np.clip((x - 125) / 122, 0, 1)
                    if name == "Consecutive detections":
                        self.consecutive_frames = int(round(1 + ratio * 9))
                    elif name == "Confidence to trigger":
                        self.confidence_threshold = round(.01 + ratio * .99, 2)
                    elif name == "Crosshair level":
                        self.crosshair_value = int(round(ratio * 255))
                    else:
                        self.alignment_tone_delay = round(.1 + ratio * 1.9, 1)
                    return True
        for index, (top, bottom) in self.mode_option_rows.items():
            if top <= y <= bottom and 125 <= x <= SIDEBAR_WIDTH - 18:
                self.mode_index = index
                self.mode_menu_open = False
                self.camera_changed = True
                return True
        if self.fps_slider_rect:
            sx, sy, sw, sh = self.fps_slider_rect
            if sx - 8 <= x <= sx + sw + 8 and sy - 8 <= y <= sy + sh + 8:
                self.set_fps_ratio((x - sx) / sw)
                return True
        for name, (top, bottom) in self.camera_rows.items():
            if top <= y <= bottom and 125 <= x <= SIDEBAR_WIDTH - 18:
                if name == "Video format":
                    formats = list(dict.fromkeys(mode[0] for mode in self.camera_modes))
                    next_format = formats[(formats.index(self.camera_modes[self.mode_index][0]) + 1) % len(formats)]
                    self.mode_index = next(i for i, mode in enumerate(self.camera_modes) if mode[0] == next_format)
                    self.camera_changed = True
                elif name == "Resolution":
                    self.mode_menu_open = not self.mode_menu_open
                elif name == "white_balance_automatic":
                    control = self.camera_controls[name]
                    control["value"] = 0 if control["value"] else 1
                    control["changed"] = True
                return True
        for name, (top, bottom) in self.camera_slider_rows.items():
            if top - 5 <= y <= bottom + 5 and 115 <= x <= SIDEBAR_WIDTH - 18:
                control = self.camera_controls[name]
                control["value"] = round(control["min"] + np.clip((x - 125) / 122, 0, 1) * (control["max"] - control["min"]))
                control["changed"] = True
                return True
        if self.slider_rect:
            sx, sy, sw, sh = self.slider_rect
            if sx - 10 <= x <= sx + sw + 10 and sy - 12 <= y <= sy + sh + 16:
                self.future_seconds = round(.1 + np.clip((x - sx) / sw, 0, 1) * 4.9, 1)
                return True
        for name, (top, bottom) in self.toggle_rows.items():
            if top <= y <= bottom and 18 <= x <= SIDEBAR_WIDTH - 18:
                self.toggles[name] = not self.toggles[name]
                return True
        return False


def get_camera_list(requested_device=None):
    requested_device = requested_device or os.environ.get("HPP_CAMERA_DEVICE")
    if requested_device:
        if not os.path.exists(requested_device):
            print(f"Camera device does not exist: {requested_device}", flush=True)
            return []
        index_match = re.search(r"\d+$", requested_device)
        index = int(index_match.group()) if index_match else 0
        return [{"index": index, "device": requested_device, "res": "auto"}]
    cameras = []
    for device in sorted(glob.glob("/dev/video*"), key=lambda item: int(re.search(r"\d+$", item).group())):
        index = int(re.search(r"\d+$", device).group())
        if IS_RADXA_A7S:
            device_info = v4l2_output(device, "--all").lower()
            if "video capture" in device_info or "sunxi-vin" in device_info:
                cameras.append({"index": index, "device": device,
                                "res": "1920x1080" if "sunxi-vin" in device_info else "auto"})
            continue
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                height, width = frame.shape[:2]
                cameras.append({"index": index, "device": device, "res": f"{width}x{height}"})
        cap.release()
    return cameras


def v4l2_output(device, argument):
    try:
        return subprocess.run(["v4l2-ctl", "-d", device, argument], text=True,
                              capture_output=True, timeout=2, check=False).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def get_camera_modes(device):
    text = v4l2_output(device, "--list-formats-ext")
    modes, pixel_format = [], None
    pending_size = None
    for line in text.splitlines():
        match = re.search(r"Pixel Format: '(.{4})'", line)
        if match:
            pixel_format = match.group(1)
        match = re.search(r"Size: Discrete (\d+)x(\d+)", line)
        if match and pixel_format:
            pending_size = tuple(map(int, match.groups()))
        match = re.search(r"\((\d+(?:\.\d+)?) fps\)", line)
        if match and pixel_format and pending_size:
            modes.append((pixel_format, *pending_size, float(match.group(1))))
            pending_size = None
    if pending_size and pixel_format:
        modes.append((pixel_format, *pending_size, 30))
    if not modes:
        pixel_format = "NV12" if IS_RADXA_A7S and "sunxi-vin" in v4l2_output(device, "--all").lower() else "MJPG"
        modes = [(pixel_format, width, height, 30) for width, height in (
            (1920, 1080), (1280, 720), (640, 480), (640, 360),
            (352, 288), (320, 240), (176, 144), (160, 120),
        )]
    return modes


CV_CONTROL_PROPERTIES = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS, "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION, "hue": cv2.CAP_PROP_HUE,
    "gamma": cv2.CAP_PROP_GAMMA, "white_balance_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "sharpness": cv2.CAP_PROP_SHARPNESS, "backlight_compensation": cv2.CAP_PROP_BACKLIGHT,
    "white_balance_automatic": cv2.CAP_PROP_AUTO_WB,
}


def get_camera_controls(device, cap):
    text = v4l2_output(device, "--list-ctrls")
    controls = {}
    for line in text.splitlines():
        match = re.match(r"\s*([\w_]+).*?:\s*.*?min=(-?\d+) max=(-?\d+).*?value=(-?\d+)", line)
        if match and match.group(1) in CV_CONTROL_PROPERTIES:
            name, minimum, maximum, value = match.groups()
            controls[name] = {"min": int(minimum), "max": int(maximum), "value": int(value), "changed": False}
    for name, prop in CV_CONTROL_PROPERTIES.items():
        if name not in controls:
            value = cap.get(prop)
            if value >= 0:
                controls[name] = {"min": 0, "max": 1 if name == "white_balance_automatic" else 255,
                                  "value": round(value), "changed": False}
    return controls


def radxa_gstreamer_pipeline(device, mode):
    """A7S CSI path: use the vendor ISP, NV12, and a one-frame appsink queue."""
    width, height, fps = mode[1:]
    large_mode = int(width > 1920 or height > 1080)
    return (
        f"v4l2src device={device} en-awisp=1 en-largemode={large_mode} ! "
        f"video/x-raw,format=NV12,width={width},height={height},framerate={round(fps)}/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(device, mode):
    requested_backend = os.environ.get("HPP_CAMERA_BACKEND", "auto").lower()
    if IS_RADXA_A7S and requested_backend in ("auto", "gstreamer"):
        cap = cv2.VideoCapture(radxa_gstreamer_pipeline(device, mode), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        if requested_backend == "gstreamer":
            print("GStreamer CSI pipeline could not open; falling back to V4L2.")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fourcc = cv2.VideoWriter_fourcc(*mode[0])
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode[2])
    cap.set(cv2.CAP_PROP_FPS, mode[3])
    return cap


def select_camera_ui(requested_device=None):
    cameras = get_camera_list(requested_device)
    if not cameras:
        print("No cameras found!")
        return None
    print("\nAvailable cameras:")
    for camera in cameras:
        print(f"  [{camera['index']}] {camera['res']}")
    if len(cameras) == 1:
        return cameras[0]
    available = {camera["index"] for camera in cameras}
    while True:
        choice = input(f"Select camera {sorted(available)} (Enter = first): ").strip()
        if not choice:
            return cameras[0]
        try:
            if int(choice) in available:
                return next(camera for camera in cameras if camera["index"] == int(choice))
        except ValueError:
            pass
        print("Choose one of the listed camera numbers.")


def find_red_laser(frame):
    """Return the strongest compact red or bright monochrome spot."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 130, 170), (10, 255, 255))
    mask |= cv2.inRange(hsv, (170, 130, 170), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = max(200, int(np.percentile(gray, 99.9)))
        mask = cv2.inRange(gray, threshold, 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if not 2 <= area <= 1000:
        return None
    moments = cv2.moments(contour)
    if not moments["m00"]:
        return None
    return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])


def pose_detection(pose, frame):
    height, width = frame.shape[:2]
    inference_frame = frame
    if width > POSE_INFERENCE_MAX_WIDTH:
        scale = POSE_INFERENCE_MAX_WIDTH / width
        inference_frame = cv2.resize(frame, (POSE_INFERENCE_MAX_WIDTH, round(height * scale)),
                                     interpolation=cv2.INTER_AREA)
    result = pose.process(cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return None
    visible = [point for point in result.pose_landmarks.landmark if point.visibility > .5]
    if not visible:
        return None
    xs, ys = [point.x * width for point in visible], [point.y * height for point in visible]
    box = (max(0, min(xs)), max(0, min(ys)), min(width - 1, max(xs)), min(height - 1, max(ys)))
    return {"label": "person", "box": box, "center": np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])}


def hog_person_detection(hog, frame):
    """ARM-safe fallback person detector using only the OpenCV installation.

    It is intentionally limited to a 320px-wide image: an ODROID-C2 has only
    Cortex-A53 CPU cores, so full-resolution HOG would make capture unusable.
    """
    height, width = frame.shape[:2]
    scale = min(1.0, HOG_INFERENCE_MAX_WIDTH / width)
    image = frame if scale == 1 else cv2.resize(frame, (round(width * scale), round(height * scale)),
                                                 interpolation=cv2.INTER_AREA)
    boxes, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.10)
    if len(boxes) == 0:
        return None
    index = int(np.argmax(weights))
    x, y, box_width, box_height = boxes[index]
    inverse_scale = 1 / scale
    box = (x * inverse_scale, y * inverse_scale,
           (x + box_width) * inverse_scale, (y + box_height) * inverse_scale)
    return {"label": "person", "box": box,
            "center": np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])}


def yolo_detections(model, frame, toggles, model_name, confidence_threshold):
    active = {name for name in ("Humans", "Animals", "Balls", "Cars / vehicles", "Other objects", "Drones") if toggles[name]}
    if model_name != "General YOLO":
        active.add("Other objects")
    if not active:
        return []
    if YOLO_BACKEND == "npu":
        detections = []
        for x1, y1, x2, y2, class_id, confidence in model.detect(frame, confidence_threshold):
            label = NPU_CLASS_NAMES[model_name][class_id]
            custom_model = model_name != "General YOLO"
            wanted = (custom_model and ("Other objects" in active or "Drones" in active)) or (
                not custom_model and (("Humans" in active and label == "person") or
                ("Animals" in active and label in ANIMAL_LABELS) or
                ("Balls" in active and label == "sports ball") or
                ("Cars / vehicles" in active and label in VEHICLE_LABELS) or
                ("Other objects" in active and label != "person" and
                 label not in ANIMAL_LABELS and label != "sports ball" and
                 label not in VEHICLE_LABELS)))
            if wanted:
                detections.append({"label": label, "box": (x1, y1, x2, y2),
                                   "center": np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                                   "confidence": confidence})
        return detections
    output = model(frame, verbose=False, imgsz=YOLO_IMAGE_SIZE)[0]
    detections = []
    for item in output.boxes:
        confidence = float(item.conf[0])
        if confidence < confidence_threshold:
            continue
        label = output.names[int(item.cls[0])]
        custom_model = model_name != "General YOLO"
        wanted = (custom_model and ("Other objects" in active or "Drones" in active)) or (
              not custom_model and (("Animals" in active and label in ANIMAL_LABELS) or
              ("Balls" in active and label == "sports ball") or
              ("Cars / vehicles" in active and label in VEHICLE_LABELS) or
              ("Other objects" in active and label != "person" and
               label not in ANIMAL_LABELS and label != "sports ball" and
               label not in VEHICLE_LABELS)))
        if wanted:
            x1, y1, x2, y2 = item.xyxy[0].cpu().numpy()
            detections.append({"label": label, "box": (x1, y1, x2, y2),
                               "center": np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
                               "confidence": confidence})
    return detections


def main():
    parser = argparse.ArgumentParser(description="OScope camera viewer")
    parser.add_argument(
        "--camera", "-c", metavar="DEVICE",
        help="A733 camera device, for example /dev/video0",
    )
    parser.add_argument("--web-host", default=os.environ.get("HPP_WEB_HOST", "0.0.0.0"),
                        help="Settings server bind address (default: all interfaces)")
    parser.add_argument("--web-port", type=int, default=int(os.environ.get("HPP_WEB_PORT", "8765")),
                        help="Settings server port (default: 8765)")
    parser.add_argument("--windowed", action="store_true",
                        help="Keep the camera window resizable instead of fullscreen")
    args = parser.parse_args()
    print(f"Hardware profile: {PROFILE} ({MACHINE}); camera backend: "
          f"{os.environ.get('HPP_CAMERA_BACKEND', 'auto')}; YOLO backend: {YOLO_BACKEND}")
    if YOLO_BACKEND == "npu" and not os.path.isfile("liba733_npu.so"):
        raise SystemExit("Radxa A7S detected but liba733_npu.so is missing; run ./build_a733.sh first")
    camera = select_camera_ui(args.camera if IS_RADXA_A7S else None)
    if camera is None:
        return
    modes = get_camera_modes(camera["device"])
    print("Supported modes:")
    for pixel_format, mode_width, mode_height, fps in modes:
        display_format = "Motion-JPEG" if pixel_format == "MJPG" else pixel_format
        print(f"  {display_format}: {mode_width}x{mode_height} @ {fps:g} fps")
    preferred_size = (640, 480) if DETECTOR_BACKEND == "opencv_hog" else (1920, 1080)
    if IS_RADXA_A7S:
        preferred_mode = next((mode for mode in modes if mode[1:3] == (1920, 1080)), modes[0])
    else:
        preferred_mode = next((mode for mode in modes if mode[:3] == ("MJPG", *preferred_size)), modes[0])
    cap = open_camera(camera["device"], preferred_mode)
    if not cap.isOpened():
        print(f"Failed to open camera {camera['index']}")
        return
    actual_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_format = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(f"Using camera {camera['index']}: {actual_format} {actual_width}x{actual_height}")

    def create_pose(model_complexity):
        return mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=.5,
            min_tracking_confidence=.5,
        )

    pose = create_pose(POSE_MODEL_COMPLEXITY) if DETECTOR_BACKEND == "mediapipe_pose" else None
    hog = None
    if DETECTOR_BACKEND == "opencv_hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    print(f"Human detector: {DETECTOR_BACKEND}")
    controls = Controls(modes, get_camera_controls(camera["device"], cap))
    controls.mode_index = next((i for i, mode in enumerate(modes)
                                if mode[0] == actual_format and mode[1:3] == (actual_width, actual_height)),
                               controls.mode_index)
    predictor, led, tones = MotionPredictor(), LedOutput(), ToneOutput()
    detection_streak = 0
    streak_label = None
    previous_detection_count = 0
    aligned_since = None
    alignment_lost_since = None
    object_models, frame_times = {}, deque(maxlen=30)
    cached_detections, last_detection_at, detection_signature = [], 0.0, None
    web_controls = WebControls(controls, args.web_host, args.web_port)
    web_controls.publish_state(led.message, led.state)
    web_controls.start()
    print(f"LAN settings: {web_controls.lan_url()}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if not args.windowed:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    window_sized = False
    print("Running. Browser settings are available on the LAN; press q or Escape to quit.")
    last_state_publish = 0.0
    try:
        while True:
            web_controls.apply_pending()
            started = time.time()
            if controls.camera_changed:
                cap.release()
                cap = open_camera(camera["device"], controls.camera_modes[controls.mode_index])
                controls.camera_changed = False
                window_sized = False
                predictor.reset()
                if not cap.isOpened():
                    print("Could not apply that camera mode; restoring the previous stream.")
                    cap = open_camera(camera["device"], preferred_mode)
            for name, control in controls.camera_controls.items():
                if control["changed"]:
                    cap.set(CV_CONTROL_PROPERTIES[name], control["value"])
                    control["changed"] = False
            ok, frame = cap.read()
            capture_ms = (time.time() - started) * 1000
            if not ok:
                selected_mode = controls.camera_modes[controls.mode_index]
                print(f"Camera stream timed out; reopening {selected_mode[1]}x{selected_mode[2]}...")
                cap.release()
                time.sleep(.5)
                cap = open_camera(camera["device"], selected_mode)
                window_sized = False
                cv2.waitKey(1)
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            height, width = frame.shape[:2]
            if not window_sized:
                cv2.resizeWindow(WINDOW_NAME, width, height)
                window_sized = True
            if controls.aim_position is None:
                controls.aim_position = np.array([width / 2, height / 2])
            aim = np.clip(controls.aim_position, [0, 0], [width - 1, height - 1])
            if controls.toggles["Calibration mode"]:
                laser = find_red_laser(frame)
                if laser is not None:
                    controls.aim_position = laser
                    aim = laser
                    controls.toggles["Calibration mode"] = False
                    cv2.circle(frame, tuple(laser.astype(int)), 7, (0, 0, 255), 1)

            general_requested = any(controls.toggles[name] for name in
                                    ("Humans", "Animals", "Balls", "Cars / vehicles", "Other objects"))
            selected_models = []
            if general_requested:
                selected_models.append("General YOLO")
            if controls.toggles["Drones"]:
                selected_models.append("Drone")
            if controls.toggles["Lite gun"]:
                selected_models.append("Lite gun")
            yolo_requested = bool(selected_models)
            yolo_needed = yolo_requested and (IS_RADXA_A7S or DETECTOR_BACKEND != "opencv_hog")
            signature = (controls.confidence_threshold,
                         tuple(controls.toggles[name] for name in ("Humans", "Animals", "Balls", "Cars / vehicles", "Other objects", "Drones", "Lite gun")))
            detection_hz = 5 if DETECTOR_BACKEND == "opencv_hog" else (YOLO_DETECTION_HZ if yolo_needed else 30)
            interval = 1 / detection_hz
            ran_detection = signature != detection_signature or started - last_detection_at >= interval
            if ran_detection:
                cached_detections = []
                if controls.toggles["Humans"] and (not yolo_needed or "General YOLO" not in selected_models):
                    person = (hog_person_detection(hog, frame) if DETECTOR_BACKEND == "opencv_hog"
                              else pose_detection(pose, frame))
                    if person:
                        cached_detections.append(person)
                if yolo_needed:
                    for model_name in selected_models:
                        if model_name not in object_models:
                            if YOLO_BACKEND == "npu":
                                from a733_npu import A733NPUDetector
                                object_models[model_name] = A733NPUDetector(NPU_MODEL_FILES[model_name], model_name)
                            else:
                                from ultralytics import YOLO
                                object_models[model_name] = YOLO(MODEL_FILES[model_name])
                        cached_detections.extend(yolo_detections(object_models[model_name], frame, controls.toggles,
                                                                model_name, controls.confidence_threshold))
                last_detection_at, detection_signature = started, signature
            detections = cached_detections
            if ran_detection:
                detection_count = len(detections)
                if detection_count > previous_detection_count:
                    tones.play(880)
                elif detection_count < previous_detection_count:
                    tones.play(440)
                previous_detection_count = detection_count

            target = min(detections, key=lambda item: np.linalg.norm(item["center"] - aim), default=None)
            qualifying_target = (target is not None and
                                 target.get("confidence", 1.0) >= controls.confidence_threshold)
            if qualifying_target and streak_label == target["label"]:
                if ran_detection:
                    detection_streak += 1
            elif qualifying_target and ran_detection:
                streak_label = target["label"]
                detection_streak = 1
            elif ran_detection:
                streak_label = None
                detection_streak = 0
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection["box"])
                color = (0, 255, 255) if detection is target else (0, 190, 80)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, detection["label"], (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)

            future = None
            if target is not None:
                if ran_detection:
                    predictor.add(target["label"], target["center"], started)
                cv2.circle(frame, tuple(target["center"].astype(int)), 6, (0, 255, 0), -1)
                future = predictor.future(controls.future_seconds)
                if future is not None:
                    future = np.clip(future, [0, 0], [width - 1, height - 1])
                    future_point = tuple(future.astype(int))
                    cv2.line(frame, tuple(target["center"].astype(int)), future_point, (255, 255, 0), 1)
                    cv2.circle(frame, future_point, 7, (0, 0, 255), -1)
                    cv2.putText(frame, f"+{controls.future_seconds:.1f}s", (future_point[0] + 10, future_point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 255), 2)
            elif ran_detection:
                predictor.reset()

            aligned = future is not None and np.linalg.norm(future - aim) <= AIM_RADIUS
            if aligned:
                alignment_lost_since = None
                if aligned_since is None:
                    aligned_since = started
                tones.set_solid(started - aligned_since >= controls.alignment_tone_delay)
            elif aligned_since is not None:
                if alignment_lost_since is None:
                    alignment_lost_since = started
                elif started - alignment_lost_since >= ALIGNMENT_LOSS_GRACE:
                    aligned_since = None
                    alignment_lost_since = None
                    tones.set_solid(False)
            led.enabled = controls.toggles["Enable GPIO LED"]
            led.set(aligned and detection_streak >= controls.consecutive_frames)
            aim_color = (controls.crosshair_value,) * 3
            aim_point = tuple(aim.astype(int))
            overlay_scale = max(width / 640, height / 480)
            cross_arm = max(8, round(8 * overlay_scale))
            cross_thickness = max(2, round(overlay_scale))
            cv2.circle(frame, aim_point, AIM_RADIUS, aim_color, max(2, round(overlay_scale)), cv2.LINE_AA)
            cv2.line(frame, (aim_point[0] - cross_arm, aim_point[1]),
                     (aim_point[0] + cross_arm, aim_point[1]), aim_color, cross_thickness, cv2.LINE_AA)
            cv2.line(frame, (aim_point[0], aim_point[1] - cross_arm),
                     (aim_point[0], aim_point[1] + cross_arm), aim_color, cross_thickness, cv2.LINE_AA)

            if started - last_state_publish >= .25:
                web_controls.publish_state(led.message, led.state)
                last_state_publish = started

            elapsed_ms = (time.time() - started) * 1000
            frame_times.append(elapsed_ms)
            work_ms = max(0, elapsed_ms - capture_ms)
            cv2.putText(frame, f"FPS: {1000 / np.mean(frame_times):.1f} | camera: {capture_ms:.0f}ms | work: {work_ms:.0f}ms",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2)
            if target is not None:
                cv2.putText(frame, f"confidence: {target.get('confidence', 1.0):.2f} | detections: {detection_streak}/{controls.consecutive_frames}",
                            (10, 51), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)

            display_frame = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                             if controls.toggles["Monochrome display"] else frame)
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        web_controls.close()
        tones.set_solid(False)
        led.close()
        if pose is not None:
            pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
