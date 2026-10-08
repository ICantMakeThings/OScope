import json
import math
import queue
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SETTING_RANGES = {
    "future_seconds": (0.1, 5.0, 1),
    "consecutive_frames": (1, 10, 0),
    "confidence_threshold": (0.01, 1.0, 2),
    "crosshair_value": (0, 255, 0),
    "alignment_tone_delay": (0.1, 2.0, 1),
}


class WebControls:
    def __init__(self, controls, host="0.0.0.0", port=8765):
        self.controls = controls
        self.commands = queue.Queue()
        self._state_lock = threading.Lock()
        self._state = b"{}"
        web_controls = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/state":
                    with web_controls._state_lock:
                        body = web_controls._state
                    self.send_bytes(200, "application/json; charset=utf-8", body)
                    return
                if self.path in ("/", "/index.html"):
                    try:
                        body = (Path(__file__).resolve().parent / "web" / "index.html").read_bytes()
                    except OSError:
                        self.send_bytes(500, "text/plain; charset=utf-8", b"Settings page is unavailable")
                        return
                    self.send_bytes(200, "text/html; charset=utf-8", body)
                    return
                self.send_bytes(404, "text/plain; charset=utf-8", b"Not found")

            def do_POST(self):
                if self.path != "/api/control":
                    self.send_bytes(404, "application/json", b'{"error":"Not found"}')
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size < 1 or size > 8192:
                        raise ValueError("Invalid request size")
                    command = json.loads(self.rfile.read(size))
                    web_controls.validate_command(command)
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    body = json.dumps({"error": str(error)}).encode("utf-8")
                    self.send_bytes(400, "application/json; charset=utf-8", body)
                    return
                web_controls.commands.put(command)
                self.send_bytes(202, "application/json; charset=utf-8", b'{"queued":true}')

            def send_bytes(self, status, content_type, body):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format_string, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = Server((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self):
        return self.server.server_address[1]

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def lan_url(self):
        address = "127.0.0.1"
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 80))
            address = probe.getsockname()[0]
        except OSError:
            pass
        finally:
            probe.close()
        return f"http://{address}:{self.port}/"

    def validate_command(self, command):
        if not isinstance(command, dict):
            raise ValueError("Expected a settings command")
        action = command.get("action")
        if action == "toggle":
            if command.get("name") not in self.controls.toggles:
                raise ValueError("Unknown toggle")
            if not isinstance(command.get("value"), bool):
                raise ValueError("Toggle value must be true or false")
            return
        if action == "setting":
            key = command.get("key")
            if key not in SETTING_RANGES:
                raise ValueError("Unknown setting")
            self.require_number(command.get("value"))
            return
        if action == "camera_mode":
            index = command.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("Camera mode must be an integer")
            if not 0 <= index < len(self.controls.camera_modes):
                raise ValueError("Camera mode is out of range")
            return
        if action == "camera_control":
            name = command.get("name")
            if name not in self.controls.camera_controls:
                raise ValueError("Unknown camera control")
            self.require_number(command.get("value"))
            return
        raise ValueError("Unknown action")

    @staticmethod
    def require_number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Setting value must be a finite number")

    def apply_pending(self):
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            action = command["action"]
            if action == "toggle":
                name = command["name"]
                if name in self.controls.toggles:
                    self.controls.toggles[name] = command["value"]
            elif action == "setting":
                key = command["key"]
                if key in SETTING_RANGES:
                    minimum, maximum, digits = SETTING_RANGES[key]
                    value = min(max(command["value"], minimum), maximum)
                    setattr(self.controls, key, round(value, digits))
            elif action == "camera_mode":
                index = command["index"]
                if 0 <= index < len(self.controls.camera_modes):
                    self.controls.mode_index = index
                    self.controls.camera_changed = True
            elif action == "camera_control":
                name = command["name"]
                control = self.controls.camera_controls.get(name)
                if control is not None:
                    value = min(max(command["value"], control["min"]), control["max"])
                    control["value"] = round(value)
                    control["changed"] = True

    def publish_state(self, led_message="", led_state=False):
        controls = self.controls
        mode = controls.camera_modes[controls.mode_index]
        state = {
            "toggles": dict(controls.toggles),
            "settings": {key: getattr(controls, key) for key in SETTING_RANGES},
            "camera_modes": [
                {"format": item[0], "width": item[1], "height": item[2], "fps": item[3]}
                for item in controls.camera_modes
            ],
            "camera_mode_index": controls.mode_index,
            "camera_mode": {"format": mode[0], "width": mode[1], "height": mode[2], "fps": mode[3]},
            "camera_controls": {
                name: {"min": item["min"], "max": item["max"], "value": item["value"]}
                for name, item in controls.camera_controls.items()
            },
            "led": {"message": led_message, "enabled": led_state},
        }
        body = json.dumps(state, separators=(",", ":")).encode("utf-8")
        with self._state_lock:
            self._state = body