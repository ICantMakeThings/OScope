"""ctypes wrapper for the native A733 YOLO11 detector."""

import ctypes
from pathlib import Path


_CLASS_COUNTS = {"General YOLO": 80, "Lite gun": 2, "Drone": 1}


class _Detection(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("class_id", ctypes.c_int),
        ("confidence", ctypes.c_float),
    ]


class A733NPUDetector:
    def __init__(self, model_path, model_name):
        if model_name not in _CLASS_COUNTS:
            raise ValueError(f"Unsupported A733 model: {model_name}")
        library_path = Path(__file__).with_name("liba733_npu.so")
        if not library_path.is_file():
            raise RuntimeError(
                "A733 NPU backend is not built. Run ./build_a733.sh on the Radxa A7S first."
            )
        self._library = ctypes.CDLL(str(library_path))
        self._library.a733_npu_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self._library.a733_npu_create.restype = ctypes.c_int
        self._library.a733_npu_detect.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.POINTER(_Detection), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ]
        self._library.a733_npu_detect.restype = ctypes.c_int
        self._library.a733_npu_destroy.argtypes = [ctypes.c_void_p]
        self._library.a733_npu_last_error.restype = ctypes.c_char_p
        self._handle = ctypes.c_void_p()
        result = self._library.a733_npu_create(
            str(Path(model_path).resolve()).encode(), _CLASS_COUNTS[model_name], ctypes.byref(self._handle)
        )
        if result != 0:
            self.close()
            message = self._library.a733_npu_last_error()
            detail = message.decode(errors="replace") if message else "unknown native error"
            raise RuntimeError(f"Could not load A733 model {model_path}: {detail}")

    def detect(self, frame, confidence_threshold):
        if self._handle is None:
            raise RuntimeError("A733 detector is closed")
        height, width = frame.shape[:2]
        output = (_Detection * 256)()
        count = ctypes.c_int()
        result = self._library.a733_npu_detect(
            self._handle,
            frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            width,
            height,
            confidence_threshold,
            output,
            len(output),
            ctypes.byref(count),
        )
        if result != 0:
            message = self._library.a733_npu_last_error()
            detail = message.decode(errors="replace") if message else "unknown native error"
            raise RuntimeError(f"A733 NPU inference failed: {detail}")
        return [
            (item.x1, item.y1, item.x2, item.y2, item.class_id, item.confidence)
            for item in output[:count.value]
        ]

    def close(self):
        if getattr(self, "_handle", None):
            self._library.a733_npu_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
