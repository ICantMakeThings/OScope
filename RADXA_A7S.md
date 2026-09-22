# Cubie A7S / 13M IMX214 deployment

## First boot

Use a current official Radxa OS image. Fit the 31-pin, 0.3 mm **opposite-side**
FPC cable between the IMX214 and A7S, with both connectors' bottom contacts
correctly oriented. In `rsetup`, select **Overlays → Manage overlays → Enable
Radxa Camera 13M 214**, save, and reboot.

Confirm the vendor camera path before starting Python:

```bash
v4l2-ctl --list-devices
DISPLAY=:0 gst-launch-1.0 v4l2src device=/dev/video0 en-awisp=1 en-largemode=0 ! \
  video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! xvimagesink
```

Install the required GStreamer pieces if that command is unavailable:

```bash
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base-apps
```

Then copy this project, create an aarch64 virtual environment, and install its
Python dependencies. Run the app with:

```bash
HPP_PROFILE=radxa-a7s ./hpp.py
```

The default A7S capture pipeline is the same 1080p NV12 ISP path, converted to
BGR only at OpenCV's appsink. It uses a one-frame leaky queue to avoid an old
frame being displayed after an expensive detection. Use `HPP_CAMERA_BACKEND=v4l2`
only if the installed OpenCV was built without GStreamer support. `HPP_PROFILE=pc`
forces desktop behaviour for development and testing.

## Performance choices


On the 4 GB board, the A7S profile defaults YOLO to 416 pixels and 8 Hz
(`HPP_YOLO_IMGSZ=416`, `HPP_YOLO_HZ=8`).

`.pt` is a PyTorch model and cannot directly use the A733 NPU. The
current optional YOLO path is therefore CPU fallback, compile and quantize it with the
Radxa ACUITY **v3** toolchain for A733.

## Useful checks

```bash
python3 -c "import cv2; print('GStreamer:', 'GStreamer' in cv2.getBuildInformation())"
v4l2-ctl -d /dev/video0 --list-formats-ext
```

If `/dev/video*` is missing, revisit the overlay, cable orientation/latch, and
then reboot. If MediaPipe has no compatible aarch64 build in the chosen image,
the app prints a message and uses the slower but dependency-free HOG person
detector rather than failing at import time.
