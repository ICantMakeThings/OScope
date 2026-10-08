# OScope

OScope is an OpenCV camera viewer with Human, Drone, ect. detection and motion prediction, named as "OpenScope"

It is a (Airsoft!!!) target prediction algorythm that detects where something is going, and tells you where to shoot.

The hardware is: A Radxa Cubie A7S, Display (if a/v, then also a usb-c dp alt to a/v converter), High speed camera eg: OV9281 and a power source like a power bank.

## Models

| Display name | File | purpose |
| --- | --- | --- |
| Animals, Balls, Cars / vehicles, Other objects | `yolo11n.pt` | Ultralytics |
| Lite gun | `gunnknife.pt` | [gun-knife-yolo11n](https://huggingface.co/cosgun99/gun-knife-yolo11n) |
| Drones | `Drone.pt` | [Drone-Detection-YOLOv11x](https://github.com/doguilmak/Drone-Detection-YOLOv11x) |


## Backend selection

`HPP_YOLO_BACKEND=auto` selects Ultralytics/PyTorch and `.pt` on amd64, and the native A733 NPU backend and `.nb` models on the Radxa Cubie A7S. Use `cpu` or `npu` to override the selection where supported.

On the Radxa, run `./build_a733.sh` before starting the application. The script cross-compiles `liba733_npu.so`.

## Run

```bash
source venv2/bin/activate
export DISPLAY=:0
export XAUTHORITY="$HOME/.Xauthority"
python3 ./hpp02.py --camera /dev/video0
```

The camera window opens fullscreen on the device's `:0` display. The settings server listens on all network interfaces at port `8765`; open `http://<radxa-lan-ip>:8765` from a browser on the same LAN. Use `--windowed` to keep the camera window resizable, `--web-port` to change the port, or `--web-host 127.0.0.1` to restrict the server to the device itself. The LAN settings page has no authentication, so run it only on a trusted network.

Profiles can be selected with `HPP_PROFILE=pc`, `HPP_PROFILE=radxa-a7s`, or `HPP_PROFILE=legacy-arm`. `HPP_YOLO_IMGSZ` and `HPP_YOLO_HZ` can tune CPU inference load.
