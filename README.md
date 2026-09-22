# OScope

OScope is an OpenCV camera viewer with Human, Drone, ect. detection and motion prediction, named as "OpenScope" or "oh-scope!"

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
source activate.sh
./hpp.py
```

Profiles can be selected with `HPP_PROFILE=pc`, `HPP_PROFILE=radxa-a7s`, or `HPP_PROFILE=legacy-arm`. `HPP_YOLO_IMGSZ` and `HPP_YOLO_HZ` can tune CPU inference load.
