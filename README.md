# Remote Inference SDK

## Overview
The **Remote Inference SDK** provides an easy-to-use interface for interacting with a WebSocket-based remote inference server. It allows users to send image data, run vision-based inference modules, and retrieve processed results asynchronously or synchronously.

## Features
- **Asynchronous and Synchronous Support**: Use `RemoteInferenceClient` for async operations or `SyncRemoteInferenceClient` for synchronous execution.
- **Multiple Vision Modules**: Supports object detection, human pose estimation, segmentation tracking, depth estimation, and more.
- **WebSocket Communication**: Efficient real-time data transmission.
- **Auto-Reconnection**: Handles connection loss and automatically reconnects.
- **Queue-based Response Handling**: Ensures proper retrieval of inference results.

## Installation
Ensure you have Python 3.7+ installed. Then, install the required dependencies:

```bash
pip install websockets opencv-python numpy
```

## Usage

### 1) Initialization
```python
from sdk import SyncRemoteInferenceClient

def main():
    client = SyncRemoteInferenceClient("ws://localhost:PORT")
```

### 2) Start Module (Loads Model Weights & Starts Inferencing for Continuous Modules)
```python
    client.start_module("midas_depth")
```

### 3) Get Inference Results
#### For Continuous Modules
```python
    result = client.get_current_result("hand_pose_detection")
```
#### For Single-Run Modules (No Need to Start Module)
```python
    result = client.get_result("ocr_detection")
```

### 4) Termination & Closing Client
```python
    client.stop_module("midas_depth")
    client.close()
```

## Supported Modules

| Module Name          | Description               | Type |
|----------------------|---------------------------|------|
| `obj_detection`      | YOLO-based Object Detection | Continuous |
| `hum_pose_detection` | Human Pose Estimation      | Continuous |
| `yolo_seg_track`     | YOLO-based Segmentation Tracking | Continuous |
| `hand_pose_detection`| Hand Pose Estimation       | Continuous |
| `midas_depth`        | Depth Estimation using MiDaS | Continuous |
| `gdino_inference`    | Grounding DINO Inference  | Single-Run |
| `owl_vit`           | OWL-ViT Module            | Single-Run |
| `ocr_detection`      | OCR Text Recognition       | Single-Run |
| `llava_inference`    | LLaVA-based Image Captioning | Single-Run |
| `stop_detection`     | Stop any active module     | Control |

## Contributing
Contributions are welcome! Please submit an issue or a pull request on the repository.

## License
This SDK is licensed under the MIT License.

## Author
Developed by **Kaivalya Shah**.

---
For more examples, check the provided scripts in the repository.

