# Fingers v0.1.0 🖐️

A real-time, high-performance hand detection and tracking system written in **Rust**. This project leverages MediaPipe's hand detection models via ONNX Runtime to provide a foundation for gesture-based PC control (mouse emulation, virtual steering wheels, etc.).

## 🚀 Features

* **Real-time Inference**: Optimized ONNX model execution using the `ort` (ONNX Runtime) crate.
* **High-Speed Video**: Powered by `nokhwa` for cross-platform, low-latency webcam access.
* **Custom Post-processing**: Manual implementation of anchor generation and Non-Maximum Suppression (NMS) for precise bounding box detection.
* **Minimal UI**: A lightweight visualization window using `minifb` for real-time debugging and coordinate verification.



## 🛠️ Project Structure

The project is modularized to separate hardware sensing, AI inference, and OS-level input:

| File | Responsibility |
| :--- | :--- |
| `main.rs` | Application orchestration, buffer management, and visualization. |
| `detector.rs` | MediaPipe model logic, tensor preprocessing (letterboxing), and NMS. |
| `sensor.rs` | Webcam initialization and frame decoding using `nokhwa`. |
| `controller.rs` | PC input emulation (Mouse/Keyboard) using `enigo`. |

## 🏗️ Getting Started

### Prerequisites

* **Rust**: Installed via [rustup](https://rustup.rs/).
* **ONNX Runtime**: Ensure the ONNX Runtime library is accessible on your system for the `ort` crate.
* **Webcam**: A standard USB or integrated camera.

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/rohantammara/fingers.git](https://github.com/rohantammara/fingers.git)
    cd fingers
    ```
2.  Place your MediaPipe ONNX model in the `models/` directory:
    * Expected path: `models/MediaPipeHandDetector.onnx`
3.  Run the project:
    ```bash
    cargo run --release
    ```

## 🧠 Technical Implementation

### The Detection Pipeline
1.  **Preprocessing**: The webcam frame is "letterboxed" into a 256x256 canvas to maintain aspect ratio without stretching the hand features.
2.  **Inference**: The `ort` session processes the image tensor, returning raw score and coordinate tensors.
3.  **Decoding**:
    * **Anchors**: We generate 2,944 anchors across four different scales (32x32, 16x16, 8x8, and a second 8x8 layer).
    * **BBox Regression**: Raw model outputs are transformed from anchor-relative coordinates to normalized 0.0 - 1.0 coordinates.
4.  **Non-Maximum Suppression (NMS)**: Overlapping detections