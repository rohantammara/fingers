#![allow(unused_imports, dead_code)] //TODO: remove this after everything is in place

use enigo::Mouse;
use minifb::{Key, Window, WindowOptions};
use std::time::Duration;

mod sensor;
use sensor::webcam;
mod controller;
use controller::input_device;
mod detector;
use detector::hand_tracker;

fn main() -> anyhow::Result<()> {
    let mut camera = webcam::setup()?;
    let (width, height) = webcam::start_stream(&mut camera)?;

    // Setup Window (The "Screen") //
    let mut window = Window::new(
        "Fingers (Press ESC to exit)",
        width,
        height,
        WindowOptions::default(),
    )?;

    // Pre-allocate the pixel buffer to avoid allocating a new vector every frame (Performance)
    let mut buffer_u32 = vec![0u32; width * height];

    // Limit to ~60 fps to reduce CPU usage and potential instability
    window.limit_update_rate(Some(Duration::from_micros(16600)));

    // Setup Input Device
    // let mut input_controller = input_device::create()?;

    // Setup Hand Gesture Detector
    let mut detector = hand_tracker::HandGestureDetector::new("MediaPipeHandDetector.onnx")?;

    // The Input Feed Loop
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let decoded = match webcam::capture_and_decode_frame(&mut camera) {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("Failed to capture or decode frame: {}", e);
                continue; // Skip this frame
            }
        };

        let raw_data = decoded.as_raw();

        // Pixel Conversion //
        // The camera gives us a long list of u8 bytes: [R, G, B, R, G, B...]
        // The window wants u32 integers: [00RGB, 00RGB...]
        // We must map them.

        // Sanity check buffer size
        if raw_data.len() != width * height * 3 {
            eprintln!(
                "Buffer size mismatch: Expected {}, got {}",
                width * height * 3,
                raw_data.len()
            );
            continue;
        }

        // Efficiently update the pre-allocated buffer
        for (i, chunk) in raw_data.chunks_exact(3).enumerate() {
            let r = chunk[0] as u32;
            let g = chunk[1] as u32;
            let b = chunk[2] as u32;
            buffer_u32[i] = (r << 16) | (g << 8) | b;
        }

        // // B. Hand Tracking & Logic (The "Brain")
        if let Ok(Some(details)) = detector.detect(&decoded) {
            println!(
                "Gesture: {:?} (Score: {:.2}) at ({:.2}, {:.2})",
                details.gesture, details.score, details.x, details.y
            );
        }

        // Draw to Window
        window.update_with_buffer(&buffer_u32, width, height)?;
    }

    Ok(())
}
