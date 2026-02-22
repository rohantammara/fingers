#![allow(unused_imports, dead_code)] //TODO: remove this after everything is in place

use enigo::Mouse;
use minifb::{Key, Window, WindowOptions};
use std::time::Duration;

mod sensor;
use sensor::webcam;
mod detector;
use detector::hand_detector;
mod controller;
use controller::input_device;

const MODEL_BYTES: &[u8] = include_bytes!("../models/MediaPipeHandDetector.onnx");
const RED: u32 = 0xFF0000;
const GREEN: u32 = 0x00FF00;
const BLUE: u32 = 0x0000FF;

fn main() -> anyhow::Result<()> {
    // Start camera
    let mut camera = webcam::setup()?;
    camera.open_stream()?;

    // Setup window
    let window_width = 960;
    let window_height = 540;

    let mut window = Window::new(
        "fingers v0.1.0",
        window_width,
        window_height,
        WindowOptions::default(),
    )?;

    // Pre-allocate the pixel buffer to avoid allocating a new vector every frame (Performance)
    let mut window_buffer = vec![0u32; window_width * window_height];

    // Limit fps to reduce CPU usage and potential instability
    let fps = 24;
    let duration_per_frame = Duration::from_micros(1000000 / fps as u64);
    window.limit_update_rate(Some(duration_per_frame));

    // Setup Input Device
    // let mut input_controller = input_device::create()?;

    // Load detector model
    let mut detector = hand_detector::HandDetector::new_embedded(MODEL_BYTES)?;

    // Define closure to convert normalised coordinates to pixel coordinates in window
    let in_window_px = |l: f32, window_dim_size: usize| {
        ((l * window_dim_size as f32) as i32).clamp(0, window_dim_size as i32 - 1)
    };

    // THE WINDOW UPDATE LOOP
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let decoded_frame = match webcam::capture_and_decode_frame(&mut camera) {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("Failed to capture or decode frame: {}", e);
                continue; // Skip this frame
            }
        };

        let resized_frame = image::imageops::resize(
            &decoded_frame,
            window_width as u32,
            window_height as u32,
            image::imageops::FilterType::Nearest,
        );

        let resized_frame_raw = resized_frame.as_raw();

        // Pixel Conversion //
        // The camera gives us a long list of u8 bytes: [R, G, B, R, G, B...]
        // The window wants u32 integers: [00RGB, 00RGB...]
        // We must map them.

        // Sanity check buffer size matches window dimensions
        if resized_frame_raw.len() != window_width * window_height * 3 {
            eprintln!(
                "Buffer size mismatch: Expected {}, got {}",
                window_width * window_height * 3,
                resized_frame_raw.len()
            );
            continue;
        }

        // Efficiently update the pre-allocated buffer
        for (i, chunk) in resized_frame_raw.chunks_exact(3).enumerate() {
            let r = chunk[0] as u32;
            let g = chunk[1] as u32;
            let b = chunk[2] as u32;
            window_buffer[i] = (r << 16) | (g << 8) | b;
        }

        // Pass the frame through the detector and get detector results
        if let Ok(Some(hands)) = detector.detect(&resized_frame) {
            for details in hands {
                // Hand Tracking //
                println!(
                    "Hand detected >> score: {} | bbox: ({} {}) ({} {}) | wrist: ({} {})",
                    details.score,
                    details.bbox.xmin,
                    details.bbox.ymin,
                    details.bbox.xmax,
                    details.bbox.ymax,
                    details.wrist.x,
                    details.wrist.y
                );

                // Convert normalized coordinates to pixel coordinates
                let [p_xmin, p_xmax, p_wrist_x] =
                    [details.bbox.xmin, details.bbox.xmax, details.wrist.x]
                        .map(|x| in_window_px(x, window_width));
                let [p_ymin, p_ymax, p_wrist_y] =
                    [details.bbox.ymin, details.bbox.ymax, details.wrist.y]
                        .map(|x| in_window_px(x, window_height));

                // --- Draw the Bounding Box (Green: 0x00FF00) ---
                let box_color = GREEN;

                // Horizontal lines (top and bottom)
                for x in p_xmin..=p_xmax {
                    window_buffer[(p_ymin as usize * window_width) + x as usize] = box_color;
                    window_buffer[(p_ymax as usize * window_width) + x as usize] = box_color;
                }
                // Vertical lines (left and right)
                for y in p_ymin..=p_ymax {
                    window_buffer[(y as usize * window_width) + p_xmin as usize] = box_color;
                    window_buffer[(y as usize * window_width) + p_xmax as usize] = box_color;
                }

                // --- Draw the Wrist Point (Blue) Dot) ---
                let dot_color = BLUE;
                let radius = 3;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let rx = p_wrist_x + dx;
                        let ry = p_wrist_y + dy;
                        if rx >= 0
                            && rx < window_width as i32
                            && ry >= 0
                            && ry < window_height as i32
                        {
                            window_buffer[(ry as usize * window_width) + rx as usize] = dot_color;
                        }
                    }
                }
            }
        }

        // Draw to Window //
        window.update_with_buffer(&window_buffer, window_width, window_height)?;
    }

    Ok(())
}
