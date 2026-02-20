#![allow(unused_imports, dead_code)] //TODO: remove this after everything is in place

use enigo::Mouse;
use minifb::{Key, Window, WindowOptions};
use std::time::Duration;

mod sensor;
use sensor::webcam;
mod controller;
use controller::input_device;
mod detector;
use detector::{gesture_detector, gesture_detector::Gesture, hand_detector};

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
    let mut buffer_u32 = vec![0u32; window_width * window_height];

    // Limit fps to reduce CPU usage and potential instability
    let fps = 30;
    let duration_per_frame = Duration::from_micros(1000000 / fps as u64);
    window.limit_update_rate(Some(duration_per_frame));

    // Setup Input Device
    // let mut input_controller = input_device::create()?;

    // Load detector model
    let mut detector = hand_detector::HandDetector::new("models/MediaPipeHandDetector.onnx")?;

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

        let raw_data = resized_frame.as_raw();

        // Pixel Conversion //
        // The camera gives us a long list of u8 bytes: [R, G, B, R, G, B...]
        // The window wants u32 integers: [00RGB, 00RGB...]
        // We must map them.

        // Sanity check buffer size matches window dimensions
        if raw_data.len() != window_width * window_height * 3 {
            eprintln!(
                "Buffer size mismatch: Expected {}, got {}",
                window_width * window_height * 3,
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

        // Pass the frame through the detector and get detector results
        if let Ok(Some(details)) = detector.detect(&resized_frame) {
            // Hand Tracking //

            println!(
                "Normal/Adjusted: bbox: {} {} {} {} | wrist: {} {}",
                details.bbox.xmin,
                details.bbox.ymin,
                details.bbox.xmax,
                details.bbox.ymax,
                details.wrist.x,
                details.wrist.y
            );

            // Convert normalized coordinates to pixel coordinates
            let p_xmin = ((details.bbox.xmin * window_width as f32) as i32)
                .clamp(0, window_width as i32 - 1);
            let p_ymin = ((details.bbox.ymin * window_height as f32) as i32)
                .clamp(0, window_height as i32 - 1);
            let p_xmax = ((details.bbox.xmax * window_width as f32) as i32)
                .clamp(0, window_width as i32 - 1);
            let p_ymax = ((details.bbox.ymax * window_height as f32) as i32)
                .clamp(0, window_height as i32 - 1);

            let p_wrist_x =
                ((details.wrist.x * window_width as f32) as i32).clamp(0, window_width as i32 - 1);
            let p_wrist_y = ((details.wrist.y * window_height as f32) as i32)
                .clamp(0, window_height as i32 - 1);

            println!(
                "Pixel scaled: bbox: {} {} {} {} | wrist: {} {}",
                p_xmin, p_ymin, p_xmax, p_ymax, p_wrist_x, p_wrist_y
            );

            // --- Draw the Bounding Box (Green: 0x00FF00) ---
            let box_color = 0x00FF00;

            // Horizontal lines (top and bottom)
            for x in p_xmin..=p_xmax {
                buffer_u32[(p_ymin as usize * window_width) + x as usize] = box_color;
                buffer_u32[(p_ymax as usize * window_width) + x as usize] = box_color;
            }
            // Vertical lines (left and right)
            for y in p_ymin..=p_ymax {
                buffer_u32[(y as usize * window_width) + p_xmin as usize] = box_color;
                buffer_u32[(y as usize * window_width) + p_xmax as usize] = box_color;
            }

            // --- Draw the Wrist Point (Blue) Dot) ---
            let dot_color = 0x0000FF;
            let radius = 3;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let rx = p_wrist_x + dx;
                    let ry = p_wrist_y + dy;
                    if rx >= 0 && rx < window_width as i32 && ry >= 0 && ry < window_height as i32 {
                        buffer_u32[(ry as usize * window_width) + rx as usize] = dot_color;
                    }
                }
            }

            // Draw to Window //
            window.update_with_buffer(&buffer_u32, window_width, window_height)?;
        }
    }

    Ok(())
}
